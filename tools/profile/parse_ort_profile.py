#!/usr/bin/env python3
"""Parse ONNX Runtime Chrome-trace profiles into ranked hotspot reports"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("artifacts/profile")
CAST_KEYWORDS = ("cast", "castlike")
CACHE_KEYWORDS = (
    "cache",
    "past",
    "present",
    "base_k",
    "base_v",
    "residual_k",
    "residual_v",
    "key_cache",
    "value_cache",
    "kv",
)
CACHE_OP_HINTS = {"Concat", "Slice", "Gather", "ScatterND", "Expand", "Unsqueeze"}


def _load_profile(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"ORT profile must be a JSON event list: {path}")
    return [event for event in data if isinstance(event, dict)]


def _module_from_path(path: Path) -> str:
    text = path.name.lower()
    for module in ("audio_encoder", "audio_decoder", "decode_chunk", "decode_step", "prefill"):
        if module in text:
            return module
    if "audio_vae_encoder" in text:
        return "audio_encoder"
    if "audio_vae_decoder" in text:
        return "audio_decoder"
    return "unknown"


def _clean_node_name(name: str) -> str:
    cleaned = re.sub(r"_(kernel|fence)_time$", "", name)
    return cleaned or name


def _event_op_type(event: dict[str, Any]) -> str:
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    op_name = args.get("op_name") or args.get("op_type")
    if op_name:
        return str(op_name)
    name = str(event.get("name", ""))
    return name.split("_", 1)[0] if "_" in name else name or "unknown"


def _is_node_event(event: dict[str, Any]) -> bool:
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    return event.get("cat") == "Node" or "op_name" in args or str(event.get("name", "")).endswith("_kernel_time")


def _event_text(event: dict[str, Any], node_name: str, op_type: str, module: str) -> str:
    return json.dumps(
        {
            "module": module,
            "node": node_name,
            "op_type": op_type,
            "args": event.get("args", {}),
        },
        sort_keys=True,
        default=str,
    ).lower()


def _cache_reason(event_text: str, op_type: str, module: str) -> str | None:
    for keyword in CACHE_KEYWORDS:
        if keyword in event_text:
            return f"matched keyword {keyword!r}"
    if module in {"decode_chunk", "decode_step"} and op_type in CACHE_OP_HINTS:
        return f"{module} {op_type} can participate in explicit cache/state movement"
    return None


def _code_sites(module: str, op_type: str, node_name: str, event_text: str) -> list[dict[str, str]]:
    sites: list[dict[str, str]] = []
    lowered = f"{node_name} {op_type} {event_text}".lower()
    if module == "prefill":
        sites.append(
            {
                "file": "src/export/export_prefill.py",
                "symbol": "VoxCPM2PrefillWrapper.forward",
                "reason": "prefill graph is exported from this tensor-only wrapper",
            }
        )
        if "cache" in lowered or op_type in {"Concat", "Unsqueeze", "Gather"}:
            sites.append(
                {
                    "file": "src/export/export_prefill.py",
                    "symbol": "VoxCPM2PrefillWrapper._stack_cache",
                    "reason": "prefill returns explicit base/residual K/V cache tensors",
                }
            )
    elif module in {"decode_chunk", "decode_step"}:
        wrapper_file = (
            "src/export/export_decode_chunk.py" if module == "decode_chunk" else "src/export/export_decode_step.py"
        )
        wrapper_symbol = (
            "VoxCPM2DecodeChunkWrapper.forward" if module == "decode_chunk" else "VoxCPM2DecodeStepWrapper.forward"
        )
        sites.append(
            {
                "file": wrapper_file,
                "symbol": wrapper_symbol,
                "reason": f"{module} graph is exported from this wrapper",
            }
        )
        if op_type in {"Attention", "MatMul", "Gemm", "Softmax"} or "attention" in lowered:
            sites.append(
                {
                    "file": "src/export/export_decode_step.py",
                    "symbol": "VoxCPM2DecodeStepWrapper._attention_step",
                    "reason": "chunked decode reuses exact one-step attention math here",
                }
            )
        if "cache" in lowered or op_type in CACHE_OP_HINTS:
            sites.append(
                {
                    "file": "src/runtime/pipeline.py",
                    "symbol": "VoxCPM2OnnxPipeline.synthesize_with_metadata",
                    "reason": "host loop passes explicit cache/state tensors between decode chunks",
                }
            )
    elif module == "audio_encoder":
        sites.append(
            {
                "file": "src/runtime/pipeline.py",
                "symbol": "VoxCPM2OnnxPipeline._encode_wav",
                "reason": "runtime invokes AudioVAEEncoder for reference/prompt audio",
            }
        )
    elif module == "audio_decoder":
        sites.append(
            {
                "file": "src/runtime/pipeline.py",
                "symbol": "VoxCPM2OnnxPipeline.synthesize_with_metadata",
                "reason": "runtime invokes AudioVAEDecoder after generated feature layout conversion",
            }
        )
    else:
        sites.append(
            {
                "file": "src/runtime/pipeline.py",
                "symbol": "VoxCPM2OnnxPipeline",
                "reason": "unknown profile filename; inspect the session path and runtime call site",
            }
        )
    return sites


def _add_metric(bucket: dict[str, Any], duration_us: float) -> None:
    bucket["calls"] += 1
    bucket["total_us"] += duration_us
    bucket["max_us"] = max(bucket["max_us"], duration_us)


def parse_profiles(paths: list[Path], *, top_n: int = 20) -> dict[str, Any]:
    node_buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    op_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    cast_buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    cache_buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    profile_summaries = []

    for path in paths:
        module = _module_from_path(path)
        events = _load_profile(path)
        node_events = 0
        total_us = 0.0
        for event in events:
            if not _is_node_event(event) or "dur" not in event:
                continue
            duration_us = float(event.get("dur") or 0.0)
            if duration_us <= 0.0:
                continue
            node_events += 1
            total_us += duration_us
            node_name = _clean_node_name(str(event.get("name", "<unnamed>")))
            op_type = _event_op_type(event)
            event_text = _event_text(event, node_name, op_type, module)

            node_key = (module, node_name, op_type)
            node_bucket = node_buckets.setdefault(
                node_key,
                {
                    "module": module,
                    "node_name": node_name,
                    "op_type": op_type,
                    "calls": 0,
                    "total_us": 0.0,
                    "max_us": 0.0,
                    "source_profiles": set(),
                    "code_sites": _code_sites(module, op_type, node_name, event_text),
                },
            )
            _add_metric(node_bucket, duration_us)
            node_bucket["source_profiles"].add(str(path))

            op_key = (module, op_type)
            op_bucket = op_buckets.setdefault(
                op_key,
                {"module": module, "op_type": op_type, "calls": 0, "total_us": 0.0, "max_us": 0.0},
            )
            _add_metric(op_bucket, duration_us)

            if any(keyword in event_text for keyword in CAST_KEYWORDS):
                cast_bucket = cast_buckets.setdefault(node_key, dict(node_bucket, total_us=0.0, calls=0, max_us=0.0))
                _add_metric(cast_bucket, duration_us)

            cache_reason = _cache_reason(event_text, op_type, module)
            if cache_reason is not None:
                cache_bucket = cache_buckets.setdefault(
                    node_key,
                    dict(node_bucket, total_us=0.0, calls=0, max_us=0.0, cache_reason=cache_reason),
                )
                _add_metric(cache_bucket, duration_us)

        profile_summaries.append(
            {
                "path": str(path),
                "module": module,
                "events": len(events),
                "node_events": node_events,
                "node_total_ms": round(total_us / 1000.0, 6),
            }
        )

    return {
        "schema_version": 1,
        "profiles": profile_summaries,
        "top_nodes": _rank_buckets(node_buckets.values(), top_n),
        "top_op_types": _rank_buckets(op_buckets.values(), top_n),
        "cast_hotspots": _rank_buckets(cast_buckets.values(), top_n),
        "cache_hotspots": _rank_buckets(cache_buckets.values(), top_n),
        "shortlist": _shortlist(node_buckets.values(), op_buckets.values(), cache_buckets.values()),
    }


def _rank_buckets(buckets: Any, top_n: int) -> list[dict[str, Any]]:
    ranked = []
    for bucket in sorted(buckets, key=lambda item: float(item["total_us"]), reverse=True)[:top_n]:
        item = dict(bucket)
        item["total_ms"] = round(float(item.pop("total_us")) / 1000.0, 6)
        item["max_ms"] = round(float(item.pop("max_us")) / 1000.0, 6)
        item["mean_ms"] = round(item["total_ms"] / int(item["calls"]), 6) if item["calls"] else 0.0
        if "source_profiles" in item:
            item["source_profiles"] = sorted(item["source_profiles"])
        ranked.append(item)
    return ranked


def _shortlist(nodes: Any, ops: Any, cache_nodes: Any) -> list[dict[str, str]]:
    node_list = sorted(nodes, key=lambda item: float(item["total_us"]), reverse=True)
    op_list = sorted(ops, key=lambda item: float(item["total_us"]), reverse=True)
    cache_list = sorted(cache_nodes, key=lambda item: float(item["total_us"]), reverse=True)
    items: list[dict[str, str]] = []
    if op_list:
        top = op_list[0]
        items.append(
            {
                "reason": f"{top['module']}::{top['op_type']} dominates total profiled node time",
                "next_step": "inspect top_nodes for exact nodes before changing graph/export math",
            }
        )
    if cache_list:
        top = cache_list[0]
        items.append(
            {
                "reason": f"cache/state movement hotspot: {top['module']}::{top['node_name']}",
                "next_step": "review explicit cache tensor contract in decode chunk wrapper and runtime loop",
            }
        )
    cast_nodes = [item for item in node_list if item["op_type"] == "Cast" or "cast" in item["node_name"].lower()]
    if cast_nodes:
        top = cast_nodes[0]
        items.append(
            {
                "reason": f"Cast hotspot: {top['module']}::{top['node_name']}",
                "next_step": "check whether this comes from wrapper dtype guards before considering removal",
            }
        )
    if len(op_list) > 1:
        top = op_list[1]
        items.append(
            {
                "reason": f"second-highest op family is {top['module']}::{top['op_type']}",
                "next_step": "compare with top_nodes and code_sites to decide if it is model math or glue",
            }
        )
    if node_list:
        top = node_list[0]
        items.append(
            {
                "reason": f"hottest single node is {top['module']}::{top['node_name']}",
                "next_step": "treat as the first candidate for focused parity-safe investigation",
            }
        )
    return items[:5]


def _profile_paths(args: argparse.Namespace) -> list[Path]:
    paths = [path.expanduser() for path in args.profile_files]
    for directory in args.profile_dirs:
        paths.extend(sorted(directory.expanduser().glob("*.json")))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise ValueError("no ORT profile JSON files found")
    return unique


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_ms(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _write_markdown(path: Path, report: dict[str, Any], *, top_n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ONNX Runtime Profile Hotspots",
        "",
        f"- profile files: `{len(report['profiles'])}`",
        f"- top N: `{top_n}`",
        "",
        "## Shortlist",
        "",
    ]
    for index, item in enumerate(report["shortlist"], start=1):
        lines.append(f"{index}. {item['reason']}. Next: {item['next_step']}.")
    if not report["shortlist"]:
        lines.append("- No node events found.")

    _append_table(
        lines,
        "Top Nodes By Latency",
        ["rank", "module", "op", "node", "calls", "total ms", "mean ms", "max ms", "code sites"],
        [
            [
                index,
                item["module"],
                item["op_type"],
                item["node_name"],
                item["calls"],
                _format_ms(item["total_ms"]),
                _format_ms(item["mean_ms"]),
                _format_ms(item["max_ms"]),
                _site_text(item.get("code_sites", [])),
            ]
            for index, item in enumerate(report["top_nodes"], start=1)
        ],
    )
    _append_table(
        lines,
        "Top Op Types By Total Latency",
        ["rank", "module", "op", "calls", "total ms", "mean ms", "max ms"],
        [
            [
                index,
                item["module"],
                item["op_type"],
                item["calls"],
                _format_ms(item["total_ms"]),
                _format_ms(item["mean_ms"]),
                _format_ms(item["max_ms"]),
            ]
            for index, item in enumerate(report["top_op_types"], start=1)
        ],
    )
    _append_table(
        lines,
        "Cast Hotspots",
        ["rank", "module", "op", "node", "calls", "total ms", "code sites"],
        [
            [
                index,
                item["module"],
                item["op_type"],
                item["node_name"],
                item["calls"],
                _format_ms(item["total_ms"]),
                _site_text(item.get("code_sites", [])),
            ]
            for index, item in enumerate(report["cast_hotspots"], start=1)
        ],
    )
    _append_table(
        lines,
        "Cache-Related Hotspots",
        ["rank", "module", "op", "node", "calls", "total ms", "reason", "code sites"],
        [
            [
                index,
                item["module"],
                item["op_type"],
                item["node_name"],
                item["calls"],
                _format_ms(item["total_ms"]),
                item.get("cache_reason", ""),
                _site_text(item.get("code_sites", [])),
            ]
            for index, item in enumerate(report["cache_hotspots"], start=1)
        ],
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_table(lines: list[str], title: str, headers: list[str], rows: list[list[Any]]) -> None:
    lines.extend(["", f"## {title}", ""])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    if not rows:
        lines.append("| " + " | ".join("-" for _ in headers) + " |")


def _site_text(sites: list[dict[str, str]]) -> str:
    return "; ".join(f"{site['file']}::{site['symbol']}" for site in sites[:3])


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _profile_paths(args)
    report = parse_profiles(paths, top_n=args.top_n)
    report["profile_files"] = [str(path) for path in paths]
    if args.json_report:
        _write_json(args.json_report.expanduser(), report)
    if args.markdown_report:
        _write_markdown(args.markdown_report.expanduser(), report, top_n=args.top_n)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse ONNX Runtime profiling JSON files and rank VoxCPM2 hotspots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile-files", nargs="*", type=Path, default=[], help="ORT profile JSON files.")
    parser.add_argument(
        "--profile-dirs",
        nargs="*",
        type=Path,
        default=[DEFAULT_OUTPUT_DIR / "profiles"],
        help="Directories containing ORT profile JSON files.",
    )
    parser.add_argument("--json-report", type=Path, help="Optional parsed JSON report path.")
    parser.add_argument("--markdown-report", type=Path, help="Optional Markdown hotspot report path.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of ranked entries per section.")
    return parser


def main() -> int:
    try:
        report = run(_parser().parse_args())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    print(
        json.dumps(
            {
                "profiles": len(report["profiles"]),
                "top_nodes": len(report["top_nodes"]),
                "top_op_types": len(report["top_op_types"]),
                "cast_hotspots": len(report["cast_hotspots"]),
                "cache_hotspots": len(report["cache_hotspots"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
