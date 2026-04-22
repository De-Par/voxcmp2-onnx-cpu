#!/usr/bin/env python3
"""Summarize dtype Cast patterns in exported ONNX graphs.

The tool is intentionally graph-only: it does not run inference and does not
change model files. Use it before and after re-exporting production artifacts
to verify that wrapper dtype cleanup actually reduces Cast churn.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

CAST_OPS = {"Cast", "CastLike"}
PING_PONG_DTYPES = {TensorProto.FLOAT, TensorProto.BFLOAT16}


def _dtype_name(data_type: int | None) -> str:
    if data_type is None:
        return "UNKNOWN"
    try:
        return TensorProto.DataType.Name(data_type)
    except ValueError:
        return f"UNKNOWN_{data_type}"


def _cast_to(node: onnx.NodeProto) -> int | None:
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _value_info_dtype(value_info: onnx.ValueInfoProto) -> int | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    tensor_type = value_info.type.tensor_type
    elem_type = int(tensor_type.elem_type)
    return elem_type or None


def _value_dtype_map(model: onnx.ModelProto) -> dict[str, int]:
    dtypes: dict[str, int] = {}
    for initializer in model.graph.initializer:
        dtypes[initializer.name] = int(initializer.data_type)
    for value_info in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        dtype = _value_info_dtype(value_info)
        if dtype is not None:
            dtypes[value_info.name] = dtype
    for node in model.graph.node:
        if node.op_type == "Cast":
            to_dtype = _cast_to(node)
            if to_dtype is not None:
                for output in node.output:
                    dtypes[output] = to_dtype
    return dtypes


def _node_name(node: onnx.NodeProto, index: int) -> str:
    return node.name or (node.output[0] if node.output else f"<unnamed:{index}>")


def _node_key(path: Path, index: int, node: onnx.NodeProto) -> dict[str, Any]:
    return {
        "model": str(path),
        "index": index,
        "name": _node_name(node, index),
        "op_type": node.op_type,
        "to": _dtype_name(_cast_to(node)) if node.op_type == "Cast" else "CastLike",
    }


def analyze_casts(path: Path) -> dict[str, Any]:
    """Return a compact Cast-pattern summary for one ONNX model."""

    model = onnx.load(str(path), load_external_data=False)
    value_dtypes = _value_dtype_map(model)
    graph_inputs = {item.name for item in model.graph.input}
    graph_outputs = {item.name for item in model.graph.output}
    initializer_dtypes = {initializer.name: int(initializer.data_type) for initializer in model.graph.initializer}

    cast_outputs: dict[str, tuple[int, onnx.NodeProto]] = {}
    cast_to_counts: Counter[str] = Counter()
    direct_chains: list[dict[str, Any]] = []
    redundant_casts: list[dict[str, Any]] = []
    ping_pong: list[dict[str, Any]] = []
    storage_only: list[dict[str, Any]] = []
    boundary_casts: list[dict[str, Any]] = []
    castlike_nodes: list[dict[str, Any]] = []

    for index, node in enumerate(model.graph.node):
        if node.op_type not in CAST_OPS:
            continue
        if node.op_type == "CastLike":
            castlike_nodes.append(_node_key(path, index, node))
            cast_to_counts["CastLike"] += 1
            continue

        to_dtype = _cast_to(node)
        cast_to_counts[_dtype_name(to_dtype)] += 1
        input_name = node.input[0] if node.input else ""
        input_dtype = value_dtypes.get(input_name)
        key = _node_key(path, index, node)
        key["input_dtype"] = _dtype_name(input_dtype)

        if input_dtype == to_dtype:
            redundant_casts.append(key)

        if input_name in cast_outputs:
            parent_index, parent_node = cast_outputs[input_name]
            parent_to = _cast_to(parent_node)
            direct_chain = {
                "parent": _node_key(path, parent_index, parent_node),
                "child": key,
            }
            direct_chains.append(direct_chain)
            if parent_to == to_dtype:
                redundant_casts.append(key | {"reason": "Cast-to-Cast with same target dtype"})
            if {parent_to, to_dtype} == PING_PONG_DTYPES:
                ping_pong.append(direct_chain)

        if initializer_dtypes.get(input_name) == TensorProto.BFLOAT16 and to_dtype == TensorProto.FLOAT:
            storage_only.append(key | {"initializer": input_name})

        output_names = set(node.output)
        if input_name in graph_inputs or output_names & graph_outputs:
            boundary_casts.append(key)

        for output in node.output:
            cast_outputs[output] = (index, node)

    total_cast_nodes = sum(1 for node in model.graph.node if node.op_type == "Cast")
    return {
        "path": str(path),
        "exists": path.exists(),
        "nodes": len(model.graph.node),
        "cast_nodes": total_cast_nodes,
        "castlike_nodes": len(castlike_nodes),
        "cast_to": dict(sorted(cast_to_counts.items())),
        "redundant_casts": {
            "count": len(redundant_casts),
            "samples": redundant_casts[:20],
        },
        "direct_cast_chains": {
            "count": len(direct_chains),
            "samples": direct_chains[:20],
        },
        "fp32_bf16_ping_pong": {
            "count": len(ping_pong),
            "samples": ping_pong[:20],
        },
        "storage_only_bf16_to_fp32": {
            "count": len(storage_only),
            "samples": storage_only[:20],
        },
        "unavoidable_precision_boundaries": {
            "count": len(boundary_casts),
            "samples": boundary_casts[:20],
        },
        "exporter_artifacts": {
            "castlike_count": len(castlike_nodes),
            "castlike_samples": castlike_nodes[:20],
        },
    }


def _discover_onnx(root: Path | None) -> list[Path]:
    if root is None or not root.exists():
        return []
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.onnx"))


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path.expanduser())
    return result


def _analyze_many(paths: list[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in _dedupe(paths):
        if not path.exists():
            reports.append({"path": str(path), "exists": False, "error": "missing"})
            continue
        reports.append(analyze_casts(path))
    return reports


def _totals(items: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for item in items:
        if not item.get("exists", True):
            continue
        totals["models"] += 1
        totals["cast_nodes"] += int(item.get("cast_nodes", 0))
        totals["castlike_nodes"] += int(item.get("castlike_nodes", 0))
        totals["redundant_casts"] += int(item.get("redundant_casts", {}).get("count", 0))
        totals["direct_cast_chains"] += int(item.get("direct_cast_chains", {}).get("count", 0))
        totals["fp32_bf16_ping_pong"] += int(item.get("fp32_bf16_ping_pong", {}).get("count", 0))
        totals["storage_only_bf16_to_fp32"] += int(item.get("storage_only_bf16_to_fp32", {}).get("count", 0))
        totals["unavoidable_precision_boundaries"] += int(
            item.get("unavoidable_precision_boundaries", {}).get("count", 0)
        )
    return dict(sorted(totals.items()))


def _load_profile_cast_hotspots(profile_json: Path | None) -> list[dict[str, Any]]:
    if profile_json is None or not profile_json.exists():
        return []
    data = json.loads(profile_json.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        hotspots = data.get("cast_hotspots", [])
        return hotspots if isinstance(hotspots, list) else []
    return []


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    before_paths = [Path(item) for item in args.before] + _discover_onnx(args.before_root)
    after_paths = [Path(item) for item in args.after] + _discover_onnx(args.after_root)
    before = _analyze_many(before_paths)
    after = _analyze_many(after_paths)
    return {
        "before": before,
        "after": after,
        "summary": {
            "before": _totals(before),
            "after": _totals(after),
            "profile_cast_hotspots": _load_profile_cast_hotspots(args.profile_json),
        },
        "classification": {
            "redundant_cast_chains": "direct Cast->Cast edges and same-dtype Cast nodes",
            "fp32_bf16_ping_pong": "direct FLOAT<->BFLOAT16 Cast pairs",
            "unavoidable_precision_boundaries": "Cast nodes connected to graph inputs or outputs",
            "exporter_artifacts": "CastLike and other generated Cast-like nodes that are not model math",
        },
    }


def _markdown_table(title: str, totals: dict[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| metric | value |", "|---|---:|"]
    for key, value in totals.items():
        lines.append(f"| `{key}` | {value} |")
    if not totals:
        lines.append("| no models found | 0 |")
    lines.append("")
    return lines


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Dtype Cast Summary",
        "",
        "Graph-only Cast summary for production FP32/BF16 ONNX artifacts.",
        "",
        *_markdown_table("Before", summary["before"]),
        *_markdown_table("After", summary["after"]),
        "## Ranked Cast Hotspots",
        "",
    ]
    hotspots = summary.get("profile_cast_hotspots", [])
    if hotspots:
        lines.extend(["| rank | module | op | node | total_ms |", "|---:|---|---|---|---:|"])
        for rank, item in enumerate(hotspots[:20], start=1):
            total_ms = float(item.get("total_ms", 0.0))
            lines.append(
                f"| {rank} | `{item.get('module', '')}` | `{item.get('op_type', '')}` | "
                f"`{item.get('node_name', '')}` | {total_ms:.3f} |"
            )
    else:
        lines.append("No profile JSON with `cast_hotspots` was provided.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Cast/CastLike dtype patterns in exported ONNX graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--before", nargs="*", default=[], help="Explicit pre-cleanup ONNX files to inspect.")
    parser.add_argument("--after", nargs="*", default=[], help="Explicit post-cleanup ONNX files to inspect.")
    parser.add_argument("--before-root", type=Path, default=None, help="Directory containing pre-cleanup ONNX files.")
    parser.add_argument(
        "--after-root",
        type=Path,
        default=Path("models/onnx"),
        help="Directory containing post-cleanup ONNX files.",
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=None,
        help="Optional parsed ORT profile JSON from tools/profile/parse_ort_profile.py.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("artifacts/reports/dtype_cleanup_casts.json"),
        help="Path for the machine-readable summary.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("artifacts/reports/dtype_cleanup_casts.md"),
        help="Path for the short Markdown summary.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_report(args)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.markdown_report)
    print(f"json saved: {args.json_report}")
    print(f"markdown saved: {args.markdown_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
