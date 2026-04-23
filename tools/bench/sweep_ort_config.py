#!/usr/bin/env python3
"""Sweep ONNX Runtime CPU session settings for FP32 and BF16 VoxCPM2 artifacts"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
Precision = Literal["fp32", "bf16"]
CASE_IDS = ("text_only_short", "voice_design_short", "controllable_clone_short")
MEMORY_PROFILES = ("ort_default", "explicit_on", "arena_off", "mem_reuse_off", "all_off")


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _runtime_imports():
    _ensure_repo_root_on_path()

    from src.runtime.pipeline import VoxCPM2OnnxPipeline, VoxCPM2RuntimeConfig
    from src.runtime.session_factory import EXECUTION_MODE_CHOICES, GRAPH_OPTIMIZATION_CHOICES, OnnxModelPaths

    return VoxCPM2OnnxPipeline, VoxCPM2RuntimeConfig, OnnxModelPaths, GRAPH_OPTIMIZATION_CHOICES, EXECUTION_MODE_CHOICES


def _bench_helpers():
    _ensure_repo_root_on_path()

    from tools.bench.run_benchmarks import BenchmarkCase, _make_reference_wav

    return BenchmarkCase, _make_reference_wav


@dataclass(frozen=True)
class MemoryOptions:
    name: str
    enable_mem_pattern: bool | None
    enable_cpu_mem_arena: bool | None
    enable_mem_reuse: bool | None


@dataclass(frozen=True)
class SessionConfig:
    graph_optimization_level: str
    execution_mode: str
    intra_op_num_threads: int | None
    inter_op_num_threads: int | None
    memory: MemoryOptions

    @property
    def config_id(self) -> str:
        intra = "default" if self.intra_op_num_threads is None else str(self.intra_op_num_threads)
        inter = "default" if self.inter_op_num_threads is None else str(self.inter_op_num_threads)
        return (
            f"g{self.graph_optimization_level}_e{self.execution_mode}_intra{intra}_inter{inter}_mem{self.memory.name}"
        )

    def runtime_kwargs(self) -> dict[str, Any]:
        return {
            "graph_optimization_level": self.graph_optimization_level,
            "execution_mode": self.execution_mode,
            "log_severity_level": "error",
            "intra_op_num_threads": self.intra_op_num_threads,
            "inter_op_num_threads": self.inter_op_num_threads,
            "enable_mem_pattern": self.memory.enable_mem_pattern,
            "enable_cpu_mem_arena": self.memory.enable_cpu_mem_arena,
            "enable_mem_reuse": self.memory.enable_mem_reuse,
        }

    def as_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["config_id"] = self.config_id
        return data


def _memory_options(name: str) -> MemoryOptions:
    profiles = {
        "ort_default": MemoryOptions(name, None, None, None),
        "explicit_on": MemoryOptions(name, True, True, True),
        "arena_off": MemoryOptions(name, True, False, True),
        "mem_reuse_off": MemoryOptions(name, True, True, False),
        "all_off": MemoryOptions(name, False, False, False),
    }
    return profiles[name]


def _parse_thread_values(values: list[str]) -> list[int | None]:
    parsed: list[int | None] = []
    for value in values:
        if value == "default":
            parsed.append(None)
            continue
        parsed.append(int(value))
    return parsed


def _preset_configs(args: argparse.Namespace) -> list[SessionConfig]:
    explicit_on = _memory_options("explicit_on")
    ort_default = _memory_options("ort_default")
    if args.config_preset == "recommended":
        return [SessionConfig("all", "sequential", 8, 1, explicit_on)]
    if args.config_preset == "focused":
        return [
            SessionConfig("disable", "sequential", None, None, ort_default),
            SessionConfig("extended", "sequential", 8, 1, explicit_on),
            SessionConfig("all", "sequential", 8, 1, explicit_on),
            SessionConfig("all", "sequential", 4, 1, explicit_on),
            SessionConfig("all", "sequential", 0, 1, explicit_on),
            SessionConfig("all", "parallel", 8, 1, explicit_on),
            SessionConfig("all", "sequential", 8, 1, _memory_options("arena_off")),
        ]
    graph_levels = args.graph_optimization_levels
    execution_modes = args.execution_modes
    intra_values = _parse_thread_values(args.intra_op_threads)
    inter_values = _parse_thread_values(args.inter_op_threads)
    memory_profiles = [_memory_options(name) for name in args.memory_profiles]
    return [
        SessionConfig(graph_level, execution_mode, intra, inter, memory)
        for graph_level, execution_mode, intra, inter, memory in product(
            graph_levels,
            execution_modes,
            intra_values,
            inter_values,
            memory_profiles,
        )
    ]


def _paths_for_precision(precision: Precision):
    _, _, OnnxModelPaths, *_ = _runtime_imports()
    if precision == "fp32":
        return OnnxModelPaths()
    root = REPO_ROOT / "models" / "onnx" / "bf16"
    return OnnxModelPaths(
        audio_encoder=root / "audio_vae_encoder" / "audio_vae_encoder.onnx",
        audio_decoder=root / "audio_vae_decoder" / "audio_vae_decoder.onnx",
        prefill=root / "prefill" / "voxcpm2_prefill.onnx",
        decode_chunk=root / "decode_chunk" / "voxcpm2_decode_chunk.onnx",
    )


def _cases() -> dict[str, object]:
    BenchmarkCase, _ = _bench_helpers()
    return {
        "text_only_short": BenchmarkCase("text_only_short", "text_only", "Hello from VoxCPM2."),
        "voice_design_short": BenchmarkCase(
            "voice_design_short",
            "voice_design",
            "Hello from VoxCPM2.",
            voice_design="pretty girl with sugar voice, slow",
        ),
        "controllable_clone_short": BenchmarkCase(
            "controllable_clone_short",
            "controllable_clone",
            "Hello from VoxCPM2.",
            needs_reference=True,
        ),
    }


def _preload_sessions(pipeline: object, selected_cases: list[object]) -> None:
    _ = pipeline.sessions.prefill
    _ = pipeline.sessions.decode_chunk
    _ = pipeline.sessions.audio_decoder
    if any(case.needs_reference for case in selected_cases):
        _ = pipeline.sessions.audio_encoder


def _percentile(values: list[float], percentile: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return round(clean[0], 6)
    pos = (len(clean) - 1) * percentile / 100.0
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))
    if lower == upper:
        return round(clean[lower], 6)
    weight = pos - lower
    return round(clean[lower] * (1.0 - weight) + clean[upper] * weight, 6)


def _stats(values: list[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None]
    return {
        "min": round(min(clean), 6) if clean else None,
        "max": round(max(clean), 6) if clean else None,
        "mean": round(float(np.mean(clean)), 6) if clean else None,
        "p50": _percentile(clean, 50),
        "p90": _percentile(clean, 90),
    }


def _audio_stats(waveform: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    mono = np.nan_to_num(np.asarray(waveform, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=-1.0)
    return {
        "sample_rate": int(sample_rate),
        "samples": int(mono.size),
        "duration_seconds": round(float(mono.size / sample_rate), 6) if sample_rate else 0.0,
        "peak": round(float(np.max(np.abs(mono))) if mono.size else 0.0, 8),
        "rms": round(float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0, 8),
    }


def _run_precision_config(
    args: argparse.Namespace,
    precision: Precision,
    session_config: SessionConfig,
    selected_cases: list[object],
    reference_wav: Path,
) -> tuple[float | None, list[dict[str, Any]]]:
    VoxCPM2OnnxPipeline, *_ = _runtime_imports()
    records: list[dict[str, Any]] = []
    load_start = time.perf_counter()
    try:
        pipeline = VoxCPM2OnnxPipeline.from_default_artifacts(
            model_path=args.model_path,
            local_files_only=args.local_files_only,
            onnx_paths=_paths_for_precision(precision),
            **session_config.runtime_kwargs(),
        )
        pipeline.validate()
        if args.preload_sessions:
            _preload_sessions(pipeline, selected_cases)
        load_seconds = round(time.perf_counter() - load_start, 6)
    except Exception as exc:  # noqa: BLE001 - sweep reports failed configs instead of aborting the full matrix.
        error = f"{type(exc).__name__}: {exc}"
        for case in selected_cases:
            records.append(
                {
                    "ok": False,
                    "precision": precision,
                    "config_id": session_config.config_id,
                    "case_id": case.case_id,
                    "error": error,
                }
            )
        return None, records

    for case in selected_cases:
        for repeat_index in range(args.repeats):
            wav_path = (
                args.output_dir
                / "wavs"
                / session_config.config_id
                / f"{precision}_{case.case_id}_r{repeat_index + 1:02d}.wav"
            )
            start = time.perf_counter()
            try:
                result = pipeline.synthesize_with_metadata(
                    case.text,
                    mode=case.mode,
                    voice_design=case.voice_design,
                    reference_wav_path=reference_wav if case.needs_reference else None,
                    max_steps=args.max_steps,
                    min_steps=args.min_steps,
                    cfg_value=args.cfg_value,
                    seed=args.seed + repeat_index,
                )
                synth_seconds = time.perf_counter() - start
                pipeline.write_wav(wav_path, result.waveform)
                records.append(
                    {
                        "ok": True,
                        "precision": precision,
                        "config_id": session_config.config_id,
                        "case_id": case.case_id,
                        "repeat_index": repeat_index,
                        "model_load_seconds": load_seconds,
                        "synth_seconds": round(synth_seconds, 6),
                        "decode_steps": result.metadata.decode_steps,
                        "stop_reason": result.metadata.stop_reason,
                        "audio": _audio_stats(result.waveform, pipeline.config.decode_sample_rate),
                        "output_wav": str(wav_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                records.append(
                    {
                        "ok": False,
                        "precision": precision,
                        "config_id": session_config.config_id,
                        "case_id": case.case_id,
                        "repeat_index": repeat_index,
                        "model_load_seconds": load_seconds,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return load_seconds, records


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("ok"):
            grouped.setdefault((record["config_id"], record["precision"]), []).append(record)

    aggregates: list[dict[str, Any]] = []
    for (config_id, precision), items in sorted(grouped.items()):
        aggregates.append(
            {
                "config_id": config_id,
                "precision": precision,
                "runs": len(items),
                "model_load_seconds": items[0]["model_load_seconds"],
                "synth_seconds": _stats([item["synth_seconds"] for item in items]),
                "decode_steps": _stats([float(item["decode_steps"]) for item in items]),
                "output_duration_seconds": _stats([item["audio"]["duration_seconds"] for item in items]),
            }
        )
    return aggregates


def _select_recommendation(
    configs: list[SessionConfig],
    aggregates: list[dict[str, Any]],
    selected_precisions: list[str],
    override_threshold: float,
) -> dict[str, Any]:
    by_config: dict[str, list[dict[str, Any]]] = {}
    for item in aggregates:
        by_config.setdefault(item["config_id"], []).append(item)

    eligible: list[dict[str, Any]] = []
    for config in configs:
        items = by_config.get(config.config_id, [])
        present_precisions = {item["precision"] for item in items}
        if not set(selected_precisions).issubset(present_precisions):
            continue
        mean_values = [item["synth_seconds"]["mean"] for item in items if item["synth_seconds"]["mean"] is not None]
        p50_values = [item["synth_seconds"]["p50"] for item in items if item["synth_seconds"]["p50"] is not None]
        if not mean_values:
            continue
        eligible.append(
            {
                "config": config.as_json(),
                "mean_synth_seconds": round(float(np.mean(mean_values)), 6),
                "p50_synth_seconds": round(float(np.mean(p50_values)), 6) if p50_values else None,
            }
        )
    eligible.sort(key=lambda item: (item["mean_synth_seconds"], item["config"]["config_id"]))
    common = eligible[0] if eligible else None

    precision_best: dict[str, dict[str, Any]] = {}
    optional_overrides: dict[str, dict[str, Any]] = {}
    for precision in selected_precisions:
        precision_items = [item for item in aggregates if item["precision"] == precision]
        precision_items = [item for item in precision_items if item["synth_seconds"]["mean"] is not None]
        precision_items.sort(key=lambda item: (item["synth_seconds"]["mean"], item["config_id"]))
        if not precision_items:
            continue
        best = precision_items[0]
        precision_best[precision] = {
            "config_id": best["config_id"],
            "mean_synth_seconds": best["synth_seconds"]["mean"],
        }
        if common is None:
            continue
        common_item = next(
            (
                item
                for item in precision_items
                if item["config_id"] == common["config"]["config_id"] and item["synth_seconds"]["mean"] is not None
            ),
            None,
        )
        if common_item is None:
            continue
        common_mean = common_item["synth_seconds"]["mean"]
        best_mean = best["synth_seconds"]["mean"]
        if common_mean and best_mean:
            improvement = (common_mean - best_mean) / common_mean
            if improvement >= override_threshold:
                optional_overrides[precision] = {
                    "config_id": best["config_id"],
                    "mean_synth_seconds": best_mean,
                    "improvement_vs_common": round(improvement, 6),
                }

    return {
        "common_recommended": common,
        "eligible_configs": eligible,
        "precision_best": precision_best,
        "optional_precision_overrides": optional_overrides,
        "override_threshold": override_threshold,
    }


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    recommendation = report["recommendation"]["common_recommended"]
    lines = [
        "# VoxCPM2 ORT Session Config Sweep",
        "",
        f"- JSON report: `{Path(report['json_report']).as_posix()}`",
        f"- preset: `{report['config']['config_preset']}`",
        f"- precisions: `{', '.join(report['config']['precisions'])}`",
        f"- cases: `{', '.join(report['config']['cases'])}`",
        f"- repeats: `{report['config']['repeats']}`",
        f"- max_steps: `{report['config']['max_steps']}`",
        "",
        "## Recommended Common Config",
        "",
    ]
    if recommendation is None:
        lines.append("No common successful config was found for the selected precision set.")
    else:
        config = recommendation["config"]
        memory = config["memory"]
        lines.extend(
            [
                f"- config_id: `{config['config_id']}`",
                f"- graph_optimization_level: `{config['graph_optimization_level']}`",
                f"- execution_mode: `{config['execution_mode']}`",
                f"- intra_op_num_threads: `{config['intra_op_num_threads']}`",
                f"- inter_op_num_threads: `{config['inter_op_num_threads']}`",
                f"- enable_mem_pattern: `{memory['enable_mem_pattern']}`",
                f"- enable_cpu_mem_arena: `{memory['enable_cpu_mem_arena']}`",
                f"- enable_mem_reuse: `{memory['enable_mem_reuse']}`",
                f"- aggregate mean synth: `{_fmt(recommendation['mean_synth_seconds'])}` s",
            ]
        )

    overrides = report["recommendation"]["optional_precision_overrides"]
    lines.extend(["", "## Precision Overrides", ""])
    if overrides:
        for precision, override in sorted(overrides.items()):
            lines.append(
                f"- `{precision}`: `{override['config_id']}` "
                f"({override['improvement_vs_common'] * 100:.1f}% faster than common)"
            )
    else:
        lines.append("No precision-specific override cleared the improvement threshold.")

    lines.extend(
        [
            "",
            "## Top Configs",
            "",
            "| rank | config | aggregate mean synth s | aggregate p50 synth s |",
            "|---:|---|---:|---:|",
        ]
    )
    for rank, item in enumerate(report["recommendation"]["eligible_configs"][:10], start=1):
        lines.append(
            f"| {rank} | `{item['config']['config_id']}` | "
            f"{_fmt(item['mean_synth_seconds'])} | {_fmt(item['p50_synth_seconds'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Precision Aggregates",
            "",
            "| config | precision | runs | load s | synth mean s | synth p50 s | decode steps p50 | duration p50 s |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["aggregates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['config_id']}`",
                    item["precision"],
                    _fmt(item["runs"]),
                    _fmt(item["model_load_seconds"]),
                    _fmt(item["synth_seconds"]["mean"]),
                    _fmt(item["synth_seconds"]["p50"]),
                    _fmt(item["decode_steps"]["p50"]),
                    _fmt(item["output_duration_seconds"]["p50"]),
                ]
            )
            + " |"
        )

    failures = [record for record in report["records"] if not record.get("ok")]
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.extend(["| config | precision | case | error |", "|---|---|---|---|"])
        for record in failures[:20]:
            error = str(record.get("error", "")).replace("\n", " ")
            if len(error) > 220:
                error = error[:217] + "..."
            lines.append(
                f"| `{record.get('config_id')}` | {record.get('precision')} | {record.get('case_id')} | `{error}` |"
            )
    else:
        lines.append("No failed runs.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The sweep uses one production runtime path and changes only ONNX Runtime `SessionOptions`.",
            "- FP32 and BF16 artifacts are benchmarked with the same case set and the same runtime shape policy.",
            "- Use `--max-steps 0` for a full until-stop confirmation run; the default bounded run keeps sweeps tractable.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_header(
    args: argparse.Namespace, configs: list[SessionConfig], json_report: Path, markdown_report: Path
) -> None:
    print("=" * 72, flush=True)
    print("VoxCPM2 ORT session sweep", flush=True)
    print("=" * 72, flush=True)
    print(f"precisions     : {', '.join(args.precisions)}", flush=True)
    print(f"cases          : {', '.join(args.cases)}", flush=True)
    print(f"preset         : {args.config_preset}", flush=True)
    print(f"configs        : {len(configs)}", flush=True)
    print(f"repeats        : {args.repeats}", flush=True)
    print(f"max/min steps  : {args.max_steps}/{args.min_steps}", flush=True)
    print(f"output_dir     : {args.output_dir}", flush=True)
    print(f"json_report    : {json_report}", flush=True)
    print(f"markdown_report: {markdown_report}", flush=True)
    print(flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_report = (args.json_report or (args.output_dir / "ort_session_sweep.json")).expanduser()
    markdown_report = (args.markdown_report or (args.output_dir / "ort_session_sweep.md")).expanduser()
    _, make_reference_wav = _bench_helpers()
    reference_wav = make_reference_wav(
        args.reference_wav.expanduser() if args.reference_wav else args.output_dir / "reference_16k.wav"
    )
    selected_cases = [_cases()[case_id] for case_id in args.cases]
    configs = _preset_configs(args)
    _print_header(args, configs, json_report, markdown_report)

    records: list[dict[str, Any]] = []
    for config_index, session_config in enumerate(configs, start=1):
        print(f"[{config_index}/{len(configs)}] {session_config.config_id}", flush=True)
        for precision in args.precisions:
            print(f"  running {precision}...", flush=True)
            load_seconds, precision_records = _run_precision_config(
                args,
                precision,
                session_config,
                selected_cases,
                reference_wav,
            )
            records.extend(precision_records)
            ok_records = [record for record in precision_records if record.get("ok")]
            synth_values = [record["synth_seconds"] for record in ok_records]
            status = "OK" if ok_records else "FAIL"
            print(
                f"    [{status}] load={_fmt(load_seconds)}s synth_mean={_fmt(_stats(synth_values)['mean'])}s",
                flush=True,
            )

    aggregates = _aggregate(records)
    recommendation = _select_recommendation(
        configs,
        aggregates,
        selected_precisions=args.precisions,
        override_threshold=args.override_threshold,
    )
    report = {
        "schema_version": 1,
        "created_unix_seconds": round(time.time(), 3),
        "repo_root": str(REPO_ROOT),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "config": {
            "config_preset": args.config_preset,
            "precisions": args.precisions,
            "cases": args.cases,
            "repeats": args.repeats,
            "seed": args.seed,
            "model_path": args.model_path,
            "max_steps": args.max_steps,
            "min_steps": args.min_steps,
            "cfg_value": args.cfg_value,
            "preload_sessions": args.preload_sessions,
        },
        "sweep_configs": [config.as_json() for config in configs],
        "records": records,
        "aggregates": aggregates,
        "recommendation": recommendation,
        "json_report": str(json_report),
        "markdown_report": str(markdown_report),
    }
    _write_json(json_report, report)
    _write_markdown(markdown_report, report)

    print(flush=True)
    common = recommendation["common_recommended"]
    if common is not None:
        print(f"recommended: {common['config']['config_id']}", flush=True)
    print(f"json saved    : {json_report}", flush=True)
    print(f"markdown saved: {markdown_report}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    _, VoxCPM2RuntimeConfig, _, graph_choices, execution_choices = _runtime_imports()
    parser = argparse.ArgumentParser(
        description="Sweep ONNX Runtime CPU SessionOptions for production FP32 and BF16 VoxCPM2 artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ort_session_sweep"),
        help="Directory for reports, generated reference WAV, and output WAVs.",
    )
    parser.add_argument(
        "--json-report", type=Path, help="JSON report path. Defaults to <output-dir>/ort_session_sweep.json."
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        help="Markdown report path. Defaults to <output-dir>/ort_session_sweep.md.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=["fp32", "bf16"],
        default=["fp32", "bf16"],
        help="ONNX artifact precision sets to benchmark.",
    )
    parser.add_argument("--cases", nargs="+", choices=CASE_IDS, default=["text_only_short"], help="Workloads to run.")
    parser.add_argument(
        "--config-preset",
        choices=["recommended", "focused", "full"],
        default="focused",
        help="Session config matrix. Use recommended for a quick smoke, focused for decision data, full for cartesian.",
    )
    parser.add_argument(
        "--graph-optimization-levels",
        nargs="+",
        choices=graph_choices,
        default=["disable", "basic", "extended", "all"],
        help="Graph optimization levels used by --config-preset full.",
    )
    parser.add_argument(
        "--execution-modes",
        nargs="+",
        choices=execution_choices,
        default=["sequential", "parallel"],
        help="Execution modes used by --config-preset full.",
    )
    parser.add_argument(
        "--intra-op-threads",
        nargs="+",
        default=["default", "4", "8"],
        help="Intra-op thread values used by --config-preset full. Use 'default' for ORT default scheduling.",
    )
    parser.add_argument(
        "--inter-op-threads",
        nargs="+",
        default=["1"],
        help="Inter-op thread values used by --config-preset full. Use 'default' for ORT default scheduling.",
    )
    parser.add_argument(
        "--memory-profiles",
        nargs="+",
        choices=MEMORY_PROFILES,
        default=["ort_default", "explicit_on", "arena_off"],
        help="Memory option profiles used by --config-preset full.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Repeat count per precision/config/case.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for host diffusion noise.")
    parser.add_argument("--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or HF id.")
    parser.add_argument("--reference-wav", type=Path, help="Optional reference WAV for clone workloads.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help=f"ONNX max decode steps. Use 0 for until-stop with safety cap {VoxCPM2RuntimeConfig().decode_safety_max_steps}.",
    )
    parser.add_argument("--min-steps", type=int, default=8, help="ONNX min decode steps before stop logits.")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Require local model files.")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only", help="Allow HF downloads.")
    parser.add_argument(
        "--preload-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create all ONNX sessions during load measurement.",
    )
    parser.add_argument(
        "--override-threshold",
        type=float,
        default=0.10,
        help="Minimum precision-specific mean speedup required before recommending an override.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.repeats < 1:
        raise SystemExit("error: --repeats must be >= 1")
    if args.max_steps < 0:
        raise SystemExit("error: --max-steps must be >= 0")
    if args.min_steps < 0:
        raise SystemExit("error: --min-steps must be >= 0")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
