#!/usr/bin/env python3
"""Build preferred CPU runtime artifacts for exported VoxCPM2 ONNX modules.

This post-export step tries to create `.ort` artifacts first, but large models
can exceed what the current ORT serializer handles in one file. For those
graphs, the builder falls back to `*.optimized.onnx` plus external initializers,
which still removes a large part of startup optimization cost.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import onnxruntime as ort

try:
    from .common import MODULE_OUTPUT_LAYOUTS, ONNX_ROOT, PRECISION_CHOICES
except ImportError:
    from common import MODULE_OUTPUT_LAYOUTS, ONNX_ROOT, PRECISION_CHOICES  # type: ignore[no-redef]


DEFAULT_MODULES = ("prefill", "decode_chunk")
MODULE_CHOICES = ("audio_vae_encoder", "audio_vae_decoder", "prefill", "decode_chunk")
ORT_SINGLE_FILE_SIZE_LIMIT = 2_147_483_648

GRAPH_OPTIMIZATION_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
EXECUTION_MODES = {
    "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
    "parallel": ort.ExecutionMode.ORT_PARALLEL,
}
LOG_SEVERITY_LEVELS = {
    "verbose": 0,
    "info": 1,
    "warning": 2,
    "error": 3,
    "fatal": 4,
}


def _module_path(root: Path, precision: str, module_key: str) -> Path:
    layout = MODULE_OUTPUT_LAYOUTS[module_key]
    return root.expanduser() / precision / layout.directory / layout.filename


def _external_data_path(path: Path) -> Path | None:
    if path.suffix != ".onnx":
        return None
    return path.with_suffix(path.suffix + ".data")


def _optimized_onnx_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.optimized{path.suffix}")


def _remove_if_exists(path: Path | None) -> None:
    if path is not None and path.exists():
        path.unlink()


def _session_options(
    *,
    output_path: Path,
    graph_optimization_level: str,
    execution_mode: str,
    log_severity_level: str,
    save_model_format: str | None,
    external_initializer_filename: str | None,
) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = GRAPH_OPTIMIZATION_LEVELS[graph_optimization_level]
    options.execution_mode = EXECUTION_MODES[execution_mode]
    options.log_severity_level = LOG_SEVERITY_LEVELS[log_severity_level]
    options.optimized_model_filepath = str(output_path)
    if save_model_format is not None:
        options.add_session_config_entry("session.save_model_format", save_model_format)
    if external_initializer_filename is not None:
        options.add_session_config_entry(
            "session.optimized_model_external_initializers_file_name",
            external_initializer_filename,
        )
        options.add_session_config_entry(
            "session.optimized_model_external_initializers_min_size_in_bytes",
            "1024",
        )
    return options


def _resolve_target_platform(target_platform: str) -> str:
    if target_platform != "auto":
        return target_platform
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "amd64"
    if machine in {"arm64", "aarch64"}:
        return "arm"
    raise ValueError(
        f"Unsupported host architecture {platform.machine()!r}; pass --target-platform arm or amd64 explicitly."
    )


def _disabled_optimizers(graph_optimization_level: str, target_platform: str) -> list[str] | None:
    if graph_optimization_level == "all" and target_platform != "amd64":
        # Match ONNX Runtime's own convert_onnx_models_to_ort guidance: NCHWc is
        # not applicable on ARM and can bake in device-specific rewrites that do
        # not help the CPU-only target platforms we support.
        return ["NchwcTransformer"]
    return None


def _validate_artifact(path: Path, *, graph_optimization_level: str = "disable") -> dict[str, Any]:
    options = ort.SessionOptions()
    options.graph_optimization_level = GRAPH_OPTIMIZATION_LEVELS[graph_optimization_level]
    start = time.perf_counter()
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    return {
        "validate_seconds": round(time.perf_counter() - start, 6),
        "providers": session.get_providers(),
    }


def _build_optimized_onnx(
    model_path: Path,
    *,
    graph_optimization_level: str,
    execution_mode: str,
    log_severity_level: str,
    target_platform: str,
) -> dict[str, Any]:
    output_path = _optimized_onnx_path(model_path)
    external_path = _external_data_path(output_path)
    _remove_if_exists(output_path)
    _remove_if_exists(external_path)
    options = _session_options(
        output_path=output_path,
        graph_optimization_level=graph_optimization_level,
        execution_mode=execution_mode,
        log_severity_level=log_severity_level,
        save_model_format=None,
        external_initializer_filename=external_path.name if external_path is not None else None,
    )
    disabled_optimizers = _disabled_optimizers(graph_optimization_level, target_platform)
    start = time.perf_counter()
    _ = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
        disabled_optimizers=disabled_optimizers,
    )
    build_seconds = time.perf_counter() - start
    validation = _validate_artifact(output_path)
    return {
        "artifact_kind": "optimized_onnx",
        "path": str(output_path),
        "external_data_path": str(external_path) if external_path is not None else None,
        "build_seconds": round(build_seconds, 6),
        "target_platform": target_platform,
        "disabled_optimizers": disabled_optimizers or [],
        **validation,
    }


def _build_ort(
    model_path: Path,
    *,
    graph_optimization_level: str,
    execution_mode: str,
    log_severity_level: str,
    target_platform: str,
) -> dict[str, Any]:
    output_path = model_path.with_suffix(".ort")
    _remove_if_exists(output_path)
    options = _session_options(
        output_path=output_path,
        graph_optimization_level=graph_optimization_level,
        execution_mode=execution_mode,
        log_severity_level=log_severity_level,
        save_model_format="ORT",
        external_initializer_filename=None,
    )
    disabled_optimizers = _disabled_optimizers(graph_optimization_level, target_platform)
    start = time.perf_counter()
    _ = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
        disabled_optimizers=disabled_optimizers,
    )
    build_seconds = time.perf_counter() - start
    validation = _validate_artifact(output_path)
    return {
        "artifact_kind": "ort",
        "path": str(output_path),
        "external_data_path": None,
        "build_seconds": round(build_seconds, 6),
        "target_platform": target_platform,
        "disabled_optimizers": disabled_optimizers or [],
        **validation,
    }


def _size_blocker(model_path: Path) -> str | None:
    external_path = _external_data_path(model_path)
    if external_path is None or not external_path.exists():
        return None
    if external_path.stat().st_size < ORT_SINGLE_FILE_SIZE_LIMIT:
        return None
    gib = external_path.stat().st_size / (1024**3)
    return (
        f"external data is {gib:.2f} GiB; local ORT 1.24.4 failed to save `.ort` for heavy models above "
        "the 2 GiB protobuf-scale threshold, so the builder will use optimized ONNX instead"
    )


def build_runtime_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser()
    target_platform = _resolve_target_platform(args.target_platform)
    reports: list[dict[str, Any]] = []

    for precision in args.precisions:
        for module_key in args.modules:
            model_path = _module_path(root, precision, module_key)
            if not model_path.is_file():
                raise FileNotFoundError(f"missing ONNX artifact for {precision}/{module_key}: {model_path}")

            model_report: dict[str, Any] = {
                "precision": precision,
                "module": module_key,
                "model_path": str(model_path),
                "artifacts": [],
                "blockers": [],
            }
            size_blocker = _size_blocker(model_path)
            should_try_ort = args.target in {"ort", "both"} or (
                args.target == "auto" and (args.force_ort_attempt or size_blocker is None)
            )
            if size_blocker is not None and not args.force_ort_attempt:
                model_report["blockers"].append(size_blocker)

            if should_try_ort:
                try:
                    ort_report = _build_ort(
                        model_path,
                        graph_optimization_level=args.graph_optimization_level,
                        execution_mode=args.execution_mode,
                        log_severity_level=args.log_severity_level,
                        target_platform=target_platform,
                    )
                    model_report["artifacts"].append(ort_report)
                except Exception as exc:  # noqa: BLE001 - keep fallback working for large graphs.
                    model_report["blockers"].append(f"ort_build_failed: {type(exc).__name__}: {exc}")

            should_build_optimized = args.target in {"optimized_onnx", "both"} or (
                args.target == "auto" and not any(item["artifact_kind"] == "ort" for item in model_report["artifacts"])
            )
            if should_build_optimized:
                optimized_report = _build_optimized_onnx(
                    model_path,
                    graph_optimization_level=args.graph_optimization_level,
                    execution_mode=args.execution_mode,
                    log_severity_level=args.log_severity_level,
                    target_platform=target_platform,
                )
                model_report["artifacts"].append(optimized_report)

            reports.append(model_report)
            print("runtime_artifact=" + json.dumps(model_report, sort_keys=True), flush=True)

    result = {
        "schema_version": 1,
        "root": str(root),
        "graph_optimization_level": args.graph_optimization_level,
        "execution_mode": args.execution_mode,
        "log_severity_level": args.log_severity_level,
        "target_platform": target_platform,
        "reports": reports,
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"json saved: {args.report_json}", flush=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build preferred ONNX Runtime CPU artifacts for exported VoxCPM2 modules. "
            "Large graphs fall back from `.ort` to `*.optimized.onnx` when `.ort` serialization is blocked."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/export/build_runtime_artifacts.py "
            "--precisions fp32 bf16 --modules prefill decode_chunk"
        ),
    )
    parser.add_argument(
        "--root", type=Path, default=ONNX_ROOT, help="Root directory containing <precision>/<module> ONNX exports."
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=PRECISION_CHOICES,
        default=list(PRECISION_CHOICES),
        help="Precision families to process.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=MODULE_CHOICES,
        default=list(DEFAULT_MODULES),
        help="Modules to postprocess for runtime startup.",
    )
    parser.add_argument(
        "--target",
        choices=["auto", "ort", "optimized_onnx", "both"],
        default="auto",
        help=("Artifact kind to build. `auto` prefers `.ort` when practical and otherwise creates optimized ONNX."),
    )
    parser.add_argument(
        "--force-ort-attempt",
        action="store_true",
        help="Attempt `.ort` save even when the model external data exceeds the observed 2 GiB blocker threshold.",
    )
    parser.add_argument(
        "--graph-optimization-level",
        choices=tuple(GRAPH_OPTIMIZATION_LEVELS),
        default="all",
        help="Offline ORT graph optimization level used while building runtime artifacts.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=tuple(EXECUTION_MODES),
        default="sequential",
        help="Execution mode used during offline ORT optimization.",
    )
    parser.add_argument(
        "--log-severity-level",
        choices=tuple(LOG_SEVERITY_LEVELS),
        default="error",
        help="ORT log severity while building artifacts.",
    )
    parser.add_argument(
        "--target-platform",
        choices=["auto", "arm", "amd64"],
        default="auto",
        help="Target CPU platform used when selecting ORT offline optimizers.",
    )
    parser.add_argument("--report-json", type=Path, help="Optional JSON report path.")
    return parser


def main() -> int:
    build_runtime_artifacts(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
