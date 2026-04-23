#!/usr/bin/env python3
"""Probe ORT CPU IO binding for the production decode-chunk graph.

This is a diagnostic tool, not a runtime implementation. It measures whether
CPU IO binding and preallocated output OrtValues reduce boundary overhead for
the hottest ONNX module before that complexity is moved into production code.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.runtime.run_decode_chunk_ort import OUTPUT_NAMES, _make_inputs
from src.runtime.session_factory import (
    EXECUTION_MODE_CHOICES,
    EXECUTION_MODES,
    GRAPH_OPTIMIZATION_CHOICES,
    GRAPH_OPTIMIZATION_LEVELS,
    LOG_SEVERITY_CHOICES,
    LOG_SEVERITY_LEVELS,
    ONNX_MODELS_ROOT,
)


FLOAT_OUTPUT_SHAPES = {
    "pred_audio_feature": ("batch_size", "chunk_size", "patch_size", "feat_dim"),
    "decoder_latent": ("batch_size", "feat_dim", "chunk_latent_steps"),
    "stop_logits": ("batch_size", "chunk_size", 2),
    "next_lm_hidden": ("batch_size", "hidden_size"),
    "next_residual_hidden": ("batch_size", "hidden_size"),
    "next_prefix_feat_cond": ("batch_size", "patch_size", "feat_dim"),
    "base_k_update": ("base_layers", "batch_size", "kv_heads", "chunk_size", "head_dim"),
    "base_v_update": ("base_layers", "batch_size", "kv_heads", "chunk_size", "head_dim"),
    "residual_k_update": ("residual_layers", "batch_size", "kv_heads", "chunk_size", "head_dim"),
    "residual_v_update": ("residual_layers", "batch_size", "kv_heads", "chunk_size", "head_dim"),
}
INT64_OUTPUT_SHAPES = {
    "next_base_current_length": (1,),
    "next_residual_current_length": (1,),
}


def _default_decode_chunk_path(precision: str) -> Path:
    return ONNX_MODELS_ROOT / precision / "decode_chunk" / "voxcpm2_decode_chunk.onnx"


def _session_options(args: argparse.Namespace) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = GRAPH_OPTIMIZATION_LEVELS[args.onnx_graph_optimization]
    options.execution_mode = EXECUTION_MODES[args.onnx_execution_mode]
    options.log_severity_level = LOG_SEVERITY_LEVELS[args.onnx_log_severity]
    if args.onnx_intra_op_threads is not None:
        options.intra_op_num_threads = args.onnx_intra_op_threads
    if args.onnx_inter_op_threads is not None:
        options.inter_op_num_threads = args.onnx_inter_op_threads
    options.enable_mem_pattern = bool(args.onnx_enable_mem_pattern)
    options.enable_cpu_mem_arena = bool(args.onnx_enable_cpu_mem_arena)
    options.enable_mem_reuse = bool(args.onnx_enable_mem_reuse)
    if args.enable_profiling:
        options.enable_profiling = True
        output_json = _output_json_path(args)
        prefix = args.profile_prefix.expanduser() if args.profile_prefix else output_json.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        options.profile_file_prefix = str(prefix)
    return options


def _shape_from_spec(spec: tuple[str | int, ...], args: argparse.Namespace) -> tuple[int, ...]:
    values = vars(args) | {"chunk_latent_steps": args.chunk_size * args.patch_size}
    return tuple(int(item if isinstance(item, int) else values[item]) for item in spec)


def _make_output_buffers(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, ort.OrtValue]]:
    buffers: dict[str, np.ndarray] = {}
    for name, spec in FLOAT_OUTPUT_SHAPES.items():
        buffers[name] = np.empty(_shape_from_spec(spec, args), dtype=np.float32)
    for name, spec in INT64_OUTPUT_SHAPES.items():
        buffers[name] = np.empty(_shape_from_spec(spec, args), dtype=np.int64)
    ortvalues = {name: ort.OrtValue.ortvalue_from_numpy(buffer) for name, buffer in buffers.items()}
    return buffers, ortvalues


def _bind_inputs(binding: ort.IOBinding, inputs: dict[str, np.ndarray]) -> None:
    for name, value in inputs.items():
        binding.bind_cpu_input(name, value)


def _run_iobinding_dynamic(session: ort.InferenceSession, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    binding = session.io_binding()
    _bind_inputs(binding, inputs)
    for name in OUTPUT_NAMES:
        binding.bind_output(name, "cpu")
    session.run_with_iobinding(binding)
    return binding.copy_outputs_to_cpu()


def _make_preallocated_binding(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    output_ortvalues: dict[str, ort.OrtValue],
) -> ort.IOBinding:
    binding = session.io_binding()
    _bind_inputs(binding, inputs)
    for name in OUTPUT_NAMES:
        binding.bind_ortvalue_output(name, output_ortvalues[name])
    return binding


def _run_iobinding_preallocated(session: ort.InferenceSession, binding: ort.IOBinding) -> None:
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()


def _time_call(fn, *, repeats: int) -> list[float]:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_seconds": 0.0, "median_seconds": 0.0, "min_seconds": 0.0, "max_seconds": 0.0}
    return {
        "mean_seconds": round(statistics.fmean(values), 6),
        "median_seconds": round(statistics.median(values), 6),
        "min_seconds": round(min(values), 6),
        "max_seconds": round(max(values), 6),
    }


def _compare_outputs(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict[str, Any]:
    max_abs = 0.0
    mismatches = []
    for name, expected, actual in zip(OUTPUT_NAMES, reference, candidate, strict=True):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            mismatches.append(
                {
                    "name": name,
                    "expected_shape": list(expected.shape),
                    "actual_shape": list(actual.shape),
                    "expected_dtype": str(expected.dtype),
                    "actual_dtype": str(actual.dtype),
                }
            )
            continue
        if np.issubdtype(expected.dtype, np.floating):
            max_abs = max(max_abs, float(np.max(np.abs(expected - actual))) if expected.size else 0.0)
        elif not np.array_equal(expected, actual):
            mismatches.append({"name": name, "reason": "integer output mismatch"})
    return {"max_abs_diff": round(max_abs, 8), "mismatches": mismatches}


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_run_id(args.run_id)
    output_json = _output_json_path(args)
    model_path = (args.onnx_path or _default_decode_chunk_path(args.precision)).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"decode_chunk ONNX not found: {model_path}")
    if args.max_cache_seq < args.cache_seq + args.chunk_size:
        raise ValueError("--max-cache-seq must be at least --cache-seq + --chunk-size")

    inputs = _make_inputs(args)
    load_start = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path), sess_options=_session_options(args), providers=["CPUExecutionProvider"]
    )
    load_seconds = time.perf_counter() - load_start
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"expected CPUExecutionProvider only, got {session.get_providers()}")

    for _ in range(args.warmup):
        session.run(OUTPUT_NAMES, inputs)

    reference = session.run(OUTPUT_NAMES, inputs)
    dynamic_outputs = _run_iobinding_dynamic(session, inputs)
    output_buffers, output_ortvalues = _make_output_buffers(args)
    preallocated_binding = _make_preallocated_binding(session, inputs, output_ortvalues)
    _run_iobinding_preallocated(session, preallocated_binding)
    preallocated_outputs = [output_buffers[name].copy() for name in OUTPUT_NAMES]

    timings = {
        "session_run": _stats(_time_call(lambda: session.run(OUTPUT_NAMES, inputs), repeats=args.repeats)),
        "iobinding_dynamic_outputs": _stats(
            _time_call(lambda: _run_iobinding_dynamic(session, inputs), repeats=args.repeats)
        ),
        "iobinding_preallocated_outputs": _stats(
            _time_call(lambda: _run_iobinding_preallocated(session, preallocated_binding), repeats=args.repeats)
        ),
    }
    profile_path = session.end_profiling() if args.enable_profiling else ""
    report = {
        "schema_version": 1,
        "precision": args.precision,
        "onnx_path": str(model_path),
        "providers": session.get_providers(),
        "load_seconds": round(load_seconds, 6),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timings": timings,
        "correctness": {
            "dynamic_outputs": _compare_outputs(reference, dynamic_outputs),
            "preallocated_outputs": _compare_outputs(reference, preallocated_outputs),
        },
        "profile_path": profile_path,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("io_binding_probe=ok")
    print(f"json saved: {output_json}")
    for name, item in timings.items():
        print(f"{name:28} mean={item['mean_seconds']:.6f}s median={item['median_seconds']:.6f}s")
    return report


def _output_json_path(args: argparse.Namespace) -> Path:
    if args.output_json:
        return args.output_json.expanduser()
    name = f"io_binding_probe_{args.run_id}.json" if args.run_id else "io_binding_probe.json"
    return Path("artifacts/profile") / name


def _validate_run_id(run_id: str | None) -> None:
    if run_id is None:
        return
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("--run-id may contain only letters, digits, dot, underscore, and dash")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare decode_chunk session.run with ORT CPU IO binding and preallocated output binding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32", help="Default artifact family.")
    parser.add_argument("--run-id", help="Optional file-name suffix for concurrent IO-binding probes.")
    parser.add_argument("--onnx-path", type=Path, help="Override decode_chunk ONNX path.")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="JSON report output path. Defaults to artifacts/profile/io_binding_probe[_<run-id>].json.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup session.run calls before timing.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed calls for each method.")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size.")
    parser.add_argument("--chunk-size", type=int, default=4, help="Decode steps per chunk.")
    parser.add_argument("--cache-seq", type=int, default=16, help="Valid synthetic cache length.")
    parser.add_argument("--max-cache-seq", type=int, default=64, help="Synthetic fixed cache capacity.")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden size.")
    parser.add_argument("--patch-size", type=int, default=4, help="Audio patch size.")
    parser.add_argument("--feat-dim", type=int, default=64, help="Audio feature dimension.")
    parser.add_argument("--base-layers", type=int, default=28, help="Base LM cache layer count.")
    parser.add_argument("--residual-layers", type=int, default=8, help="Residual LM cache layer count.")
    parser.add_argument("--kv-heads", type=int, default=2, help="K/V heads.")
    parser.add_argument("--head-dim", type=int, default=128, help="K/V head dimension.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic input seed.")
    parser.add_argument("--onnx-graph-optimization", choices=GRAPH_OPTIMIZATION_CHOICES, default="all", help="ORT opt.")
    parser.add_argument("--onnx-execution-mode", choices=EXECUTION_MODE_CHOICES, default="sequential", help="ORT mode.")
    parser.add_argument("--onnx-log-severity", choices=LOG_SEVERITY_CHOICES, default="error", help="ORT log level.")
    parser.add_argument("--onnx-intra-op-threads", type=int, default=8, help="ORT intra-op threads.")
    parser.add_argument("--onnx-inter-op-threads", type=int, default=1, help="ORT inter-op threads.")
    parser.add_argument(
        "--onnx-enable-mem-pattern", action=argparse.BooleanOptionalAction, default=True, help="ORT mem pattern."
    )
    parser.add_argument(
        "--onnx-enable-cpu-mem-arena", action=argparse.BooleanOptionalAction, default=True, help="ORT CPU arena."
    )
    parser.add_argument(
        "--onnx-enable-mem-reuse", action=argparse.BooleanOptionalAction, default=True, help="ORT memory reuse."
    )
    parser.add_argument("--enable-profiling", action="store_true", help="Write an ORT profile for this probe.")
    parser.add_argument("--profile-prefix", type=Path, help="ORT profile prefix when --enable-profiling is set.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.repeats < 1:
        raise SystemExit("error: --repeats must be >= 1")
    if args.warmup < 0:
        raise SystemExit("error: --warmup must be >= 0")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
