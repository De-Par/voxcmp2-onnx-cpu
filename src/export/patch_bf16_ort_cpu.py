#!/usr/bin/env python3
"""Patch existing BF16 ONNX artifacts for ONNX Runtime CPU kernel coverage"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import (
        BF16_ORT_CPU_FP32_ISLAND_OPS,
        MODULE_OUTPUT_LAYOUTS,
        ONNX_ROOT,
        apply_bf16_ort_cpu_compatibility_pass,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        BF16_ORT_CPU_FP32_ISLAND_OPS,
        MODULE_OUTPUT_LAYOUTS,
        ONNX_ROOT,
        apply_bf16_ort_cpu_compatibility_pass,
    )


DEFAULT_MODULES = ("audio_vae_encoder", "audio_vae_decoder", "prefill", "decode_chunk")


def _default_path(module_key: str, root: Path) -> Path:
    layout = MODULE_OUTPUT_LAYOUTS[module_key]
    return root.expanduser() / "bf16" / layout.directory / layout.filename


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Insert explicit FP32 islands around BF16 ops that ONNX Runtime CPU cannot load. "
            "This is a graph-edge compatibility pass for existing production BF16 artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/export/patch_bf16_ort_cpu.py "
            "--root models/onnx --modules audio_vae_encoder audio_vae_decoder prefill decode_chunk"
        ),
    )
    parser.add_argument("--root", type=Path, default=ONNX_ROOT, help="ONNX root containing bf16/<module> artifacts.")
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=DEFAULT_MODULES,
        default=list(DEFAULT_MODULES),
        help="BF16 modules to patch in place.",
    )
    parser.add_argument(
        "--op-types",
        nargs="+",
        default=list(BF16_ORT_CPU_FP32_ISLAND_OPS),
        help="Operator types that require FP32 islands for ORT CPU BF16 session creation.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip path-based onnx.checker after patching. Intended only for debugging corrupted artifacts.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    reports = []
    for module_key in args.modules:
        model_path = _default_path(module_key, args.root)
        if not model_path.is_file():
            raise FileNotFoundError(f"missing BF16 ONNX artifact for {module_key}: {model_path}")
        report = apply_bf16_ort_cpu_compatibility_pass(
            model_path,
            op_types=tuple(args.op_types),
            check_model=not args.no_check,
        )
        reports.append({"module": module_key, **report})
        print("patched=" + json.dumps(reports[-1], sort_keys=True), flush=True)
    print("bf16_ort_cpu_patch=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
