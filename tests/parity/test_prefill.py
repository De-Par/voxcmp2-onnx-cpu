#!/usr/bin/env python3
"""Parity check for VoxCPM2 prefill: PyTorch vs ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.export.export_prefill import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    VoxCPM2PrefillWrapper,
    load_voxcpm2_prefill_model,
    make_synthetic_prefill_inputs,
)
from src.export.export_audio_vae_decoder import _resolve_model_path


def _compare_output(name: str, torch_value: np.ndarray, ort_value: np.ndarray) -> dict[str, float | list[int] | str]:
    if torch_value.dtype.kind in {"i", "u"}:
        equal = np.array_equal(torch_value, ort_value)
        max_abs_diff = 0.0 if equal else float(np.max(np.abs(torch_value.astype(np.int64) - ort_value.astype(np.int64))))
        mean_abs_diff = max_abs_diff
    else:
        abs_diff = np.abs(torch_value - ort_value)
        max_abs_diff = float(abs_diff.max()) if abs_diff.size else 0.0
        mean_abs_diff = float(abs_diff.mean()) if abs_diff.size else 0.0
    return {
        "name": name,
        "torch_shape": list(torch_value.shape),
        "ort_shape": list(ort_value.shape),
        "torch_dtype": str(torch_value.dtype),
        "ort_dtype": str(ort_value.dtype),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def compare_prefill(args: argparse.Namespace) -> dict[str, object]:
    model_dir = _resolve_model_path(args.model_path, args.local_files_only)
    model = load_voxcpm2_prefill_model(model_dir)
    wrapper = VoxCPM2PrefillWrapper(model).eval()

    torch_inputs = make_synthetic_prefill_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        patch_size=model.patch_size,
        feat_dim=model.feat_dim,
        vocab_size=model.config.lm_config.vocab_size,
        mode=args.mode,
        seed=args.seed,
        reference_steps=args.reference_steps,
        prompt_steps=args.prompt_steps,
    )
    with torch.inference_mode():
        torch_outputs = wrapper(*(torch_inputs[name] for name in INPUT_NAMES))
    torch_outputs_np = [value.detach().cpu().numpy() for value in torch_outputs]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(args.onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {name: torch_inputs[name].detach().cpu().numpy() for name in INPUT_NAMES}
    ort_outputs = session.run(OUTPUT_NAMES, ort_inputs)

    per_output = [
        _compare_output(name, torch_value, ort_value)
        for name, torch_value, ort_value in zip(OUTPUT_NAMES, torch_outputs_np, ort_outputs, strict=True)
    ]
    failures = [item for item in per_output if item["max_abs_diff"] > args.atol]
    result = {
        "mode": args.mode,
        "atol": args.atol,
        "max_abs_diff": max(item["max_abs_diff"] for item in per_output),
        "outputs": per_output,
    }
    if failures:
        raise AssertionError(json.dumps({"failures": failures, "result": result}, sort_keys=True))
    return result


def test_prefill_parity() -> None:
    import os
    import pytest

    onnx_path = os.environ.get("VOXCPM2_PREFILL_ONNX")
    if not onnx_path:
        pytest.skip("set VOXCPM2_PREFILL_ONNX to run prefill parity")
    args = _parser().parse_args(["--onnx-path", onnx_path])
    compare_prefill(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare VoxCPM2 prefill PyTorch and ONNX Runtime outputs.")
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--model-path", default="openbmb/VoxCPM2")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--mode", choices=["plain_tts", "voice_design", "controllable_clone", "ultimate_clone"], default="plain_tts")
    parser.add_argument("--reference-steps", type=int, default=3)
    parser.add_argument("--prompt-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    return parser


def main() -> int:
    result = compare_prefill(_parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
