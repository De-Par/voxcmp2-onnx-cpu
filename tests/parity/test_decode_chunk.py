#!/usr/bin/env python3
"""Parity check for production VoxCPM2 decode chunk: PyTorch vs ONNX Runtime"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _decode_chunk_helpers():
    _ensure_repo_root_on_path()

    from src.export.common import get_precision_profile
    from src.export.export_audio_vae_decoder import _resolve_model_path
    from src.export.export_decode_chunk import (
        INPUT_NAMES,
        OUTPUT_NAMES,
        VoxCPM2DecodeChunkWrapper,
        _model_dims,
        make_synthetic_decode_chunk_inputs,
    )
    from src.export.export_prefill import load_voxcpm2_prefill_model

    return (
        INPUT_NAMES,
        OUTPUT_NAMES,
        VoxCPM2DecodeChunkWrapper,
        _model_dims,
        make_synthetic_decode_chunk_inputs,
        load_voxcpm2_prefill_model,
        _resolve_model_path,
        get_precision_profile,
    )


def _compare_output(name: str, torch_value: np.ndarray, ort_value: np.ndarray) -> dict[str, float | list[int] | str]:
    if torch_value.dtype.kind in {"i", "u"}:
        equal = np.array_equal(torch_value, ort_value)
        max_abs_diff = (
            0.0 if equal else float(np.max(np.abs(torch_value.astype(np.int64) - ort_value.astype(np.int64))))
        )
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


def compare_decode_chunk(args: argparse.Namespace) -> dict[str, object]:
    (
        INPUT_NAMES,
        OUTPUT_NAMES,
        VoxCPM2DecodeChunkWrapper,
        _model_dims,
        make_synthetic_decode_chunk_inputs,
        load_voxcpm2_prefill_model,
        _resolve_model_path,
        get_precision_profile,
    ) = _decode_chunk_helpers()
    precision = get_precision_profile(args.precision)
    model_dir = _resolve_model_path(args.model_path, args.local_files_only)
    model = load_voxcpm2_prefill_model(model_dir, precision)
    wrapper = VoxCPM2DecodeChunkWrapper(
        model,
        chunk_size=args.chunk_size,
        inference_timesteps=args.inference_timesteps,
        precision=precision,
    ).eval()
    inputs = make_synthetic_decode_chunk_inputs(
        batch_size=args.batch_size,
        current_length=args.cache_seq,
        max_cache_seq=args.max_cache_seq,
        chunk_size=args.chunk_size,
        cfg_value=args.cfg_value,
        seed=args.seed,
        **_model_dims(model),
    )

    with torch.inference_mode():
        torch_outputs = wrapper(*(inputs[name] for name in INPUT_NAMES))
    torch_outputs_np = [value.detach().cpu().numpy() for value in torch_outputs]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(args.onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {name: inputs[name].detach().cpu().numpy() for name in INPUT_NAMES}
    ort_outputs = session.run(OUTPUT_NAMES, ort_inputs)

    per_output = [
        _compare_output(name, torch_value, ort_value)
        for name, torch_value, ort_value in zip(OUTPUT_NAMES, torch_outputs_np, ort_outputs, strict=True)
    ]
    failures = [item for item in per_output if item["max_abs_diff"] > args.atol]
    result = {
        "atol": args.atol,
        "precision": precision.name,
        "chunk_size": args.chunk_size,
        "cache_seq": args.cache_seq,
        "max_cache_seq": args.max_cache_seq,
        "inference_timesteps": args.inference_timesteps,
        "max_abs_diff": max(item["max_abs_diff"] for item in per_output),
        "outputs": per_output,
    }
    if failures:
        raise AssertionError(json.dumps({"failures": failures, "result": result}, sort_keys=True))
    return result


def test_decode_chunk_parity() -> None:
    import os

    import pytest

    onnx_path = os.environ.get("VOXCPM2_DECODE_CHUNK_ONNX")
    if not onnx_path:
        pytest.skip("set VOXCPM2_DECODE_CHUNK_ONNX to run decode-chunk parity")
    args = _parser().parse_args(["--onnx-path", onnx_path])
    compare_decode_chunk(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare VoxCPM2DecodeChunk PyTorch wrapper outputs against ONNX Runtime CPU outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx-path", type=Path, required=True, help="Path to voxcpm2_decode_chunk.onnx.")
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Precision profile used to load the PyTorch wrapper for parity.",
    )
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size for parity input.")
    parser.add_argument("--chunk-size", type=int, default=4, help="Decode steps executed by one ONNX session.run.")
    parser.add_argument("--cache-seq", type=int, default=16, help="Synthetic valid KV-cache length entering the chunk.")
    parser.add_argument(
        "--max-cache-seq",
        type=int,
        default=64,
        help="Synthetic fixed KV-cache capacity entering the chunk; must be at least --cache-seq + --chunk-size.",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="CFM/LocDiT solver steps embedded in each internal decode step.",
    )
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="PyTorch RNG seed for synthetic parity input.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Maximum allowed absolute difference per tensor.")
    parser.add_argument(
        "--local-files-only", action="store_true", default=True, help="Require local Hugging Face cache/model files."
    )
    parser.add_argument(
        "--allow-download",
        action="store_false",
        dest="local_files_only",
        help="Allow snapshot_download to fetch missing files.",
    )
    return parser


def main() -> int:
    result = compare_decode_chunk(_parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
