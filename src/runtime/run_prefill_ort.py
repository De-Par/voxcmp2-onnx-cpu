#!/usr/bin/env python3
"""Check and run the exported VoxCPM2 prefill module with ONNX Runtime CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

INPUT_NAMES = ["text_tokens", "text_mask", "audio_features", "audio_mask"]
OUTPUT_NAMES = [
    "lm_hidden",
    "residual_hidden",
    "prefix_feat_cond",
    "base_k_cache",
    "base_v_cache",
    "base_cache_length",
    "residual_k_cache",
    "residual_v_cache",
    "residual_cache_length",
]


def _dim_value(dim: Any) -> str:
    if dim.dim_param:
        return f"dynamic:{dim.dim_param}"
    if dim.dim_value:
        return f"static:{dim.dim_value}"
    return "dynamic:unknown"


def _io_report(model_path: Path) -> dict[str, Any]:
    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph

    def describe(value_info: Any) -> dict[str, Any]:
        tensor_type = value_info.type.tensor_type
        elem_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        dims = [_dim_value(dim) for dim in tensor_type.shape.dim]
        return {"name": value_info.name, "dtype": elem_type, "dims": dims}

    return {
        "inputs": [describe(value_info) for value_info in graph.input],
        "outputs": [describe(value_info) for value_info in graph.output],
    }


def _check_model_path_based(model_path: Path) -> None:
    onnx.checker.check_model(str(model_path))


def _output_report(names: list[str], values: list[np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.min(value)) if value.size else None,
            "max": float(np.max(value)) if value.size else None,
        }
        for name, value in zip(names, values, strict=True)
    }


def _make_synthetic_inputs(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.seq_len < 1:
        raise ValueError("--seq-len must be >= 1")

    rng = np.random.default_rng(args.seed)
    text_tokens = rng.integers(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.seq_len),
        dtype=np.int64,
    )
    text_mask = np.ones((args.batch_size, args.seq_len), dtype=np.float32)
    audio_mask = np.zeros((args.batch_size, args.seq_len), dtype=np.float32)
    audio_features = np.zeros(
        (args.batch_size, args.seq_len, args.patch_size, args.feat_dim),
        dtype=np.float32,
    )

    def fill_audio_span(start: int, length: int) -> None:
        if length <= 0 or start >= args.seq_len:
            return
        end = min(args.seq_len, start + length)
        if end <= start:
            return
        text_tokens[:, start:end] = 0
        text_mask[:, start:end] = 0.0
        audio_mask[:, start:end] = 1.0
        audio_features[:, start:end, :, :] = rng.standard_normal(
            (args.batch_size, end - start, args.patch_size, args.feat_dim),
            dtype=np.float32,
        )

    if args.mode in ("plain_tts", "voice_design"):
        pass
    elif args.mode == "controllable_clone":
        ref_len = min(args.reference_steps, max(args.seq_len - 2, 0))
        fill_audio_span(1, ref_len)
    elif args.mode == "ultimate_clone":
        ref_len = min(args.reference_steps, max(args.seq_len - args.prompt_steps - 3, 0))
        fill_audio_span(1, ref_len)
        prompt_len = min(args.prompt_steps, max(args.seq_len - (ref_len + 3), 0))
        fill_audio_span(args.seq_len - prompt_len, prompt_len)
    else:
        raise ValueError(f"unsupported mode: {args.mode}")

    return {
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "audio_features": audio_features,
        "audio_mask": audio_mask,
    }


def run_prefill(args: argparse.Namespace) -> None:
    model_path = args.onnx_path.expanduser()
    _check_model_path_based(model_path)
    print("onnx_checker=ok")
    print("onnx_io=" + json.dumps(_io_report(model_path), sort_keys=True))

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    print(f"providers={session.get_providers()}")
    print(f"input_names={[item.name for item in session.get_inputs()]}")
    print(f"output_names={[item.name for item in session.get_outputs()]}")
    print(f"requested_mode={args.mode}")

    ort_inputs = _make_synthetic_inputs(args)
    outputs = session.run(OUTPUT_NAMES, ort_inputs)
    print("run_output=" + json.dumps(_output_report(OUTPUT_NAMES, outputs), sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Path-check and run the VoxCPM2Prefill ONNX graph with ONNX Runtime CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/runtime/run_prefill_ort.py "
            "--onnx-path artifacts/prefill/voxcpm2_prefill.onnx --mode plain_tts"
        ),
    )
    parser.add_argument("--onnx-path", type=Path, required=True, help="Path to voxcpm2_prefill.onnx.")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size for the ORT run.")
    parser.add_argument("--seq-len", type=int, default=16, help="Synthetic token/audio sequence length.")
    parser.add_argument("--patch-size", type=int, default=4, help="Audio feature patch size expected by the graph.")
    parser.add_argument(
        "--feat-dim", type=int, default=64, help="Audio feature channel dimension expected by the graph."
    )
    parser.add_argument("--vocab-size", type=int, default=73448, help="Synthetic tokenizer vocabulary upper bound.")
    parser.add_argument(
        "--mode",
        choices=["plain_tts", "voice_design", "controllable_clone", "ultimate_clone"],
        default="plain_tts",
        help="Synthetic pathway layout used to exercise text/reference/prompt masks.",
    )
    parser.add_argument(
        "--reference-steps", type=int, default=3, help="Synthetic reference-audio feature steps for clone modes."
    )
    parser.add_argument(
        "--prompt-steps", type=int, default=3, help="Synthetic prompt-audio feature steps for ultimate_clone."
    )
    parser.add_argument("--seed", type=int, default=0, help="NumPy RNG seed for synthetic input.")
    return parser


def main() -> int:
    run_prefill(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
