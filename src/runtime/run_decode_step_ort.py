#!/usr/bin/env python3
"""Check and run the exported VoxCPM2 decode-step module with ONNX Runtime CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

INPUT_NAMES = [
    "lm_hidden",
    "residual_hidden",
    "prefix_feat_cond",
    "base_k_cache",
    "base_v_cache",
    "base_cache_length",
    "residual_k_cache",
    "residual_v_cache",
    "residual_cache_length",
    "diffusion_noise",
    "cfg_value",
]
OUTPUT_NAMES = [
    "pred_audio_feature",
    "decoder_latent",
    "stop_logits",
    "next_lm_hidden",
    "next_residual_hidden",
    "next_prefix_feat_cond",
    "next_base_k_cache",
    "next_base_v_cache",
    "next_base_cache_length",
    "next_residual_k_cache",
    "next_residual_v_cache",
    "next_residual_cache_length",
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


def _make_inputs(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.cache_seq < 1:
        raise ValueError("--cache-seq must be >= 1")

    rng = np.random.default_rng(args.seed)
    return {
        "lm_hidden": rng.standard_normal((args.batch_size, args.hidden_size), dtype=np.float32),
        "residual_hidden": rng.standard_normal((args.batch_size, args.hidden_size), dtype=np.float32),
        "prefix_feat_cond": rng.standard_normal((args.batch_size, args.patch_size, args.feat_dim), dtype=np.float32),
        "base_k_cache": rng.standard_normal(
            (args.base_layers, args.batch_size, args.kv_heads, args.cache_seq, args.head_dim),
            dtype=np.float32,
        ),
        "base_v_cache": rng.standard_normal(
            (args.base_layers, args.batch_size, args.kv_heads, args.cache_seq, args.head_dim),
            dtype=np.float32,
        ),
        "base_cache_length": np.array([args.cache_seq], dtype=np.int64),
        "residual_k_cache": rng.standard_normal(
            (args.residual_layers, args.batch_size, args.kv_heads, args.cache_seq, args.head_dim),
            dtype=np.float32,
        ),
        "residual_v_cache": rng.standard_normal(
            (args.residual_layers, args.batch_size, args.kv_heads, args.cache_seq, args.head_dim),
            dtype=np.float32,
        ),
        "residual_cache_length": np.array([args.cache_seq], dtype=np.int64),
        "diffusion_noise": rng.standard_normal((args.batch_size, args.feat_dim, args.patch_size), dtype=np.float32),
        "cfg_value": np.array([args.cfg_value], dtype=np.float32),
    }


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


def run_decode_step(args: argparse.Namespace) -> None:
    model_path = args.onnx_path.expanduser()
    onnx.checker.check_model(str(model_path))
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

    outputs = session.run(OUTPUT_NAMES, _make_inputs(args))
    print("run_output=" + json.dumps(_output_report(OUTPUT_NAMES, outputs), sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Path-check and run the VoxCPM2DecodeStep ONNX graph with ONNX Runtime CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/runtime/run_decode_step_ort.py "
            "--onnx-path artifacts/decode_step/voxcpm2_decode_step.onnx --cache-seq 16"
        ),
    )
    parser.add_argument("--onnx-path", type=Path, required=True, help="Path to voxcpm2_decode_step.onnx.")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size for the ORT run.")
    parser.add_argument("--cache-seq", type=int, default=16, help="Synthetic valid KV-cache length entering the step.")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Transformer hidden size expected by the graph.")
    parser.add_argument("--patch-size", type=int, default=4, help="Audio feature patch size expected by the graph.")
    parser.add_argument(
        "--feat-dim", type=int, default=64, help="Audio feature channel dimension expected by the graph."
    )
    parser.add_argument("--base-layers", type=int, default=28, help="Base LM layer count represented in cache tensors.")
    parser.add_argument(
        "--residual-layers", type=int, default=8, help="Residual LM layer count represented in cache tensors."
    )
    parser.add_argument(
        "--kv-heads", type=int, default=2, help="Number of key/value heads represented in cache tensors."
    )
    parser.add_argument(
        "--head-dim", type=int, default=128, help="Per-head key/value dimension represented in cache tensors."
    )
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="NumPy RNG seed for synthetic input.")
    return parser


def main() -> int:
    run_decode_step(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
