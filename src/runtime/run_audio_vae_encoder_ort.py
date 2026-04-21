#!/usr/bin/env python3
"""Check and run the exported VoxCPM2 AudioVAE encoder with ONNX Runtime CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


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


def run_encoder(args: argparse.Namespace) -> None:
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

    rng = np.random.default_rng(args.seed)
    waveform = rng.standard_normal((args.batch_size, 1, args.samples), dtype=np.float32)
    outputs = session.run(["latent"], {"waveform": waveform})
    latent = outputs[0]
    print(
        "run_output="
        + json.dumps(
            {
                "latent": {
                    "shape": list(latent.shape),
                    "dtype": str(latent.dtype),
                    "min": float(np.min(latent)),
                    "max": float(np.max(latent)),
                }
            },
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Path-check and run AudioVAE encoder ONNX on ORT CPU.")
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=20480)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    run_encoder(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
