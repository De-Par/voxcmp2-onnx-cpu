#!/usr/bin/env python3
"""Export VoxCPM2 AudioVAE encoder to ONNX"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from .common import (
        MODULE_EXPORT_CONTRACTS,
        PrecisionProfile,
        add_precision_argument,
        ensure_output_dir,
        export_onnx_graph,
        get_precision_profile,
        print_export_plan,
        resolve_output_path,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        MODULE_EXPORT_CONTRACTS,
        PrecisionProfile,
        add_precision_argument,
        ensure_output_dir,
        export_onnx_graph,
        get_precision_profile,
        print_export_plan,
        resolve_output_path,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _decoder_helpers():
    _ensure_repo_root_on_path()

    from src.export.export_audio_vae_decoder import _load_audio_vae, _resolve_model_path

    return _load_audio_vae, _resolve_model_path


MODULE_KEY = "audio_vae_encoder"
INPUT_NAMES = list(MODULE_EXPORT_CONTRACTS[MODULE_KEY].input_names)
OUTPUT_NAMES = list(MODULE_EXPORT_CONTRACTS[MODULE_KEY].output_names)


class AudioVAEEncoderWrapper(torch.nn.Module):
    """ONNX-facing wrapper for the neural encoder only.

    The official ``AudioVAE.encode`` method includes host-style preprocessing:
    ndim normalization and right-padding to hop length. For an ONNX boundary,
    host code performs those steps and passes padded ``[B, 1, samples]`` audio.
    """

    def __init__(self, audio_vae: torch.nn.Module, precision: PrecisionProfile | None = None) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.precision = precision or get_precision_profile("fp32")
        self.compute_dtype = self.precision.torch_compute_dtype()
        self.host_float_dtype = self.precision.torch_host_float_dtype()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(dtype=self.compute_dtype)
        latent = self.audio_vae.encoder(waveform)["mu"]
        return latent.to(dtype=self.host_float_dtype)


def _dynamic_shapes() -> dict[str, dict[int, Any]]:
    batch = torch.export.Dim("batch", min=1)
    samples = torch.export.Dim("samples", min=1)
    return {"waveform": {0: batch, 2: samples}}


def _shape_report(batch_size: int, samples: int) -> dict[str, Any]:
    return {
        "inputs": {
            "waveform": {
                "dtype": "float32",
                "dims": ["dynamic:batch", "static:1", "dynamic:samples"],
                "example_shape": [batch_size, 1, samples],
            }
        },
        "outputs": {
            "latent": {
                "dtype": "float32",
                "dims": ["dynamic:batch", "static:64", "dynamic:latent_steps"],
            }
        },
        "host_preconditions": {
            "waveform_rank": "3",
            "waveform_channels": "1",
            "samples_padded_to_multiple_of": "audio_vae.chunk_size",
        },
    }


def export_audio_vae_encoder(args: argparse.Namespace) -> None:
    _load_audio_vae, _resolve_model_path = _decoder_helpers()
    precision = get_precision_profile(args.precision)
    model_dir = _resolve_model_path(args.model_path, args.local_files_only)
    output_path = resolve_output_path(args.output, MODULE_KEY, precision)
    ensure_output_dir(output_path)

    audio_vae = _load_audio_vae(model_dir, precision)
    wrapper = AudioVAEEncoderWrapper(audio_vae, precision).eval()

    samples = args.samples
    if samples % audio_vae.chunk_size != 0:
        raise ValueError(f"--samples must be a multiple of audio_vae.chunk_size={audio_vae.chunk_size}")
    waveform = torch.randn(args.batch_size, 1, samples, dtype=precision.torch_host_float_dtype())

    print_export_plan(
        module_key=MODULE_KEY,
        precision=precision,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        shape_report=_shape_report(args.batch_size, samples),
        output_path=output_path,
    )
    export_onnx_graph(
        wrapper=wrapper,
        inputs=(waveform,),
        output_path=output_path,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset=args.opset,
        dynamic_shapes=_dynamic_shapes(),
    )

    print(f"exported={output_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export VoxCPM2 AudioVAE encoder to a standalone ONNX graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/export/export_audio_vae_encoder.py "
            "--precision fp32 --samples 20480"
        ),
    )
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ONNX output path. Defaults to models/onnx/<precision>/audio_vae_encoder/audio_vae_encoder.onnx.",
    )
    add_precision_argument(parser)
    parser.add_argument("--batch-size", type=int, default=1, help="Example batch dimension used during export.")
    parser.add_argument(
        "--samples",
        type=int,
        default=20480,
        help="Example padded input samples; must be a multiple of audio_vae.chunk_size.",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version for torch.onnx.export.")
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
    export_audio_vae_encoder(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
