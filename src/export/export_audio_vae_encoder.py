#!/usr/bin/env python3
"""Export VoxCPM2 AudioVAE encoder to ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.export.export_audio_vae_decoder import _load_audio_vae, _resolve_model_path


INPUT_NAMES = ["waveform"]
OUTPUT_NAMES = ["latent"]


class AudioVAEEncoderWrapper(torch.nn.Module):
    """ONNX-facing wrapper for the neural encoder only.

    The official ``AudioVAE.encode`` method includes host-style preprocessing:
    ndim normalization and right-padding to hop length. For an ONNX boundary,
    host code performs those steps and passes padded ``[B, 1, samples]`` audio.
    """

    def __init__(self, audio_vae: torch.nn.Module) -> None:
        super().__init__()
        self.audio_vae = audio_vae

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.audio_vae.encoder(waveform)["mu"]


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
    model_dir = _resolve_model_path(args.model_path, args.local_files_only)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_vae = _load_audio_vae(model_dir)
    wrapper = AudioVAEEncoderWrapper(audio_vae).eval()

    samples = args.samples
    if samples % audio_vae.chunk_size != 0:
        raise ValueError(f"--samples must be a multiple of audio_vae.chunk_size={audio_vae.chunk_size}")
    waveform = torch.randn(args.batch_size, 1, samples, dtype=torch.float32)

    report = _shape_report(args.batch_size, samples)
    print("input_names=" + ",".join(INPUT_NAMES))
    print("output_names=" + ",".join(OUTPUT_NAMES))
    print("shape_report=" + json.dumps(report, sort_keys=True))
    print(f"output_path={output_path}")

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            args=(waveform,),
            f=str(output_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=args.opset,
            dynamo=True,
            external_data=True,
            dynamic_shapes=_dynamic_shapes(),
            optimize=False,
            do_constant_folding=False,
            verify=False,
        )

    print(f"exported={output_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export VoxCPM2 AudioVAE encoder to a standalone ONNX graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/export/export_audio_vae_encoder.py "
            "--output artifacts/audio_vae_encoder/audio_vae_encoder.onnx --samples 20480"
        ),
    )
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audio_vae_encoder/audio_vae_encoder.onnx"),
        help="ONNX output path.",
    )
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
