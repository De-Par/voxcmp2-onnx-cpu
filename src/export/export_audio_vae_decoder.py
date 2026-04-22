#!/usr/bin/env python3
"""Export VoxCPM2 AudioVAE decoder to ONNX"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SRC = REPO_ROOT / "third_party" / "VoxCPM" / "src"

INPUT_NAMES = ["latent", "sr_cond"]
OUTPUT_NAMES = ["waveform"]


class AudioVAEDecoderWrapper(torch.nn.Module):
    """ONNX-facing wrapper with an explicit forward signature"""

    def __init__(self, audio_vae: torch.nn.Module) -> None:
        super().__init__()
        self.audio_vae = audio_vae

    def forward(self, latent: torch.Tensor, sr_cond: torch.Tensor) -> torch.Tensor:
        return self.audio_vae.decode(latent, sr_cond)


def _install_upstream_import_path() -> None:
    if not UPSTREAM_SRC.exists():
        raise FileNotFoundError(f"missing upstream VoxCPM source: {UPSTREAM_SRC}")
    sys.path.insert(0, str(UPSTREAM_SRC))


def _resolve_model_path(model_path: str, local_files_only: bool) -> Path:
    path = Path(model_path).expanduser()
    if path.is_dir():
        return path
    return Path(snapshot_download(model_path, local_files_only=local_files_only))


def _load_audio_vae(model_dir: Path) -> torch.nn.Module:
    _install_upstream_import_path()

    from voxcpm.model.voxcpm2 import SAFETENSORS_AVAILABLE, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2

    config = VoxCPMConfig.model_validate_json((model_dir / "config.json").read_text(encoding="utf-8"))
    audio_vae_config = getattr(config, "audio_vae_config", None)
    audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()

    safetensors_path = model_dir / "audiovae.safetensors"
    pth_path = model_dir / "audiovae.pth"
    if safetensors_path.exists() and SAFETENSORS_AVAILABLE:
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
        source = safetensors_path
    elif pth_path.exists():
        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        source = pth_path
    else:
        raise FileNotFoundError(f"AudioVAE checkpoint not found in {model_dir}")

    audio_vae.load_state_dict(state_dict, strict=True)
    audio_vae = audio_vae.to(device="cpu", dtype=torch.float32).eval()
    print(f"loaded_audio_vae={source}")
    print(f"sample_rate={audio_vae.sample_rate} out_sample_rate={audio_vae.out_sample_rate}")
    print(f"latent_dim={audio_vae.latent_dim} decode_chunk_size={audio_vae.decode_chunk_size}")
    return audio_vae


def _dynamic_shapes() -> dict[str, dict[int, Any]]:
    batch = torch.export.Dim("batch", min=1)
    latent_steps = torch.export.Dim("latent_steps", min=1)
    return {
        "latent": {0: batch, 2: latent_steps},
        "sr_cond": {0: batch},
    }


def _shape_report(batch_size: int, latent_steps: int) -> dict[str, Any]:
    return {
        "inputs": {
            "latent": {
                "dtype": "float32",
                "dims": ["dynamic:batch", "static:64", "dynamic:latent_steps"],
                "example_shape": [batch_size, 64, latent_steps],
            },
            "sr_cond": {
                "dtype": "int32",
                "dims": ["dynamic:batch"],
                "example_shape": [batch_size],
            },
        },
        "outputs": {
            "waveform": {
                "dtype": "float32",
                "dims": ["dynamic:batch", "static:1", "dynamic:samples"],
            }
        },
    }


def export_audio_vae_decoder(args: argparse.Namespace) -> None:
    model_dir = _resolve_model_path(args.model_path, args.local_files_only)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_vae = _load_audio_vae(model_dir)
    wrapper = AudioVAEDecoderWrapper(audio_vae).eval()

    latent = torch.randn(args.batch_size, audio_vae.latent_dim, args.latent_steps, dtype=torch.float32)
    sr_cond = torch.full((args.batch_size,), int(audio_vae.out_sample_rate), dtype=torch.int32)

    report = _shape_report(args.batch_size, args.latent_steps)
    print("input_names=" + ",".join(INPUT_NAMES))
    print("output_names=" + ",".join(OUTPUT_NAMES))
    print("shape_report=" + json.dumps(report, sort_keys=True))
    print(f"output_path={output_path}")

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            args=(latent, sr_cond),
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
        description="Export VoxCPM2 AudioVAE decoder to a standalone ONNX graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/export/export_audio_vae_decoder.py "
            "--output models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx"
        ),
    )
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx"),
        help="ONNX output path.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Example batch dimension used during export.")
    parser.add_argument("--latent-steps", type=int, default=4, help="Example latent time steps used during export.")
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
    export_audio_vae_decoder(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
