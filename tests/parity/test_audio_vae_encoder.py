#!/usr/bin/env python3
"""Parity check for VoxCPM2 AudioVAE encoder: PyTorch vs ONNX Runtime."""

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


def _encoder_helpers():
    _ensure_repo_root_on_path()

    from src.export.export_audio_vae_decoder import _load_audio_vae, _resolve_model_path
    from src.export.export_audio_vae_encoder import AudioVAEEncoderWrapper

    return AudioVAEEncoderWrapper, _load_audio_vae, _resolve_model_path


def compare_audio_vae_encoder(args: argparse.Namespace) -> dict[str, float | list[int] | str]:
    AudioVAEEncoderWrapper, _load_audio_vae, _resolve_model_path = _encoder_helpers()
    model_dir = _resolve_model_path(args.model_path, args.local_files_only)
    audio_vae = _load_audio_vae(model_dir)
    if args.samples % audio_vae.chunk_size != 0:
        raise ValueError(f"--samples must be a multiple of audio_vae.chunk_size={audio_vae.chunk_size}")

    wrapper = AudioVAEEncoderWrapper(audio_vae).eval()

    rng = np.random.default_rng(args.seed)
    waveform_np = rng.standard_normal((args.batch_size, 1, args.samples), dtype=np.float32)
    waveform = torch.from_numpy(waveform_np)

    with torch.inference_mode():
        wrapper_output = wrapper(waveform).detach().cpu().numpy()
        reference_output = audio_vae.encode(waveform, audio_vae.sample_rate).detach().cpu().numpy()

    wrapper_abs_diff = np.abs(wrapper_output - reference_output)
    if wrapper_abs_diff.max() > args.atol:
        raise AssertionError(
            "encoder wrapper diverges from AudioVAE.encode on padded input: "
            + json.dumps(
                {
                    "max_abs_diff": float(wrapper_abs_diff.max()),
                    "mean_abs_diff": float(wrapper_abs_diff.mean()),
                },
                sort_keys=True,
            )
        )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(args.onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    ort_output = session.run(["latent"], {"waveform": waveform_np})[0]

    abs_diff = np.abs(wrapper_output - ort_output)
    result = {
        "torch_shape": list(wrapper_output.shape),
        "ort_shape": list(ort_output.shape),
        "dtype": str(ort_output.dtype),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "wrapper_vs_encode_max_abs_diff": float(wrapper_abs_diff.max()),
    }
    if result["max_abs_diff"] > args.atol:
        raise AssertionError(json.dumps(result, sort_keys=True))
    return result


def test_audio_vae_encoder_parity() -> None:
    import os
    import pytest

    onnx_path = os.environ.get("VOXCPM2_AUDIO_VAE_ENCODER_ONNX")
    if not onnx_path:
        pytest.skip("set VOXCPM2_AUDIO_VAE_ENCODER_ONNX to run encoder parity")
    args = _parser().parse_args(["--onnx-path", onnx_path])
    compare_audio_vae_encoder(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare AudioVAEEncoder PyTorch wrapper output against ONNX Runtime CPU output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx-path", type=Path, required=True, help="Path to audio_vae_encoder.onnx.")
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size for parity input.")
    parser.add_argument(
        "--samples", type=int, default=20480, help="Synthetic padded input samples; must match encoder boundary rules."
    )
    parser.add_argument("--seed", type=int, default=0, help="NumPy RNG seed for synthetic parity input.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Maximum allowed absolute difference.")
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
    result = compare_audio_vae_encoder(_parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
