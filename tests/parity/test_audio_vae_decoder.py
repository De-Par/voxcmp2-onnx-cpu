#!/usr/bin/env python3
"""Parity check for VoxCPM2 AudioVAE decoder: PyTorch vs ONNX Runtime."""

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

from src.export.export_audio_vae_decoder import AudioVAEDecoderWrapper, _load_audio_vae, _resolve_model_path


def compare_audio_vae_decoder(args: argparse.Namespace) -> dict[str, float | list[int] | str]:
    model_dir = _resolve_model_path(args.model_path, args.local_files_only)
    audio_vae = _load_audio_vae(model_dir)
    wrapper = AudioVAEDecoderWrapper(audio_vae).eval()

    rng = np.random.default_rng(args.seed)
    latent_np = rng.standard_normal((args.batch_size, audio_vae.latent_dim, args.latent_steps), dtype=np.float32)
    sr_cond_np = np.full((args.batch_size,), int(audio_vae.out_sample_rate), dtype=np.int32)

    latent = torch.from_numpy(latent_np)
    sr_cond = torch.from_numpy(sr_cond_np)
    with torch.inference_mode():
        torch_output = wrapper(latent, sr_cond).detach().cpu().numpy()

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(args.onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    ort_output = session.run(["waveform"], {"latent": latent_np, "sr_cond": sr_cond_np})[0]

    abs_diff = np.abs(torch_output - ort_output)
    result = {
        "torch_shape": list(torch_output.shape),
        "ort_shape": list(ort_output.shape),
        "dtype": str(ort_output.dtype),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
    }
    if result["max_abs_diff"] > args.atol:
        raise AssertionError(json.dumps(result, sort_keys=True))
    return result


def test_audio_vae_decoder_parity() -> None:
    import os
    import pytest

    onnx_path = os.environ.get("VOXCPM2_AUDIO_VAE_DECODER_ONNX")
    if not onnx_path:
        pytest.skip("set VOXCPM2_AUDIO_VAE_DECODER_ONNX to run decoder parity")
    args = _parser().parse_args(["--onnx-path", onnx_path])
    compare_audio_vae_decoder(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare AudioVAEDecoder PyTorch wrapper output against ONNX Runtime CPU output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx-path", type=Path, required=True, help="Path to audio_vae_decoder.onnx.")
    parser.add_argument("--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id.")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size for parity input.")
    parser.add_argument("--latent-steps", type=int, default=4, help="Synthetic latent time steps for parity input.")
    parser.add_argument("--seed", type=int, default=0, help="NumPy RNG seed for synthetic parity input.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Maximum allowed absolute difference.")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Require local Hugging Face cache/model files.")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only", help="Allow snapshot_download to fetch missing files.")
    return parser


def main() -> int:
    result = compare_audio_vae_decoder(_parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
