#!/usr/bin/env python3
"""Demo CLI for the portable VoxCPM2 ONNX application API"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _api_classes():
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from app.voxcpm2_onnx import VoxCPM2Onnx, VoxCPM2OnnxConfig

    return VoxCPM2Onnx, VoxCPM2OnnxConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VoxCPM2 ONNX CPU synthesis through the minimal app API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text", required=True, help="Target text to synthesize.")
    parser.add_argument("--output", type=Path, required=True, help="Destination WAV path.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16", help="ONNX artifact family.")
    parser.add_argument(
        "--mode",
        choices=["text_only", "voice_design", "controllable_clone", "ultimate_clone"],
        default="text_only",
        help="VoxCPM2 synthesis mode.",
    )
    parser.add_argument("--voice-design", help="Style/control text for voice_design mode.")
    parser.add_argument("--reference-wav", type=Path, help="Reference WAV for controllable_clone/ultimate_clone.")
    parser.add_argument("--prompt-wav", type=Path, help="Prompt WAV for ultimate_clone.")
    parser.add_argument("--prompt-text", help="Prompt text for ultimate_clone.")
    parser.add_argument("--model-path", default="openbmb/VoxCPM2", help="Local model directory or Hugging Face id.")
    parser.add_argument(
        "--onnx-root", type=Path, default=Path("models/onnx"), help="Root containing fp32/bf16 ONNX dirs."
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Max decode steps. 0 means run until stop logits.")
    parser.add_argument("--min-steps", type=int, default=8, help="Minimum decode steps before stop logits may end.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="Host diffusion-noise RNG seed.")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Require local model files.")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only", help="Allow HF downloads.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    VoxCPM2Onnx, VoxCPM2OnnxConfig = _api_classes()
    api = VoxCPM2Onnx(
        VoxCPM2OnnxConfig(
            precision=args.precision,
            model_path=args.model_path,
            local_files_only=args.local_files_only,
            onnx_root=args.onnx_root,
        )
    )
    api.validate()
    result = api.synthesize(
        args.text,
        mode=args.mode,
        voice_design=args.voice_design,
        reference_wav_path=args.reference_wav,
        prompt_wav_path=args.prompt_wav,
        prompt_text=args.prompt_text,
        output_wav=args.output,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        cfg_value=args.cfg_value,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "precision": args.precision,
                "mode": args.mode,
                "samples": int(result.waveform.shape[0]),
                "decode_steps": result.metadata.decode_steps,
                "stop_reason": result.metadata.stop_reason,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
