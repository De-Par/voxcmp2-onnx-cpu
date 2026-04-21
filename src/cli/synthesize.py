#!/usr/bin/env python3
"""Synthesize speech with the CPU-only VoxCPM2 ONNX Runtime pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.runtime.pipeline import VoxCPM2OnnxPipeline
from src.runtime.session_factory import OnnxModelPaths


def _parser() -> argparse.ArgumentParser:
    defaults = OnnxModelPaths()
    parser = argparse.ArgumentParser(
        description="Run the VoxCPM2 CPU-only ONNX Runtime synthesis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/cli/synthesize.py --text 'Hello from VoxCPM2.' "
            "--output artifacts/runtime_smoke.wav --max-steps 1 --mode text_only"
        ),
    )
    parser.add_argument("--text", required=True, help="Target text to synthesize.")
    parser.add_argument("--output", type=Path, required=True, help="Destination WAV path written by host code.")
    parser.add_argument(
        "--mode",
        choices=["text_only", "voice_design", "controllable_clone", "ultimate_clone"],
        default="text_only",
        help="Runtime orchestration mode. Reference and prompt arguments are validated by the pipeline.",
    )
    parser.add_argument("--voice-design", help="Optional style/control text prepended for voice_design mode.")
    parser.add_argument("--reference-wav", type=Path, help="Reference WAV path for controllable_clone or ultimate_clone.")
    parser.add_argument("--prompt-wav", type=Path, help="Prompt/continuation WAV path for ultimate_clone.")
    parser.add_argument("--prompt-text", help="Prompt/continuation text paired with --prompt-wav for ultimate_clone.")
    parser.add_argument("--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id.")
    parser.add_argument("--audio-encoder-onnx", type=Path, default=defaults.audio_encoder, help="AudioVAEEncoder ONNX file.")
    parser.add_argument("--audio-decoder-onnx", type=Path, default=defaults.audio_decoder, help="AudioVAEDecoder ONNX file.")
    parser.add_argument("--prefill-onnx", type=Path, default=defaults.prefill, help="VoxCPM2Prefill ONNX file.")
    parser.add_argument("--decode-step-onnx", type=Path, default=defaults.decode_step, help="VoxCPM2DecodeStep ONNX file.")
    parser.add_argument("--max-steps", type=int, default=1, help="Maximum host decode-loop iterations.")
    parser.add_argument("--min-steps", type=int, default=0, help="Minimum decode steps before stop logits may end synthesis.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value passed to decode_step.")
    parser.add_argument("--seed", type=int, default=0, help="NumPy RNG seed for host-supplied diffusion noise.")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Require local model files.")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only", help="Allow Hugging Face downloads.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = OnnxModelPaths(
        audio_encoder=args.audio_encoder_onnx,
        audio_decoder=args.audio_decoder_onnx,
        prefill=args.prefill_onnx,
        decode_step=args.decode_step_onnx,
    )
    pipeline = VoxCPM2OnnxPipeline.from_default_artifacts(
        model_path=args.model_path,
        local_files_only=args.local_files_only,
        onnx_paths=paths,
    )
    pipeline.validate()
    waveform = pipeline.synthesize(
        args.text,
        mode=args.mode,
        voice_design=args.voice_design,
        reference_wav_path=args.reference_wav,
        prompt_wav_path=args.prompt_wav,
        prompt_text=args.prompt_text,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        cfg_value=args.cfg_value,
        seed=args.seed,
    )
    pipeline.write_wav(args.output, waveform)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_rate": pipeline.config.decode_sample_rate,
                "samples": int(waveform.shape[0]),
                "sessions": list(pipeline.sessions.created_session_names),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
