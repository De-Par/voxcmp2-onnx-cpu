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
    parser = argparse.ArgumentParser(description="Run VoxCPM2 ONNX CPU-only synthesis.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["text_only", "voice_design", "controllable_clone", "ultimate_clone"], default="text_only")
    parser.add_argument("--voice-design")
    parser.add_argument("--reference-wav", type=Path)
    parser.add_argument("--prompt-wav", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument("--model-path", default="openbmb/VoxCPM2")
    parser.add_argument("--audio-encoder-onnx", type=Path, default=OnnxModelPaths().audio_encoder)
    parser.add_argument("--audio-decoder-onnx", type=Path, default=OnnxModelPaths().audio_decoder)
    parser.add_argument("--prefill-onnx", type=Path, default=OnnxModelPaths().prefill)
    parser.add_argument("--decode-step-onnx", type=Path, default=OnnxModelPaths().decode_step)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
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
