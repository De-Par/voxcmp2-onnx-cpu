#!/usr/bin/env python3
"""Benchmark official VoxCPM2 API vs FP32/BF16 ONNX Runtime variants.

The script writes one WAV per requested variant and prints one compact JSON
record per result. It intentionally keeps model paths explicit so BF16 artifacts
remain an experiment and never replace the FP32 runtime defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SRC = REPO_ROOT / "third_party" / "VoxCPM" / "src"
sys.path.insert(0, str(REPO_ROOT))

from src.runtime.pipeline import VoxCPM2OnnxPipeline
from src.runtime.session_factory import OnnxModelPaths


Variant = Literal["orig", "onnx_fp32", "onnx_bf16"]


BF16_MODEL_PATHS = OnnxModelPaths(
    audio_encoder=REPO_ROOT / "artifacts" / "bf16_experiment" / "audio_vae_encoder" / "audio_vae_encoder.onnx",
    audio_decoder=REPO_ROOT / "artifacts" / "bf16_experiment" / "audio_vae_decoder" / "audio_vae_decoder.onnx",
    prefill=REPO_ROOT / "artifacts" / "bf16_experiment" / "prefill" / "voxcpm2_prefill.onnx",
    decode_step=REPO_ROOT / "artifacts" / "bf16_experiment" / "decode_step" / "voxcpm2_decode_step.onnx",
)


@dataclass(frozen=True)
class BenchResult:
    variant: Variant
    ok: bool
    output_wav: str | None
    load_seconds: float | None
    synth_seconds: float | None
    total_seconds: float
    sample_rate: int | None
    samples: int | None
    error: str | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "ok": self.ok,
            "output_wav": self.output_wav,
            "load_seconds": self.load_seconds,
            "synth_seconds": self.synth_seconds,
            "total_seconds": self.total_seconds,
            "sample_rate": self.sample_rate,
            "samples": self.samples,
            "error": self.error,
        }


def _install_upstream_import_path() -> None:
    if UPSTREAM_SRC.exists():
        sys.path.insert(0, str(UPSTREAM_SRC))


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
    pcm16 = (np.clip(mono, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm16)


def _mode_text(args: argparse.Namespace) -> str:
    if args.mode == "voice_design" and args.voice_design:
        return f"({args.voice_design}){args.text}"
    return args.text


def _validate_mode_args(args: argparse.Namespace) -> None:
    if args.mode == "text_only":
        if args.voice_design or args.reference_wav or args.prompt_wav or args.prompt_text:
            raise ValueError("text_only must not use voice/reference/prompt arguments")
    elif args.mode == "voice_design":
        if not args.voice_design:
            raise ValueError("voice_design requires --voice-design")
        if args.reference_wav or args.prompt_wav or args.prompt_text:
            raise ValueError("voice_design must not use reference/prompt arguments")
    elif args.mode == "controllable_clone":
        if not args.reference_wav:
            raise ValueError("controllable_clone requires --reference-wav")
        if args.prompt_wav or args.prompt_text:
            raise ValueError("controllable_clone must not use prompt arguments")
    elif args.mode == "ultimate_clone":
        if not args.prompt_wav or args.prompt_text is None:
            raise ValueError("ultimate_clone requires --prompt-wav and --prompt-text")


def _onnx_paths(args: argparse.Namespace, variant: Variant) -> OnnxModelPaths:
    defaults = OnnxModelPaths() if variant == "onnx_fp32" else BF16_MODEL_PATHS
    prefix = "fp32" if variant == "onnx_fp32" else "bf16"
    return OnnxModelPaths(
        audio_encoder=getattr(args, f"{prefix}_audio_encoder_onnx") or defaults.audio_encoder,
        audio_decoder=getattr(args, f"{prefix}_audio_decoder_onnx") or defaults.audio_decoder,
        prefill=getattr(args, f"{prefix}_prefill_onnx") or defaults.prefill,
        decode_step=getattr(args, f"{prefix}_decode_step_onnx") or defaults.decode_step,
    )


def _run_onnx(args: argparse.Namespace, variant: Variant, output_wav: Path) -> BenchResult:
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    pipeline = VoxCPM2OnnxPipeline.from_default_artifacts(
        model_path=args.model_path,
        local_files_only=args.local_files_only,
        onnx_paths=_onnx_paths(args, variant),
    )
    pipeline.validate()
    load_seconds = time.perf_counter() - load_start

    synth_start = time.perf_counter()
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
    synth_seconds = time.perf_counter() - synth_start
    pipeline.write_wav(output_wav, waveform)
    return BenchResult(
        variant=variant,
        ok=True,
        output_wav=str(output_wav),
        load_seconds=round(load_seconds, 6),
        synth_seconds=round(synth_seconds, 6),
        total_seconds=round(time.perf_counter() - total_start, 6),
        sample_rate=pipeline.config.decode_sample_rate,
        samples=int(waveform.shape[0]),
    )


def _run_orig(args: argparse.Namespace, output_wav: Path) -> BenchResult:
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    _install_upstream_import_path()

    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained(
        args.model_path,
        load_denoiser=False,
        local_files_only=args.local_files_only,
        optimize=False,
        device=args.orig_device,
    )
    sample_rate = int(getattr(model.tts_model, "sample_rate", 48000))
    load_seconds = time.perf_counter() - load_start

    synth_start = time.perf_counter()
    waveform = model.generate(
        text=_mode_text(args),
        prompt_wav_path=str(args.prompt_wav) if args.prompt_wav else None,
        prompt_text=args.prompt_text,
        reference_wav_path=str(args.reference_wav) if args.reference_wav else None,
        cfg_value=args.cfg_value,
        inference_timesteps=args.orig_inference_timesteps,
        min_len=args.orig_min_len,
        max_len=args.orig_max_len,
        normalize=args.normalize,
        denoise=False,
        retry_badcase=args.orig_retry_badcase,
        retry_badcase_max_times=args.orig_retry_badcase_max_times,
        retry_badcase_ratio_threshold=args.orig_retry_badcase_ratio_threshold,
    )
    synth_seconds = time.perf_counter() - synth_start
    _write_wav(output_wav, waveform, sample_rate)
    return BenchResult(
        variant="orig",
        ok=True,
        output_wav=str(output_wav),
        load_seconds=round(load_seconds, 6),
        synth_seconds=round(synth_seconds, 6),
        total_seconds=round(time.perf_counter() - total_start, 6),
        sample_rate=sample_rate,
        samples=int(np.asarray(waveform).reshape(-1).shape[0]),
    )


def _run_variant(args: argparse.Namespace, variant: Variant) -> BenchResult:
    output_wav = args.output_dir.expanduser() / f"{variant}_{args.mode}.wav"
    try:
        if variant == "orig":
            return _run_orig(args, output_wav)
        return _run_onnx(args, variant, output_wav)
    except Exception as exc:  # noqa: BLE001 - a benchmark should report failed variants and continue.
        return BenchResult(
            variant=variant,
            ok=False,
            output_wav=str(output_wav),
            load_seconds=None,
            synth_seconds=None,
            total_seconds=0.0,
            sample_rate=None,
            samples=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def run(args: argparse.Namespace) -> list[BenchResult]:
    _validate_mode_args(args)
    args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    results = [_run_variant(args, variant) for variant in args.variants]
    if args.report_json:
        args.report_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.report_json.expanduser().write_text(
            json.dumps([result.as_json() for result in results], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark official VoxCPM2 API, ONNX FP32, and experimental ONNX BF16 pipelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text", required=True, help="Target text to synthesize.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["orig", "onnx_fp32", "onnx_bf16"],
        default=["orig", "onnx_fp32", "onnx_bf16"],
        help="Pipeline variants to benchmark in order.",
    )
    parser.add_argument(
        "--mode",
        choices=["text_only", "voice_design", "controllable_clone", "ultimate_clone"],
        default="text_only",
        help="Synthesis mode used for every selected variant.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bench"), help="Directory for result WAVs.")
    parser.add_argument("--report-json", type=Path, help="Optional JSON summary path.")
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument("--voice-design", help="Style/control text for voice_design mode.")
    parser.add_argument("--reference-wav", type=Path, help="Reference WAV for controllable_clone or ultimate_clone.")
    parser.add_argument("--prompt-wav", type=Path, help="Prompt WAV for ultimate_clone.")
    parser.add_argument("--prompt-text", help="Prompt text for ultimate_clone.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="NumPy RNG seed for ONNX host diffusion noise.")
    parser.add_argument("--max-steps", type=int, default=1, help="ONNX host decode-loop max steps.")
    parser.add_argument("--min-steps", type=int, default=0, help="ONNX host decode-loop min steps before stop.")

    parser.add_argument("--orig-device", default="cpu", help="Device passed to official VoxCPM.from_pretrained.")
    parser.add_argument(
        "--orig-inference-timesteps", type=int, default=10, help="Official API CFM/LocDiT solver steps."
    )
    parser.add_argument("--orig-min-len", type=int, default=2, help="Official API min generated feature length.")
    parser.add_argument("--orig-max-len", type=int, default=4096, help="Official API max generated feature length.")
    parser.add_argument("--orig-retry-badcase", action="store_true", help="Enable official retry_badcase behavior.")
    parser.add_argument("--orig-retry-badcase-max-times", type=int, default=3, help="Official retry_badcase attempts.")
    parser.add_argument(
        "--orig-retry-badcase-ratio-threshold", type=float, default=6.0, help="Official retry threshold."
    )

    parser.add_argument("--normalize", action="store_true", help="Enable official API text normalization for orig.")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Require local model files.")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only", help="Allow HF downloads.")

    parser.add_argument("--fp32-audio-encoder-onnx", type=Path, help="Override FP32 AudioVAEEncoder ONNX path.")
    parser.add_argument("--fp32-audio-decoder-onnx", type=Path, help="Override FP32 AudioVAEDecoder ONNX path.")
    parser.add_argument("--fp32-prefill-onnx", type=Path, help="Override FP32 VoxCPM2Prefill ONNX path.")
    parser.add_argument("--fp32-decode-step-onnx", type=Path, help="Override FP32 VoxCPM2DecodeStep ONNX path.")

    parser.add_argument("--bf16-audio-encoder-onnx", type=Path, help="Override BF16 AudioVAEEncoder ONNX path.")
    parser.add_argument("--bf16-audio-decoder-onnx", type=Path, help="Override BF16 AudioVAEDecoder ONNX path.")
    parser.add_argument("--bf16-prefill-onnx", type=Path, help="Override BF16 VoxCPM2Prefill ONNX path.")
    parser.add_argument("--bf16-decode-step-onnx", type=Path, help="Override BF16 VoxCPM2DecodeStep ONNX path.")

    return parser


def main() -> int:
    for result in run(_parser().parse_args()):
        print(json.dumps(result.as_json(), sort_keys=True))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
