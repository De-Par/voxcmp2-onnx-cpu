#!/usr/bin/env python3
"""Benchmark official VoxCPM2 API vs FP32/BF16 ONNX Runtime variants.

The script writes one WAV per requested variant, prints a compact human-readable
summary, and saves machine-readable JSON to disk. It intentionally keeps model
paths explicit so BF16 artifacts remain an experiment and never replace the FP32
runtime defaults.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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


Variant = Literal["orig", "onnx_fp32", "onnx_bf16"]
SILENCE_PEAK_THRESHOLD = 1e-5


BF16_MODEL_PATH_VALUES = {
    "audio_encoder": REPO_ROOT / "models" / "onnx" / "bf16" / "audio_vae_encoder" / "audio_vae_encoder.onnx",
    "audio_decoder": REPO_ROOT / "models" / "onnx" / "bf16" / "audio_vae_decoder" / "audio_vae_decoder.onnx",
    "prefill": REPO_ROOT / "models" / "onnx" / "bf16" / "prefill" / "voxcpm2_prefill.onnx",
    "decode_chunk": REPO_ROOT / "models" / "onnx" / "bf16" / "decode_chunk" / "voxcpm2_decode_chunk.onnx",
}


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _runtime_classes():
    _ensure_repo_root_on_path()

    from src.runtime.pipeline import VoxCPM2OnnxPipeline, VoxCPM2RuntimeConfig
    from src.runtime.session_factory import (
        EXECUTION_MODE_CHOICES,
        GRAPH_OPTIMIZATION_CHOICES,
        LOG_SEVERITY_CHOICES,
        OnnxModelPaths,
    )

    return (
        VoxCPM2OnnxPipeline,
        VoxCPM2RuntimeConfig,
        OnnxModelPaths,
        GRAPH_OPTIMIZATION_CHOICES,
        EXECUTION_MODE_CHOICES,
        LOG_SEVERITY_CHOICES,
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
    duration_seconds: float | None
    peak: float | None
    rms: float | None
    decode_steps: int | None = None
    stop_reason: str | None = None
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
            "duration_seconds": self.duration_seconds,
            "peak": self.peak,
            "rms": self.rms,
            "decode_steps": self.decode_steps,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }


def _install_upstream_import_path() -> None:
    if UPSTREAM_SRC.exists():
        sys.path.insert(0, str(UPSTREAM_SRC))


def _coerce_waveform(waveform: np.ndarray) -> np.ndarray:
    mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
    return np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = _coerce_waveform(waveform)
    pcm16 = (np.clip(mono, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm16)
    return mono


def _duration_seconds(samples: int, sample_rate: int) -> float:
    return round(samples / sample_rate, 6)


def _audio_stats(waveform: np.ndarray) -> tuple[float, float]:
    mono = _coerce_waveform(waveform)
    if mono.size == 0:
        return 0.0, 0.0
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    return round(peak, 8), round(rms, 8)


def _variant_output_tail(stdout: io.StringIO, stderr: io.StringIO) -> str:
    text = "\n".join(part.strip() for part in (stdout.getvalue(), stderr.getvalue()) if part.strip())
    if not text:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-20:])


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


def _onnx_paths(args: argparse.Namespace, variant: Variant):
    _, _, OnnxModelPaths, _, _, _ = _runtime_classes()
    defaults = OnnxModelPaths() if variant == "onnx_fp32" else OnnxModelPaths(**BF16_MODEL_PATH_VALUES)
    prefix = "fp32" if variant == "onnx_fp32" else "bf16"
    return OnnxModelPaths(
        audio_encoder=getattr(args, f"{prefix}_audio_encoder_onnx") or defaults.audio_encoder,
        audio_decoder=getattr(args, f"{prefix}_audio_decoder_onnx") or defaults.audio_decoder,
        prefill=getattr(args, f"{prefix}_prefill_onnx") or defaults.prefill,
        decode_chunk=getattr(args, f"{prefix}_decode_chunk_onnx") or defaults.decode_chunk,
    )


def _preload_onnx_sessions(pipeline, mode: str) -> None:
    # Runtime sessions stay lazy in production. Benchmarking preloads only the
    # sessions used by the selected mode so load/synth timings are interpretable.
    _ = pipeline.sessions.prefill
    _ = pipeline.sessions.decode_chunk
    _ = pipeline.sessions.audio_decoder
    if mode in {"controllable_clone", "ultimate_clone"}:
        _ = pipeline.sessions.audio_encoder


def _run_onnx(args: argparse.Namespace, variant: Variant, output_wav: Path) -> BenchResult:
    VoxCPM2OnnxPipeline, _, _, _, _, _ = _runtime_classes()
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    pipeline = VoxCPM2OnnxPipeline.from_default_artifacts(
        model_path=args.model_path,
        local_files_only=args.local_files_only,
        onnx_paths=_onnx_paths(args, variant),
        graph_optimization_level=args.onnx_graph_optimization,
        execution_mode=args.onnx_execution_mode,
        log_severity_level=args.onnx_log_severity,
        intra_op_num_threads=args.onnx_intra_op_threads,
        inter_op_num_threads=args.onnx_inter_op_threads,
        max_audio_encoder_samples=args.max_audio_encoder_samples,
        max_decoder_latent_steps=args.max_decoder_latent_steps,
        max_prefill_seq_len=args.max_prefill_seq_len,
        max_decode_cache_seq=args.max_decode_cache_seq,
    )
    pipeline.validate()
    if args.onnx_preload_sessions:
        _preload_onnx_sessions(pipeline, args.mode)
    load_seconds = time.perf_counter() - load_start

    synth_start = time.perf_counter()
    result = pipeline.synthesize_with_metadata(
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
        progress_callback=_make_decode_progress(args, variant),
    )
    synth_seconds = time.perf_counter() - synth_start
    waveform = result.waveform
    metadata = result.metadata
    pipeline.write_wav(output_wav, waveform)
    samples = int(waveform.shape[0])
    peak, rms = _audio_stats(waveform)
    return BenchResult(
        variant=variant,
        ok=True,
        output_wav=str(output_wav),
        load_seconds=round(load_seconds, 6),
        synth_seconds=round(synth_seconds, 6),
        total_seconds=round(time.perf_counter() - total_start, 6),
        sample_rate=pipeline.config.decode_sample_rate,
        samples=samples,
        duration_seconds=_duration_seconds(samples, pipeline.config.decode_sample_rate),
        peak=peak,
        rms=rms,
        decode_steps=metadata.decode_steps,
        stop_reason=metadata.stop_reason,
    )


def _run_orig(args: argparse.Namespace, output_wav: Path) -> BenchResult:
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    _install_upstream_import_path()

    from voxcpm import VoxCPM

    # Official VoxCPM2 samples diffusion noise with torch.randn inside the model.
    # Seeding here makes benchmark repeats stable and narrows one source of
    # mismatch against the ONNX host-supplied NumPy diffusion noise path.
    import random

    import torch

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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    waveform = _write_wav(output_wav, waveform, sample_rate)
    samples = int(waveform.shape[0])
    peak, rms = _audio_stats(waveform)
    return BenchResult(
        variant="orig",
        ok=True,
        output_wav=str(output_wav),
        load_seconds=round(load_seconds, 6),
        synth_seconds=round(synth_seconds, 6),
        total_seconds=round(time.perf_counter() - total_start, 6),
        sample_rate=sample_rate,
        samples=samples,
        duration_seconds=_duration_seconds(samples, sample_rate),
        peak=peak,
        rms=rms,
    )


def _call_variant(args: argparse.Namespace, variant: Variant, output_wav: Path) -> BenchResult:
    if variant == "orig":
        return _run_orig(args, output_wav)
    return _run_onnx(args, variant, output_wav)


def _run_variant(args: argparse.Namespace, variant: Variant) -> BenchResult:
    output_wav = args.output_dir.expanduser() / f"{variant}_{args.mode}.wav"
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        if variant == "orig" and not args.show_variant_output:
            # Official VoxCPM2 uses tqdm/logging internally. Capturing it keeps
            # the benchmark output stable and avoids half-finished progress bars
            # when the model stops generation early.
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return _call_variant(args, variant, output_wav)
        return _call_variant(args, variant, output_wav)
    except Exception as exc:  # noqa: BLE001 - a benchmark should report failed variants and continue.
        captured_tail = _variant_output_tail(stdout, stderr)
        error = f"{type(exc).__name__}: {exc}"
        if captured_tail:
            error = f"{error}\nCaptured output tail:\n{captured_tail}"
        return BenchResult(
            variant=variant,
            ok=False,
            output_wav=str(output_wav),
            load_seconds=None,
            synth_seconds=None,
            total_seconds=0.0,
            sample_rate=None,
            samples=None,
            duration_seconds=None,
            peak=None,
            rms=None,
            error=error,
        )


def _make_decode_progress(args: argparse.Namespace, variant: Variant):
    if args.progress_every <= 0:
        return None

    def _callback(completed_steps: int, stop_reason: str | None) -> None:
        should_print = completed_steps == 1 or completed_steps % args.progress_every == 0 or stop_reason is not None
        if not should_print:
            return
        limit = "auto" if args.max_steps == 0 else str(args.max_steps)
        suffix = f", stop={stop_reason}" if stop_reason else ""
        print(f"  decode {variant}: step {completed_steps}/{limit}{suffix}", flush=True)

    return _callback


def _report_path(args: argparse.Namespace) -> Path:
    return (args.report_json or (args.output_dir / "report.json")).expanduser()


def _print_header(args: argparse.Namespace) -> None:
    _, VoxCPM2RuntimeConfig, _, _, _, _ = _runtime_classes()
    max_steps_text = (
        f"auto-until-stop (safety cap: {VoxCPM2RuntimeConfig().decode_safety_max_steps})"
        if args.max_steps == 0
        else str(args.max_steps)
    )
    print("=" * 72, flush=True)
    print("VoxCPM2 benchmark", flush=True)
    print("=" * 72, flush=True)
    print(f"mode          : {args.mode}", flush=True)
    print(f"variants      : {', '.join(args.variants)}", flush=True)
    print(f"output_dir    : {args.output_dir.expanduser()}", flush=True)
    print(f"json_report   : {_report_path(args)}", flush=True)
    print(f"ONNX decode   : max_steps={max_steps_text}, min_steps={args.min_steps}", flush=True)
    print(
        "ONNX ORT      : "
        f"provider=CPUExecutionProvider, graph_opt={args.onnx_graph_optimization}, "
        f"execution={args.onnx_execution_mode}, "
        f"log={args.onnx_log_severity}, "
        f"preload_sessions={'yes' if args.onnx_preload_sessions else 'no'}, "
        f"intra_op={args.onnx_intra_op_threads if args.onnx_intra_op_threads is not None else 'default'}, "
        f"inter_op={args.onnx_inter_op_threads if args.onnx_inter_op_threads is not None else 'default'}",
        flush=True,
    )
    print(
        "official API  : "
        f"max_len={args.orig_max_len}, min_len={args.orig_min_len}, "
        f"inference_timesteps={args.orig_inference_timesteps}",
        flush=True,
    )
    if not args.show_variant_output:
        print("variant logs  : hidden; pass --show-variant-output to debug upstream output", flush=True)
    print(flush=True)


def _print_result(result: BenchResult) -> None:
    status = "OK" if result.ok else "FAIL"
    print(f"[{status}] {result.variant}", flush=True)
    if not result.ok:
        print(f"  error: {result.error}", flush=True)
        print(flush=True)
        return
    print(f"  WAV      : {result.output_wav}", flush=True)
    print(
        "  Audio    : "
        f"{result.duration_seconds:.3f}s | {result.samples} samples @ {result.sample_rate} Hz | "
        f"peak={result.peak:.6f}, rms={result.rms:.6f}",
        flush=True,
    )
    if result.decode_steps is not None:
        print(f"  Decode   : {result.decode_steps} step(s), stop={result.stop_reason}", flush=True)
    if result.peak is not None and result.peak < SILENCE_PEAK_THRESHOLD:
        print("  Warning  : very low peak amplitude; WAV may be silent", flush=True)
    print(
        "  Time     : "
        f"load={result.load_seconds:.3f}s, synth={result.synth_seconds:.3f}s, "
        f"total={result.total_seconds:.3f}s",
        flush=True,
    )
    print(flush=True)


def run(args: argparse.Namespace, *, progress: bool = False) -> list[BenchResult]:
    _validate_mode_args(args)
    args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    if progress:
        _print_header(args)
    results = []
    for index, variant in enumerate(args.variants, start=1):
        if progress:
            print(f"[{index}/{len(args.variants)}] running {variant}", flush=True)
        result = _run_variant(args, variant)
        results.append(result)
        if progress:
            _print_result(result)

    report_path = _report_path(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([result.as_json() for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if progress:
        _print_summary(results)
        print(f"json saved: {report_path}", flush=True)
    return results


def _print_summary(results: list[BenchResult]) -> None:
    print("Summary", flush=True)
    print("-" * 72, flush=True)
    for result in results:
        if not result.ok:
            print(f"{result.variant:10} FAIL", flush=True)
            continue
        decode = f"{result.decode_steps} steps, {result.stop_reason}" if result.decode_steps is not None else "-"
        print(
            f"{result.variant:10} "
            f"{result.duration_seconds:7.3f}s  "
            f"peak={result.peak:.6f}  "
            f"synth={result.synth_seconds:.3f}s  "
            f"decode={decode}  "
            f"wav={result.output_wav}",
            flush=True,
        )
    print(flush=True)


def _parser() -> argparse.ArgumentParser:
    _, _, _, graph_optimization_choices, execution_mode_choices, log_severity_choices = _runtime_classes()
    parser = argparse.ArgumentParser(
        description="Benchmark official VoxCPM2 API, ONNX FP32, and ONNX BF16 pipelines.",
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
    parser.add_argument(
        "--report-json",
        type=Path,
        help="JSON summary path. Defaults to <output-dir>/report.json.",
    )
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument("--voice-design", help="Style/control text for voice_design mode.")
    parser.add_argument("--reference-wav", type=Path, help="Reference WAV for controllable_clone or ultimate_clone.")
    parser.add_argument("--prompt-wav", type=Path, help="Prompt WAV for ultimate_clone.")
    parser.add_argument("--prompt-text", help="Prompt text for ultimate_clone.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for benchmark RNGs: NumPy host diffusion noise and official PyTorch diffusion noise.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help=(
            "ONNX host decode-loop max steps. 0 means run until stop logits, with an internal safety cap. "
            "One step is roughly 0.16 s of decoded audio."
        ),
    )
    parser.add_argument("--min-steps", type=int, default=8, help="ONNX host decode-loop min steps before stop.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=4,
        help="Print ONNX decode progress every N completed steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--show-variant-output",
        action="store_true",
        help="Do not suppress official API logs/progress bars; useful only for debugging.",
    )
    parser.add_argument(
        "--onnx-graph-optimization",
        choices=graph_optimization_choices,
        default="disable",
        help=(
            "ONNX Runtime graph optimization level for ONNX variants. "
            "Default stays conservative for FP32 parity; use all/extended for performance experiments."
        ),
    )
    parser.add_argument(
        "--onnx-execution-mode",
        choices=execution_mode_choices,
        default="sequential",
        help="ONNX Runtime execution mode for ONNX variants.",
    )
    parser.add_argument(
        "--onnx-log-severity",
        choices=log_severity_choices,
        default="error",
        help=(
            "Minimum ONNX Runtime log severity for ONNX variants. The benchmark defaults to error "
            "to keep perf output readable; use warning/info when debugging ORT graph optimization."
        ),
    )
    parser.add_argument(
        "--onnx-preload-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Create ONNX Runtime sessions during the load phase. Disable with --no-onnx-preload-sessions "
            "to measure first-request latency."
        ),
    )
    parser.add_argument(
        "--onnx-intra-op-threads",
        type=int,
        help="ONNX Runtime intra-op thread count. Omit for ORT default; 0 also requests ORT default scheduling.",
    )
    parser.add_argument(
        "--onnx-inter-op-threads",
        type=int,
        help="ONNX Runtime inter-op thread count. Omit for ORT default; 0 also requests ORT default scheduling.",
    )
    parser.add_argument(
        "--max-audio-encoder-samples", type=int, help="Runtime/export bound for reference audio samples."
    )
    parser.add_argument("--max-decoder-latent-steps", type=int, help="Runtime/export bound for decoder latent steps.")
    parser.add_argument("--max-prefill-seq-len", type=int, help="Runtime/export bound for prefill sequence length.")
    parser.add_argument("--max-decode-cache-seq", type=int, help="Runtime/export bound for decode cache capacity.")

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
    parser.add_argument("--fp32-decode-chunk-onnx", type=Path, help="Override FP32 VoxCPM2DecodeChunk ONNX path.")

    parser.add_argument("--bf16-audio-encoder-onnx", type=Path, help="Override BF16 AudioVAEEncoder ONNX path.")
    parser.add_argument("--bf16-audio-decoder-onnx", type=Path, help="Override BF16 AudioVAEDecoder ONNX path.")
    parser.add_argument("--bf16-prefill-onnx", type=Path, help="Override BF16 VoxCPM2Prefill ONNX path.")
    parser.add_argument("--bf16-decode-chunk-onnx", type=Path, help="Override BF16 VoxCPM2DecodeChunk ONNX path.")

    return parser


def main() -> int:
    run(_parser().parse_args(), progress=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
