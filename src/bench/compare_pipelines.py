#!/usr/bin/env python3
"""Benchmark official VoxCPM2 API vs FP32/BF16 ONNX Runtime variants.

The script writes one WAV per requested variant, prints a compact human-readable
summary, and saves machine-readable JSON to disk. It intentionally keeps model
paths explicit so FP32 and BF16 production artifacts can be compared without
changing the single runtime implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
    iteration: int
    ok: bool
    output_wav: str | None
    load_seconds: float | None
    synth_seconds: float | None
    wall_seconds: float
    sample_rate: int | None
    samples: int | None
    duration_seconds: float | None
    peak: float | None
    rms: float | None
    decode_steps: int | None = None
    stop_reason: str | None = None
    prefill_seconds: float | None = None
    decode_seconds: float | None = None
    decoder_seconds: float | None = None
    error: str | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "iteration": self.iteration,
            "ok": self.ok,
            "output_wav": self.output_wav,
            "load_seconds": self.load_seconds,
            "synth_seconds": self.synth_seconds,
            "wall_seconds": self.wall_seconds,
            "sample_rate": self.sample_rate,
            "samples": self.samples,
            "duration_seconds": self.duration_seconds,
            "peak": self.peak,
            "rms": self.rms,
            "decode_steps": self.decode_steps,
            "stop_reason": self.stop_reason,
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "decoder_seconds": self.decoder_seconds,
            "error": self.error,
        }


@dataclass(frozen=True)
class LoadedVariant:
    variant: Variant
    model: Any
    load_seconds: float
    sample_rate: int | None = None


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


def _load_onnx(args: argparse.Namespace, variant: Variant) -> LoadedVariant:
    VoxCPM2OnnxPipeline, _, _, _, _, _ = _runtime_classes()
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
        enable_mem_pattern=args.onnx_enable_mem_pattern,
        enable_cpu_mem_arena=args.onnx_enable_cpu_mem_arena,
        enable_mem_reuse=args.onnx_enable_mem_reuse,
        prefer_optimized_onnx=args.onnx_prefer_optimized_artifacts,
        enable_decode_chunk_iobinding=args.onnx_decode_chunk_iobinding,
        max_audio_encoder_samples=args.max_audio_encoder_samples,
        max_decoder_latent_steps=args.max_decoder_latent_steps,
        max_prefill_seq_len=args.max_prefill_seq_len,
        max_decode_cache_seq=args.max_decode_cache_seq,
    )
    pipeline.validate()
    if args.onnx_preload_sessions:
        _preload_onnx_sessions(pipeline, args.mode)
    return LoadedVariant(variant=variant, model=pipeline, load_seconds=round(time.perf_counter() - load_start, 6))


def _run_onnx_loaded(
    args: argparse.Namespace,
    loaded: LoadedVariant,
    output_wav: Path,
    *,
    iteration: int,
) -> BenchResult:
    synth_start = time.perf_counter()
    pipeline = loaded.model
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
        progress_callback=_make_decode_progress(args, loaded.variant),
    )
    synth_seconds = time.perf_counter() - synth_start
    waveform = result.waveform
    metadata = result.metadata
    pipeline.write_wav(output_wav, waveform)
    samples = int(waveform.shape[0])
    peak, rms = _audio_stats(waveform)
    return BenchResult(
        variant=loaded.variant,
        iteration=iteration,
        ok=True,
        output_wav=str(output_wav),
        load_seconds=loaded.load_seconds,
        synth_seconds=round(synth_seconds, 6),
        wall_seconds=round(synth_seconds, 6),
        sample_rate=pipeline.config.decode_sample_rate,
        samples=samples,
        duration_seconds=_duration_seconds(samples, pipeline.config.decode_sample_rate),
        peak=peak,
        rms=rms,
        decode_steps=metadata.decode_steps,
        stop_reason=metadata.stop_reason,
        prefill_seconds=metadata.prefill_seconds,
        decode_seconds=metadata.decode_seconds,
        decoder_seconds=metadata.decoder_seconds,
    )


def _load_orig(args: argparse.Namespace) -> LoadedVariant:
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
    return LoadedVariant(
        variant="orig",
        model=model,
        load_seconds=round(time.perf_counter() - load_start, 6),
        sample_rate=sample_rate,
    )


def _run_orig_loaded(
    args: argparse.Namespace, loaded: LoadedVariant, output_wav: Path, *, iteration: int
) -> BenchResult:
    # Official VoxCPM2 samples diffusion noise with torch.randn inside the model.
    # Seeding here makes benchmark repeats stable and narrows one source of
    # mismatch against the ONNX host-supplied NumPy diffusion noise path.
    import random

    import torch

    synth_start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    waveform = loaded.model.generate(
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
    sample_rate = int(loaded.sample_rate or 48000)
    waveform = _write_wav(output_wav, waveform, sample_rate)
    samples = int(waveform.shape[0])
    peak, rms = _audio_stats(waveform)
    return BenchResult(
        variant="orig",
        iteration=iteration,
        ok=True,
        output_wav=str(output_wav),
        load_seconds=loaded.load_seconds,
        synth_seconds=round(synth_seconds, 6),
        wall_seconds=round(synth_seconds, 6),
        sample_rate=sample_rate,
        samples=samples,
        duration_seconds=_duration_seconds(samples, sample_rate),
        peak=peak,
        rms=rms,
    )


def _load_variant(args: argparse.Namespace, variant: Variant) -> LoadedVariant:
    if variant == "orig":
        return _load_orig(args)
    return _load_onnx(args, variant)


def _call_loaded_variant(
    args: argparse.Namespace,
    loaded: LoadedVariant,
    output_wav: Path,
    *,
    iteration: int,
) -> BenchResult:
    if loaded.variant == "orig":
        return _run_orig_loaded(args, loaded, output_wav, iteration=iteration)
    return _run_onnx_loaded(args, loaded, output_wav, iteration=iteration)


def _run_id_prefix(args: argparse.Namespace) -> str:
    if not args.run_id:
        return ""
    return f"{args.run_id}_"


def _iteration_output_wav(args: argparse.Namespace, variant: Variant, iteration: int) -> Path:
    return args.output_dir.expanduser() / f"{_run_id_prefix(args)}{variant}_{args.mode}_i{iteration:02d}.wav"


def _run_iteration(args: argparse.Namespace, loaded: LoadedVariant, iteration: int) -> BenchResult:
    output_wav = _iteration_output_wav(args, loaded.variant, iteration)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        if loaded.variant == "orig" and not args.show_variant_output:
            # Official VoxCPM2 uses tqdm/logging internally. Capturing it keeps
            # the benchmark output stable and avoids half-finished progress bars
            # when the model stops generation early.
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return _call_loaded_variant(args, loaded, output_wav, iteration=iteration)
        return _call_loaded_variant(args, loaded, output_wav, iteration=iteration)
    except Exception as exc:  # noqa: BLE001 - a benchmark should report failed variants and continue.
        captured_tail = _variant_output_tail(stdout, stderr)
        error = f"{type(exc).__name__}: {exc}"
        if captured_tail:
            error = f"{error}\nCaptured output tail:\n{captured_tail}"
        return BenchResult(
            variant=loaded.variant,
            iteration=iteration,
            ok=False,
            output_wav=str(output_wav),
            load_seconds=loaded.load_seconds,
            synth_seconds=None,
            wall_seconds=0.0,
            sample_rate=None,
            samples=None,
            duration_seconds=None,
            peak=None,
            rms=None,
            prefill_seconds=None,
            decode_seconds=None,
            decoder_seconds=None,
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
    default_name = f"report_{args.run_id}.json" if args.run_id else "report.json"
    return (args.report_json or (args.output_dir / default_name)).expanduser()


def _selected_variant(args: argparse.Namespace) -> Variant:
    if args.variants is not None:
        if len(args.variants) != 1:
            raise ValueError("quick bench runs exactly one variant; use --variant or pass a single --variants value")
        if args.variant is not None:
            raise ValueError("use either --variant or legacy --variants, not both")
        return args.variants[0]
    if args.variant is None:
        raise ValueError("--variant is required")
    return args.variant


def _measure_data_load(args: argparse.Namespace) -> float:
    """Measure input data availability/read cost outside model load and synth.

    This is intentionally a benchmark-side probe. The selected runtime/API still
    performs its own input handling during synthesis, so this metric should be
    read as external input data load cost, not as a hidden model stage.
    """

    start = time.perf_counter()
    if args.reference_wav:
        wavfile.read(str(args.reference_wav))
    if args.prompt_wav and args.prompt_wav != args.reference_wav:
        wavfile.read(str(args.prompt_wav))
    _ = _mode_text(args)
    return round(time.perf_counter() - start, 6)


def _print_header(args: argparse.Namespace) -> None:
    _, VoxCPM2RuntimeConfig, _, _, _, _ = _runtime_classes()
    variant = _selected_variant(args)
    max_steps_text = (
        f"auto-until-stop (safety cap: {VoxCPM2RuntimeConfig().decode_safety_max_steps})"
        if args.max_steps == 0
        else str(args.max_steps)
    )
    print("=" * 72, flush=True)
    print("VoxCPM2 benchmark", flush=True)
    print("=" * 72, flush=True)
    print(f"mode          : {args.mode}", flush=True)
    if args.run_id:
        print(f"run_id        : {args.run_id}", flush=True)
    print(f"variant       : {variant}", flush=True)
    print(f"iterations    : {args.iterations}", flush=True)
    print(f"output_dir    : {args.output_dir.expanduser()}", flush=True)
    print(f"json_report   : {_report_path(args)}", flush=True)
    print(f"ONNX decode   : max_steps={max_steps_text}, min_steps={args.min_steps}", flush=True)
    print(
        "ONNX ORT      : "
        f"provider=CPUExecutionProvider, graph_opt={args.onnx_graph_optimization}, "
        f"execution={args.onnx_execution_mode}, "
        f"log={args.onnx_log_severity}, "
        f"preload_sessions={'yes' if args.onnx_preload_sessions else 'no'}, "
        f"prefer_optimized_artifacts={'yes' if args.onnx_prefer_optimized_artifacts else 'no'}, "
        f"decode_iobinding={'yes' if args.onnx_decode_chunk_iobinding else 'no'}, "
        f"intra_op={args.onnx_intra_op_threads if args.onnx_intra_op_threads is not None else 'default'}, "
        f"inter_op={args.onnx_inter_op_threads if args.onnx_inter_op_threads is not None else 'default'}, "
        f"mem_pattern={args.onnx_enable_mem_pattern}, "
        f"cpu_arena={args.onnx_enable_cpu_mem_arena}, mem_reuse={args.onnx_enable_mem_reuse}",
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
    print(f"[{status}] {result.variant} iteration {result.iteration}", flush=True)
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
    if result.prefill_seconds is not None:
        print(
            "  Stages   : "
            f"prefill={result.prefill_seconds:.3f}s, "
            f"decode={result.decode_seconds:.3f}s, "
            f"decoder={result.decoder_seconds:.3f}s",
            flush=True,
        )
    if result.peak is not None and result.peak < SILENCE_PEAK_THRESHOLD:
        print("  Warning  : very low peak amplitude; WAV may be silent", flush=True)
    print(
        "  Time     : "
        f"load={result.load_seconds:.3f}s, synth={result.synth_seconds:.3f}s, "
        f"wall={result.wall_seconds:.3f}s",
        flush=True,
    )
    print(flush=True)


def run(args: argparse.Namespace, *, progress: bool = False) -> dict[str, object]:
    _validate_run_id(args.run_id)
    _validate_mode_args(args)
    variant = _selected_variant(args)
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    if progress:
        _print_header(args)

    data_load_seconds = _measure_data_load(args)
    if progress:
        print(f"data load    : {data_load_seconds:.6f}s", flush=True)
        print(f"loading      : {variant}", flush=True)
    loaded = _load_variant(args, variant)
    if progress:
        print(f"model load   : {loaded.load_seconds:.6f}s", flush=True)
        print(flush=True)

    results = []
    for iteration in range(1, args.iterations + 1):
        if progress:
            print(f"[{iteration}/{args.iterations}] running {variant}", flush=True)
        result = _run_iteration(args, loaded, iteration)
        results.append(result)
        if progress:
            _print_result(result)

    report = _make_report(args, variant, data_load_seconds, loaded.load_seconds, results)
    report_path = _report_path(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if progress:
        _print_summary(report)
        print(f"json saved: {report_path}", flush=True)
    return report


def _validate_run_id(run_id: str | None) -> None:
    if run_id is None:
        return
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("--run-id may contain only letters, digits, dot, underscore, and dash")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.array(values, dtype=np.float64), percentile)), 6)


def _stats(values: list[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "min": None, "max": None, "p50": None, "p90": None, "p95": None, "p99": None}
    return {
        "mean": round(float(np.mean(clean)), 6),
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
        "p50": _percentile(clean, 50),
        "p90": _percentile(clean, 90),
        "p95": _percentile(clean, 95),
        "p99": _percentile(clean, 99),
    }


def _make_report(
    args: argparse.Namespace,
    variant: Variant,
    data_load_seconds: float,
    model_load_seconds: float,
    results: list[BenchResult],
) -> dict[str, object]:
    ok_results = [result for result in results if result.ok]
    synth_values = [float(result.synth_seconds) for result in ok_results if result.synth_seconds is not None]
    wall_values = [float(result.wall_seconds) for result in ok_results]
    duration_values = [float(result.duration_seconds) for result in ok_results if result.duration_seconds is not None]
    sample_values = [float(result.samples) for result in ok_results if result.samples is not None]
    decode_values = [float(result.decode_steps) for result in ok_results if result.decode_steps is not None]
    prefill_values = [float(result.prefill_seconds) for result in ok_results if result.prefill_seconds is not None]
    decode_stage_values = [float(result.decode_seconds) for result in ok_results if result.decode_seconds is not None]
    decoder_values = [float(result.decoder_seconds) for result in ok_results if result.decoder_seconds is not None]
    peak_values = [float(result.peak) for result in ok_results if result.peak is not None]
    rms_values = [float(result.rms) for result in ok_results if result.rms is not None]
    return {
        "schema_version": 3,
        "variant": variant,
        "mode": args.mode,
        "iterations": args.iterations,
        "seed": args.seed,
        "data_load_seconds": data_load_seconds,
        "model_load_seconds": model_load_seconds,
        "config": {
            "max_steps": args.max_steps,
            "min_steps": args.min_steps,
            "cfg_value": args.cfg_value,
            "onnx_graph_optimization": args.onnx_graph_optimization,
            "onnx_execution_mode": args.onnx_execution_mode,
            "onnx_log_severity": args.onnx_log_severity,
            "onnx_preload_sessions": args.onnx_preload_sessions,
            "onnx_prefer_optimized_artifacts": args.onnx_prefer_optimized_artifacts,
            "onnx_decode_chunk_iobinding": args.onnx_decode_chunk_iobinding,
            "onnx_intra_op_threads": args.onnx_intra_op_threads,
            "onnx_inter_op_threads": args.onnx_inter_op_threads,
            "onnx_enable_mem_pattern": args.onnx_enable_mem_pattern,
            "onnx_enable_cpu_mem_arena": args.onnx_enable_cpu_mem_arena,
            "onnx_enable_mem_reuse": args.onnx_enable_mem_reuse,
        },
        "aggregate": {
            "ok_iterations": len(ok_results),
            "failed_iterations": len(results) - len(ok_results),
            "synth_seconds": _stats(synth_values),
            "wall_seconds": _stats(wall_values),
            "output_duration_seconds": _stats(duration_values),
            "samples": _stats(sample_values),
            "decode_steps": _stats(decode_values),
            "prefill_seconds": _stats(prefill_values),
            "decode_stage_seconds": _stats(decode_stage_values),
            "decoder_seconds": _stats(decoder_values),
            "peak": _stats(peak_values),
            "rms": _stats(rms_values),
        },
        "runs": [result.as_json() for result in results],
    }


def _print_summary(report: dict[str, object]) -> None:
    print("Summary", flush=True)
    print("-" * 72, flush=True)
    aggregate = report["aggregate"]
    synth = aggregate["synth_seconds"]
    wall = aggregate["wall_seconds"]
    duration = aggregate["output_duration_seconds"]
    decode = aggregate["decode_steps"]
    prefill = aggregate["prefill_seconds"]
    decode_stage = aggregate["decode_stage_seconds"]
    decoder = aggregate["decoder_seconds"]
    print(f"variant       : {report['variant']}", flush=True)
    print(f"data load     : {report['data_load_seconds']:.6f}s", flush=True)
    print(f"model load    : {report['model_load_seconds']:.6f}s", flush=True)
    print(f"iterations    : ok={aggregate['ok_iterations']} failed={aggregate['failed_iterations']}", flush=True)
    print(
        "synth seconds : "
        f"mean={synth['mean']} p50={synth['p50']} p90={synth['p90']} "
        f"p95={synth['p95']} p99={synth['p99']}",
        flush=True,
    )
    print(
        f"wall seconds  : mean={wall['mean']} p50={wall['p50']} p90={wall['p90']} p95={wall['p95']} p99={wall['p99']}",
        flush=True,
    )
    print(f"audio duration: mean={duration['mean']} p50={duration['p50']} p90={duration['p90']}", flush=True)
    print(f"decode steps  : mean={decode['mean']} p50={decode['p50']} p90={decode['p90']}", flush=True)
    if prefill["mean"] is not None:
        print(
            "stage seconds: "
            f"prefill mean={prefill['mean']} p50={prefill['p50']} | "
            f"decode mean={decode_stage['mean']} p50={decode_stage['p50']} | "
            f"decoder mean={decoder['mean']} p50={decoder['p50']}",
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
        "--variant",
        choices=["orig", "onnx_fp32", "onnx_bf16"],
        help="Single pipeline variant to benchmark. Multi-variant quick bench runs are intentionally disallowed.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["orig", "onnx_fp32", "onnx_bf16"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of synthesis iterations to run after one model load. Aggregates report mean/p50/p90/p95/p99.",
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
        help="JSON summary path. Defaults to <output-dir>/report.json or report_<run-id>.json.",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Optional file-name prefix for concurrent benchmark runs. When set, default WAV/report names include it."
        ),
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
        default="all",
        help="ONNX Runtime graph optimization level for ONNX variants.",
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
        "--onnx-prefer-optimized-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prefer `*.optimized.onnx` heavy-module artifacts when present. This reduces cold-start latency, "
            "but current BF16 graphs may trade startup wins for slower synth."
        ),
    )
    parser.add_argument(
        "--onnx-decode-chunk-iobinding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use ORT CPU IO binding with preallocated decode_chunk outputs. Keep this benchmark-driven; "
            "current BF16 artifacts can regress while some FP32 cases improve."
        ),
    )
    parser.add_argument(
        "--onnx-intra-op-threads",
        type=int,
        default=8,
        help="ONNX Runtime intra-op thread count. Use 0 to request ORT default scheduling.",
    )
    parser.add_argument(
        "--onnx-inter-op-threads",
        type=int,
        default=1,
        help="ONNX Runtime inter-op thread count. Use 0 to request ORT default scheduling.",
    )
    parser.add_argument(
        "--onnx-enable-mem-pattern",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ORT memory pattern planning for ONNX CPU sessions.",
    )
    parser.add_argument(
        "--onnx-enable-cpu-mem-arena",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ORT CPU memory arena for ONNX CPU sessions.",
    )
    parser.add_argument(
        "--onnx-enable-mem-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ORT memory reuse for ONNX CPU sessions.",
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
    try:
        run(_parser().parse_args(), progress=True)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
