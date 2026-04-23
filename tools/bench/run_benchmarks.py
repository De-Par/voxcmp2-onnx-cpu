#!/usr/bin/env python3
"""Production baseline benchmark for official VoxCPM2 API vs ONNX CPU runtime.

This tool does not change model math, export code, or runtime semantics. It
loads each requested variant once, runs a fixed case matrix, writes WAV outputs,
and emits JSON plus a compact Markdown report.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import random
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
Variant = Literal["official", "onnx"]
CASE_IDS = ("text_only_short", "text_only_medium", "voice_design_short", "controllable_clone_short")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    mode: str
    text: str
    voice_design: str | None = None
    needs_reference: bool = False


@dataclass(frozen=True)
class LoadedVariant:
    variant: Variant
    model: Any
    load_seconds: float


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _install_upstream_import_path() -> None:
    if UPSTREAM_SRC.exists() and str(UPSTREAM_SRC) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_SRC))


def _runtime_imports():
    _ensure_repo_root_on_path()

    from src.runtime.pipeline import DECODE_OUTPUTS, PREFILL_OUTPUTS, VoxCPM2OnnxPipeline
    from src.runtime.session_factory import (
        EXECUTION_MODE_CHOICES,
        GRAPH_OPTIMIZATION_CHOICES,
        LOG_SEVERITY_CHOICES,
        OnnxModelPaths,
    )

    return (
        VoxCPM2OnnxPipeline,
        OnnxModelPaths,
        PREFILL_OUTPUTS,
        DECODE_OUTPUTS,
        GRAPH_OPTIMIZATION_CHOICES,
        EXECUTION_MODE_CHOICES,
        LOG_SEVERITY_CHOICES,
    )


def _cases(reference_wav: Path) -> list[BenchmarkCase]:
    del reference_wav
    return [
        BenchmarkCase(
            case_id="text_only_short",
            mode="text_only",
            text="Hello from VoxCPM2.",
        ),
        BenchmarkCase(
            case_id="text_only_medium",
            mode="text_only",
            text=(
                "VoxCPM2 converts text into speech through a separated CPU-only ONNX Runtime pipeline. "
                "This medium sentence is fixed for repeatable latency measurements."
            ),
        ),
        BenchmarkCase(
            case_id="voice_design_short",
            mode="voice_design",
            text="Hello from VoxCPM2.",
            voice_design="pretty girl with sugar voice, slow",
        ),
        BenchmarkCase(
            case_id="controllable_clone_short",
            mode="controllable_clone",
            text="Hello from VoxCPM2.",
            needs_reference=True,
        ),
    ]


def _selected_cases(args: argparse.Namespace, reference_wav: Path) -> list[BenchmarkCase]:
    cases_by_id = {case.case_id: case for case in _cases(reference_wav)}
    return [cases_by_id[case_id] for case_id in args.cases]


def _mode_text(case: BenchmarkCase) -> str:
    if case.mode == "voice_design" and case.voice_design:
        return f"({case.voice_design}){case.text}"
    return case.text


def _make_reference_wav(path: Path) -> Path:
    """Create a deterministic small reference WAV for controllable-clone cases"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    seconds = 2.0
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False, dtype=np.float32)
    envelope = np.minimum(t / 0.15, 1.0) * np.minimum((seconds - t) / 0.2, 1.0)
    waveform = 0.12 * envelope * (np.sin(2.0 * np.pi * 180.0 * t) + 0.4 * np.sin(2.0 * np.pi * 360.0 * t))
    pcm16 = (np.clip(waveform, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm16)
    return path


def _percentile(values: list[float], percentile: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return round(clean[0], 6)
    pos = (len(clean) - 1) * percentile / 100.0
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))
    if lower == upper:
        return round(clean[lower], 6)
    weight = pos - lower
    return round(clean[lower] * (1.0 - weight) + clean[upper] * weight, 6)


def _stats(values: list[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None]
    return {
        "min": round(min(clean), 6) if clean else None,
        "max": round(max(clean), 6) if clean else None,
        "mean": round(float(np.mean(clean)), 6) if clean else None,
        "p50": _percentile(clean, 50),
        "p90": _percentile(clean, 90),
    }


def _audio_stats(waveform: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    mono = np.nan_to_num(np.asarray(waveform, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=-1.0)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0
    return {
        "sample_rate": int(sample_rate),
        "samples": int(mono.size),
        "duration_seconds": round(float(mono.size / sample_rate), 6) if sample_rate else 0.0,
        "peak": round(peak, 8),
        "rms": round(rms, 8),
    }


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.nan_to_num(np.asarray(waveform, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=-1.0)
    pcm16 = (np.clip(mono, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm16)


def _onnx_paths(args: argparse.Namespace):
    _, OnnxModelPaths, *_ = _runtime_imports()
    defaults = OnnxModelPaths()
    return OnnxModelPaths(
        audio_encoder=args.audio_encoder_onnx or defaults.audio_encoder,
        audio_decoder=args.audio_decoder_onnx or defaults.audio_decoder,
        prefill=args.prefill_onnx or defaults.prefill,
        decode_chunk=args.decode_chunk_onnx or defaults.decode_chunk,
    )


def _preload_onnx_sessions(pipeline: Any) -> None:
    _ = pipeline.sessions.audio_encoder
    _ = pipeline.sessions.prefill
    _ = pipeline.sessions.decode_chunk
    _ = pipeline.sessions.audio_decoder


def _load_onnx(args: argparse.Namespace) -> LoadedVariant:
    VoxCPM2OnnxPipeline, *_ = _runtime_imports()
    start = time.perf_counter()
    pipeline = VoxCPM2OnnxPipeline.from_default_artifacts(
        model_path=args.model_path,
        local_files_only=args.local_files_only,
        onnx_paths=_onnx_paths(args),
        graph_optimization_level=args.onnx_graph_optimization,
        execution_mode=args.onnx_execution_mode,
        log_severity_level=args.onnx_log_severity,
        intra_op_num_threads=args.onnx_intra_op_threads,
        inter_op_num_threads=args.onnx_inter_op_threads,
        enable_mem_pattern=args.onnx_enable_mem_pattern,
        enable_cpu_mem_arena=args.onnx_enable_cpu_mem_arena,
        enable_mem_reuse=args.onnx_enable_mem_reuse,
        max_audio_encoder_samples=args.max_audio_encoder_samples,
        max_decoder_latent_steps=args.max_decoder_latent_steps,
        max_prefill_seq_len=args.max_prefill_seq_len,
        max_decode_cache_seq=args.max_decode_cache_seq,
    )
    pipeline.validate()
    if args.onnx_preload_sessions:
        _preload_onnx_sessions(pipeline)
    return LoadedVariant(variant="onnx", model=pipeline, load_seconds=round(time.perf_counter() - start, 6))


def _load_official(args: argparse.Namespace) -> LoadedVariant:
    _install_upstream_import_path()

    from voxcpm import VoxCPM

    start = time.perf_counter()
    context = contextlib.nullcontext()
    if not args.show_official_output:
        context = contextlib.ExitStack()
        context.enter_context(contextlib.redirect_stdout(io.StringIO()))
        context.enter_context(contextlib.redirect_stderr(io.StringIO()))
    with context:
        model = VoxCPM.from_pretrained(
            args.model_path,
            load_denoiser=False,
            local_files_only=args.local_files_only,
            optimize=False,
            device=args.official_device,
        )
    return LoadedVariant(variant="official", model=model, load_seconds=round(time.perf_counter() - start, 6))


def _load_variants(args: argparse.Namespace) -> dict[Variant, LoadedVariant]:
    loaded: dict[Variant, LoadedVariant] = {}
    for variant in args.variants:
        print(f"loading {variant}...", flush=True)
        loaded[variant] = _load_official(args) if variant == "official" else _load_onnx(args)
        print(f"  load {variant}: {loaded[variant].load_seconds:.3f}s", flush=True)
    return loaded


def _run_onnx_case(
    args: argparse.Namespace,
    loaded: LoadedVariant,
    case: BenchmarkCase,
    reference_wav: Path,
    output_wav: Path,
    seed: int,
) -> dict[str, Any]:
    _, _, prefill_outputs, decode_outputs, *_ = _runtime_imports()
    pipeline = loaded.model
    if args.max_steps < 0:
        raise ValueError("max_steps must be >= 0")
    if args.min_steps < 0:
        raise ValueError("min_steps must be >= 0")
    effective_max_steps = pipeline.config.decode_safety_max_steps if args.max_steps == 0 else args.max_steps

    synth_start = time.perf_counter()
    input_start = time.perf_counter()
    sequence = pipeline.build_prefill_inputs(
        case.text,
        mode=case.mode,
        voice_design=case.voice_design,
        reference_wav_path=reference_wav if case.needs_reference else None,
        prompt_wav_path=None,
        prompt_text=None,
    )
    input_build_seconds = time.perf_counter() - input_start

    prefill_start = time.perf_counter()
    initial_cache_steps = pipeline._initial_decode_cache_steps(
        requested_max_steps=args.max_steps,
        effective_max_steps=effective_max_steps,
    )
    state = pipeline._init_fixed_capacity_decode_state(
        dict(zip(prefill_outputs, pipeline.sessions.prefill.run(prefill_outputs, sequence), strict=True)),
        max_decode_steps=initial_cache_steps,
    )
    prefill_seconds = time.perf_counter() - prefill_start

    rng = np.random.default_rng(seed)
    generated: list[np.ndarray] = []
    chunk_seconds: list[float] = []
    stop_reason = None
    completed_steps = 0
    while completed_steps < effective_max_steps:
        remaining_steps = effective_max_steps - completed_steps
        candidate_steps = min(pipeline.config.decode_chunk_size, remaining_steps)
        pipeline._ensure_decode_cache_capacity(state, required_update_steps=pipeline.config.decode_chunk_size)
        decode_inputs = {
            "lm_hidden": state["lm_hidden"],
            "residual_hidden": state["residual_hidden"],
            "prefix_feat_cond": state["prefix_feat_cond"],
            "base_k_cache": state["base_k_cache"],
            "base_v_cache": state["base_v_cache"],
            "base_current_length": state["base_current_length"],
            "residual_k_cache": state["residual_k_cache"],
            "residual_v_cache": state["residual_v_cache"],
            "residual_current_length": state["residual_current_length"],
            "diffusion_noise": rng.standard_normal(
                (
                    pipeline.config.decode_chunk_size,
                    1,
                    pipeline.config.feat_dim,
                    pipeline.config.patch_size,
                ),
                dtype=np.float32,
            ),
            "cfg_value": np.array([args.cfg_value], dtype=np.float32),
        }
        chunk_start = time.perf_counter()
        outputs = dict(
            zip(decode_outputs, pipeline.sessions.decode_chunk.run(decode_outputs, decode_inputs), strict=True)
        )
        chunk_seconds.append(time.perf_counter() - chunk_start)
        accepted_steps = pipeline._accept_decode_chunk_outputs(
            outputs,
            generated=generated,
            completed_steps=completed_steps,
            candidate_steps=candidate_steps,
            effective_max_steps=effective_max_steps,
            requested_max_steps=args.max_steps,
            min_steps=args.min_steps,
            progress_callback=None,
        )
        completed_steps += accepted_steps
        stop_reason = pipeline._chunk_stop_reason(
            outputs,
            accepted_steps=accepted_steps,
            completed_steps=completed_steps,
            effective_max_steps=effective_max_steps,
            requested_max_steps=args.max_steps,
            min_steps=args.min_steps,
        )
        if stop_reason is not None:
            break
        pipeline._apply_decode_chunk_cache_updates(state, outputs, update_steps=accepted_steps)
        state["lm_hidden"] = outputs["next_lm_hidden"]
        state["residual_hidden"] = outputs["next_residual_hidden"]
        state["prefix_feat_cond"] = outputs["next_prefix_feat_cond"]

    decoder_start = time.perf_counter()
    feature_seq = np.concatenate(generated, axis=1)
    decoder_latent = np.transpose(feature_seq, (0, 3, 1, 2)).reshape(1, pipeline.config.feat_dim, -1)
    sr_cond = np.array([pipeline.config.decode_sample_rate], dtype=np.int32)
    waveform = pipeline.sessions.audio_decoder.run(["waveform"], {"latent": decoder_latent, "sr_cond": sr_cond})[0][
        0, 0
    ]
    audio_decode_seconds = time.perf_counter() - decoder_start

    total_synth_seconds = time.perf_counter() - synth_start
    pipeline.write_wav(output_wav, waveform)
    return {
        "variant": "onnx",
        "case_id": case.case_id,
        "mode": case.mode,
        "ok": True,
        "output_wav": str(output_wav),
        "seed": seed,
        "model_load_seconds": loaded.load_seconds,
        "latencies": {
            "input_build_seconds": round(input_build_seconds, 6),
            "prefill_seconds": round(prefill_seconds, 6),
            "decode_step_total_seconds": round(sum(chunk_seconds), 6),
            "decode_step_seconds": [round(value, 6) for value in chunk_seconds],
            "decode_step_seconds_p50": _percentile(chunk_seconds, 50),
            "decode_step_seconds_p90": _percentile(chunk_seconds, 90),
            "decode_chunk_total_seconds": round(sum(chunk_seconds), 6),
            "decode_chunk_seconds": [round(value, 6) for value in chunk_seconds],
            "decode_chunk_seconds_p50": _percentile(chunk_seconds, 50),
            "decode_chunk_seconds_p90": _percentile(chunk_seconds, 90),
            "audio_decode_seconds": round(audio_decode_seconds, 6),
            "total_synth_seconds": round(total_synth_seconds, 6),
        },
        "decode_steps": len(generated),
        "stop_reason": stop_reason or ("safety_max_steps" if args.max_steps == 0 else "max_steps"),
        "audio": _audio_stats(waveform, pipeline.config.decode_sample_rate),
    }


def _official_capture_decode_steps(model: Any, capture: dict[str, Any]):
    original = model.tts_model._generate_with_prompt_cache

    def _wrapped(*call_args: Any, **call_kwargs: Any):
        generator = original(*call_args, **call_kwargs)
        for wav, target_text_token, pred_audio_feat in generator:
            capture["decode_steps"] = int(pred_audio_feat.shape[0])
            yield wav, target_text_token, pred_audio_feat

    return original, _wrapped


def _run_official_case(
    args: argparse.Namespace,
    loaded: LoadedVariant,
    case: BenchmarkCase,
    reference_wav: Path,
    output_wav: Path,
    seed: int,
) -> dict[str, Any]:
    import torch

    model = loaded.model
    capture: dict[str, Any] = {}
    original_generate_with_prompt_cache, wrapped_generate_with_prompt_cache = _official_capture_decode_steps(
        model, capture
    )
    sample_rate = int(getattr(model.tts_model, "sample_rate", 48_000))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    stdout = io.StringIO()
    stderr = io.StringIO()
    synth_start = time.perf_counter()
    try:
        model.tts_model._generate_with_prompt_cache = wrapped_generate_with_prompt_cache
        context = contextlib.nullcontext()
        if not args.show_official_output:
            context = contextlib.ExitStack()
            context.enter_context(contextlib.redirect_stdout(stdout))
            context.enter_context(contextlib.redirect_stderr(stderr))
        with context:
            waveform = model.generate(
                text=_mode_text(case),
                reference_wav_path=str(reference_wav) if case.needs_reference else None,
                cfg_value=args.cfg_value,
                inference_timesteps=args.official_inference_timesteps,
                min_len=args.official_min_len,
                max_len=args.official_max_len,
                normalize=args.normalize,
                denoise=False,
                retry_badcase=args.official_retry_badcase,
                retry_badcase_max_times=args.official_retry_badcase_max_times,
                retry_badcase_ratio_threshold=args.official_retry_badcase_ratio_threshold,
            )
    finally:
        model.tts_model._generate_with_prompt_cache = original_generate_with_prompt_cache

    total_synth_seconds = time.perf_counter() - synth_start
    _write_wav(output_wav, waveform, sample_rate)
    return {
        "variant": "official",
        "case_id": case.case_id,
        "mode": case.mode,
        "ok": True,
        "output_wav": str(output_wav),
        "seed": seed,
        "model_load_seconds": loaded.load_seconds,
        "latencies": {
            "input_build_seconds": None,
            "prefill_seconds": None,
            "decode_step_total_seconds": None,
            "decode_step_seconds": [],
            "decode_step_seconds_p50": None,
            "decode_step_seconds_p90": None,
            "decode_chunk_total_seconds": None,
            "decode_chunk_seconds": [],
            "decode_chunk_seconds_p50": None,
            "decode_chunk_seconds_p90": None,
            "audio_decode_seconds": None,
            "total_synth_seconds": round(total_synth_seconds, 6),
        },
        "decode_steps": capture.get("decode_steps"),
        "stop_reason": None,
        "audio": _audio_stats(waveform, sample_rate),
    }


def _run_case(
    args: argparse.Namespace,
    loaded: LoadedVariant,
    case: BenchmarkCase,
    reference_wav: Path,
    repeat_index: int,
) -> dict[str, Any]:
    prefix = f"{args.run_id}_" if args.run_id else ""
    output_wav = args.output_dir / "wavs" / f"{prefix}{loaded.variant}_{case.case_id}_r{repeat_index + 1:02d}.wav"
    seed = args.seed
    if loaded.variant == "official":
        return _run_official_case(args, loaded, case, reference_wav, output_wav, seed)
    return _run_onnx_case(args, loaded, case, reference_wav, output_wav, seed)


def _aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        if run.get("ok"):
            grouped.setdefault((run["case_id"], run["variant"]), []).append(run)

    aggregates = []
    for (case_id, variant), items in sorted(grouped.items()):
        synth_values = [item["latencies"]["total_synth_seconds"] for item in items]
        wall_values = [item["wall_seconds"] for item in items]
        duration_values = [item["audio"]["duration_seconds"] for item in items]
        decode_steps = [float(item["decode_steps"]) for item in items if item.get("decode_steps") is not None]
        input_build_values = [
            item["latencies"]["input_build_seconds"]
            for item in items
            if item["latencies"]["input_build_seconds"] is not None
        ]
        prefill_values = [
            item["latencies"]["prefill_seconds"] for item in items if item["latencies"]["prefill_seconds"] is not None
        ]
        decode_step_totals = [
            item["latencies"]["decode_step_total_seconds"]
            for item in items
            if item["latencies"]["decode_step_total_seconds"] is not None
        ]
        decode_step_values = [value for item in items for value in item["latencies"].get("decode_step_seconds", [])]
        decode_chunk_totals = [
            item["latencies"]["decode_chunk_total_seconds"]
            for item in items
            if item["latencies"].get("decode_chunk_total_seconds") is not None
        ]
        decode_chunk_values = [value for item in items for value in item["latencies"].get("decode_chunk_seconds", [])]
        audio_decode_values = [
            item["latencies"]["audio_decode_seconds"]
            for item in items
            if item["latencies"]["audio_decode_seconds"] is not None
        ]
        aggregates.append(
            {
                "case_id": case_id,
                "variant": variant,
                "repeats": len(items),
                "model_load_seconds": items[0]["model_load_seconds"],
                "wall_seconds": _stats(wall_values),
                "total_synth_seconds": _stats(synth_values),
                "input_build_seconds": _stats(input_build_values),
                "prefill_seconds": _stats(prefill_values),
                "decode_step_total_seconds": _stats(decode_step_totals),
                "decode_step_seconds": _stats(decode_step_values),
                "decode_chunk_total_seconds": _stats(decode_chunk_totals),
                "decode_chunk_seconds": _stats(decode_chunk_values),
                "audio_decode_seconds": _stats(audio_decode_values),
                "output_duration_seconds": _stats(duration_values),
                "decode_steps": _stats(decode_steps),
            }
        )
    return aggregates


def _comparison_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case_variant = {(item["case_id"], item["variant"]): item for item in aggregates}
    rows = []
    for case_id in CASE_IDS:
        official = by_case_variant.get((case_id, "official"))
        onnx = by_case_variant.get((case_id, "onnx"))
        if not official or not onnx:
            continue
        official_p50 = official["total_synth_seconds"]["p50"]
        onnx_p50 = onnx["total_synth_seconds"]["p50"]
        ratio = round(onnx_p50 / official_p50, 3) if official_p50 and onnx_p50 else None
        rows.append(
            {
                "case_id": case_id,
                "official_synth_p50": official_p50,
                "onnx_synth_p50": onnx_p50,
                "onnx_vs_official_ratio": ratio,
                "onnx_input_build_p50": onnx["input_build_seconds"]["p50"],
                "onnx_prefill_p50": onnx["prefill_seconds"]["p50"],
                "onnx_decode_step_total_p50": onnx["decode_step_total_seconds"]["p50"],
                "onnx_decode_step_p50": onnx["decode_step_seconds"]["p50"],
                "onnx_decode_chunk_total_p50": onnx["decode_chunk_total_seconds"]["p50"],
                "onnx_decode_chunk_p50": onnx["decode_chunk_seconds"]["p50"],
                "onnx_audio_decode_p50": onnx["audio_decode_seconds"]["p50"],
            }
        )
    return rows


def _make_report(
    args: argparse.Namespace, runs: list[dict[str, Any]], aggregates: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_unix_seconds": round(time.time(), 3),
        "repo_root": str(REPO_ROOT),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "config": {
            "variants": args.variants,
            "cases": args.cases,
            "repeats": args.repeats,
            "seed": args.seed,
            "model_path": args.model_path,
            "max_steps": args.max_steps,
            "min_steps": args.min_steps,
            "cfg_value": args.cfg_value,
            "onnx": {
                "graph_optimization": args.onnx_graph_optimization,
                "execution_mode": args.onnx_execution_mode,
                "log_severity": args.onnx_log_severity,
                "preload_sessions": args.onnx_preload_sessions,
                "intra_op_threads": args.onnx_intra_op_threads,
                "inter_op_threads": args.onnx_inter_op_threads,
                "enable_mem_pattern": args.onnx_enable_mem_pattern,
                "enable_cpu_mem_arena": args.onnx_enable_cpu_mem_arena,
                "enable_mem_reuse": args.onnx_enable_mem_reuse,
                "max_audio_encoder_samples": args.max_audio_encoder_samples,
                "max_decoder_latent_steps": args.max_decoder_latent_steps,
                "max_prefill_seq_len": args.max_prefill_seq_len,
                "max_decode_cache_seq": args.max_decode_cache_seq,
            },
            "official": {
                "device": args.official_device,
                "inference_timesteps": args.official_inference_timesteps,
                "min_len": args.official_min_len,
                "max_len": args.official_max_len,
                "retry_badcase": args.official_retry_badcase,
            },
        },
        "runs": runs,
        "aggregates": aggregates,
        "comparisons": _comparison_rows(aggregates),
    }


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    intra_threads = report["config"]["onnx"]["intra_op_threads"]
    inter_threads = report["config"]["onnx"]["inter_op_threads"]
    lines = [
        "# VoxCPM2 Production Performance Baseline",
        "",
        f"- JSON report: `{Path(report['json_report']).as_posix()}`",
        f"- repeats: `{report['config']['repeats']}`",
        f"- variants: `{', '.join(report['config']['variants'])}`",
        f"- ONNX graph optimization: `{report['config']['onnx']['graph_optimization']}`",
        f"- ONNX threads: intra=`{intra_threads if intra_threads is not None else 'default'}`, "
        f"inter=`{inter_threads if inter_threads is not None else 'default'}`",
        f"- ONNX memory: mem_pattern=`{report['config']['onnx']['enable_mem_pattern']}`, "
        f"cpu_arena=`{report['config']['onnx']['enable_cpu_mem_arena']}`, "
        f"mem_reuse=`{report['config']['onnx']['enable_mem_reuse']}`",
        "",
        "## Summary",
        "",
        "| case | variant | load s | wall p50 s | wall p90 s | synth p50 s | synth p90 s | duration p50 s | decode steps p50 | prefill p50 s | decode-chunk p50 s | decode-total p50 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["aggregates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["case_id"],
                    item["variant"],
                    _format_cell(item["model_load_seconds"]),
                    _format_cell(item["wall_seconds"]["p50"]),
                    _format_cell(item["wall_seconds"]["p90"]),
                    _format_cell(item["total_synth_seconds"]["p50"]),
                    _format_cell(item["total_synth_seconds"]["p90"]),
                    _format_cell(item["output_duration_seconds"]["p50"]),
                    _format_cell(item["decode_steps"]["p50"]),
                    _format_cell(item["prefill_seconds"]["p50"]),
                    _format_cell(item["decode_chunk_seconds"]["p50"]),
                    _format_cell(item["decode_chunk_total_seconds"]["p50"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## ONNX vs Official",
            "",
            "| case | official synth p50 s | ONNX synth p50 s | ONNX / official | ONNX input p50 s | ONNX prefill p50 s | ONNX decode total p50 s | ONNX audio decode p50 s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["comparisons"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"],
                    _format_cell(row["official_synth_p50"]),
                    _format_cell(row["onnx_synth_p50"]),
                    _format_cell(row["onnx_vs_official_ratio"]),
                    _format_cell(row["onnx_input_build_p50"]),
                    _format_cell(row["onnx_prefill_p50"]),
                    _format_cell(row["onnx_decode_chunk_total_p50"]),
                    _format_cell(row["onnx_audio_decode_p50"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Official API exposes reliable load and total synthesis timing. Per-stage official prefill/decode timing is intentionally reported as `-` because this tool does not rewrite official model internals.",
            "- ONNX stage timings are measured at current explicit runtime boundaries: host input build, `VoxCPM2Prefill`, repeated `VoxCPM2DecodeChunk`, and `AudioVAEDecoder`.",
            "- No model math, export path, or runtime stop/cache semantics are changed by this benchmark.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_progress_header(args: argparse.Namespace, json_report: Path, markdown_report: Path) -> None:
    print("=" * 72, flush=True)
    print("VoxCPM2 production baseline", flush=True)
    print("=" * 72, flush=True)
    print(f"variants       : {', '.join(args.variants)}", flush=True)
    if args.run_id:
        print(f"run_id         : {args.run_id}", flush=True)
    print(f"cases          : {', '.join(args.cases)}", flush=True)
    print(f"repeats        : {args.repeats}", flush=True)
    print(f"output_dir     : {args.output_dir}", flush=True)
    print(f"json_report    : {json_report}", flush=True)
    print(f"markdown_report: {markdown_report}", flush=True)
    print(
        "ONNX ORT       : "
        f"graph_opt={args.onnx_graph_optimization}, execution={args.onnx_execution_mode}, "
        f"intra={args.onnx_intra_op_threads or 'default'}, inter={args.onnx_inter_op_threads or 'default'}, "
        f"mem_pattern={args.onnx_enable_mem_pattern}, cpu_arena={args.onnx_enable_cpu_mem_arena}, "
        f"mem_reuse={args.onnx_enable_mem_reuse}",
        flush=True,
    )
    print(flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_run_id(args.run_id)
    args.output_dir = args.output_dir.expanduser()
    json_name = f"baseline_{args.run_id}.json" if args.run_id else "baseline.json"
    markdown_name = f"baseline_{args.run_id}.md" if args.run_id else "baseline.md"
    json_report = (args.json_report or (args.output_dir / json_name)).expanduser()
    markdown_report = (args.markdown_report or (args.output_dir / markdown_name)).expanduser()
    reference_name = f"reference_16k_{args.run_id}.wav" if args.run_id else "reference_16k.wav"
    reference_wav = _make_reference_wav(
        args.reference_wav.expanduser() if args.reference_wav else args.output_dir / reference_name
    )
    cases = _selected_cases(args, reference_wav)
    _print_progress_header(args, json_report, markdown_report)

    loaded_variants = _load_variants(args)
    runs: list[dict[str, Any]] = []
    for case in cases:
        for variant in args.variants:
            loaded = loaded_variants[variant]
            for repeat_index in range(args.repeats):
                label = f"{variant}/{case.case_id}/r{repeat_index + 1}"
                print(f"running {label}...", flush=True)
                start = time.perf_counter()
                try:
                    run_record = _run_case(args, loaded, case, reference_wav, repeat_index)
                except Exception as exc:  # noqa: BLE001 - baseline should preserve failures in the report.
                    run_record = {
                        "variant": variant,
                        "case_id": case.case_id,
                        "mode": case.mode,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                run_record["repeat_index"] = repeat_index
                run_record["wall_seconds"] = round(time.perf_counter() - start, 6)
                runs.append(run_record)
                status = "OK" if run_record.get("ok") else "FAIL"
                synth = run_record.get("latencies", {}).get("total_synth_seconds")
                steps = run_record.get("decode_steps")
                print(f"  [{status}] synth={_format_cell(synth)}s steps={_format_cell(steps)}", flush=True)

    aggregates = _aggregate_runs(runs)
    report = _make_report(args, runs, aggregates)
    report["json_report"] = str(json_report)
    report["markdown_report"] = str(markdown_report)
    _write_json(json_report, report)
    _write_markdown(markdown_report, report)
    print(flush=True)
    print(f"json saved    : {json_report}", flush=True)
    print(f"markdown saved: {markdown_report}", flush=True)
    return report


def _validate_run_id(run_id: str | None) -> None:
    if run_id is None:
        return
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("--run-id may contain only letters, digits, dot, underscore, and dash")


def _parser() -> argparse.ArgumentParser:
    _, _, _, _, graph_choices, execution_choices, log_choices = _runtime_imports()
    parser = argparse.ArgumentParser(
        description="Run the fixed production performance baseline for official VoxCPM2 API and ONNX CPU runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/perf_baseline"), help="Output directory.")
    parser.add_argument("--run-id", help="Optional file-name prefix for concurrent baseline runs.")
    parser.add_argument(
        "--json-report",
        type=Path,
        help="JSON report path. Defaults to <output-dir>/baseline.json or baseline_<run-id>.json.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        help="Markdown report path. Defaults to <output-dir>/baseline.md or baseline_<run-id>.md.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["official", "onnx"],
        default=["official", "onnx"],
        help="Variants to benchmark.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASE_IDS,
        default=list(CASE_IDS),
        help="Fixed test cases to run. Defaults to the full production matrix.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeat count per case and variant.")
    parser.add_argument("--seed", type=int, default=0, help="Fixed RNG seed reused for every repeat.")
    parser.add_argument("--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or HF id.")
    parser.add_argument("--reference-wav", type=Path, help="Optional reference WAV for controllable_clone_short.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value.")
    parser.add_argument("--max-steps", type=int, default=0, help="ONNX max decode steps. 0 means until stop logits.")
    parser.add_argument("--min-steps", type=int, default=8, help="ONNX min decode steps before stop logits.")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Require local model files.")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only", help="Allow HF downloads.")
    parser.add_argument("--normalize", action="store_true", help="Enable official API text normalization.")

    parser.add_argument("--official-device", default="cpu", help="Device passed to official VoxCPM.from_pretrained.")
    parser.add_argument("--official-inference-timesteps", type=int, default=10, help="Official CFM solver steps.")
    parser.add_argument("--official-min-len", type=int, default=2, help="Official API min generated feature length.")
    parser.add_argument("--official-max-len", type=int, default=4096, help="Official API max generated feature length.")
    parser.add_argument("--official-retry-badcase", action="store_true", help="Enable official retry_badcase.")
    parser.add_argument("--official-retry-badcase-max-times", type=int, default=3, help="Official retry attempts.")
    parser.add_argument("--official-retry-badcase-ratio-threshold", type=float, default=6.0, help="Retry threshold.")
    parser.add_argument("--show-official-output", action="store_true", help="Show official API tqdm/log output.")

    parser.add_argument("--onnx-graph-optimization", choices=graph_choices, default="all", help="ORT graph opt.")
    parser.add_argument("--onnx-execution-mode", choices=execution_choices, default="sequential", help="ORT mode.")
    parser.add_argument("--onnx-log-severity", choices=log_choices, default="error", help="ORT log severity.")
    parser.add_argument(
        "--onnx-preload-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create all ONNX sessions during model load.",
    )
    parser.add_argument("--onnx-intra-op-threads", type=int, default=8, help="ORT intra-op threads. Use 0 for default.")
    parser.add_argument("--onnx-inter-op-threads", type=int, default=1, help="ORT inter-op threads. Use 0 for default.")
    parser.add_argument(
        "--onnx-enable-mem-pattern",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ORT memory pattern planning.",
    )
    parser.add_argument(
        "--onnx-enable-cpu-mem-arena",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ORT CPU memory arena.",
    )
    parser.add_argument(
        "--onnx-enable-mem-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ORT memory reuse.",
    )
    parser.add_argument(
        "--max-audio-encoder-samples", type=int, help="Runtime/export bound for reference audio samples."
    )
    parser.add_argument("--max-decoder-latent-steps", type=int, help="Runtime/export bound for decoder latent steps.")
    parser.add_argument("--max-prefill-seq-len", type=int, help="Runtime/export bound for prefill sequence length.")
    parser.add_argument("--max-decode-cache-seq", type=int, help="Runtime/export bound for decode cache capacity.")
    parser.add_argument("--audio-encoder-onnx", type=Path, help="Override AudioVAEEncoder ONNX path.")
    parser.add_argument("--audio-decoder-onnx", type=Path, help="Override AudioVAEDecoder ONNX path.")
    parser.add_argument("--prefill-onnx", type=Path, help="Override VoxCPM2Prefill ONNX path.")
    parser.add_argument("--decode-chunk-onnx", type=Path, help="Override production VoxCPM2DecodeChunk ONNX path.")
    parser.add_argument("--list-cases", action="store_true", help="Print the fixed case matrix and exit.")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("error: --repeats must be >= 1")
    if args.list_cases:
        reference_wav = args.reference_wav or Path("<generated-reference-wav>")
        for case in _cases(reference_wav):
            print(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "mode": case.mode,
                        "text": case.text,
                        "voice_design": case.voice_design,
                        "needs_reference": case.needs_reference,
                    },
                    sort_keys=True,
                )
            )
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
