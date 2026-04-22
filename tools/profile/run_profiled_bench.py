#!/usr/bin/env python3
"""Run a small ONNX synthesis benchmark with ORT profiling enabled"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_IDS = ("text_only_short", "voice_design_short", "controllable_clone_short")


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _runtime_imports():
    _ensure_repo_root_on_path()

    from src.runtime.pipeline import VoxCPM2OnnxPipeline
    from src.runtime.session_factory import (
        EXECUTION_MODE_CHOICES,
        GRAPH_OPTIMIZATION_CHOICES,
        LOG_SEVERITY_CHOICES,
        OnnxModelPaths,
    )

    return VoxCPM2OnnxPipeline, OnnxModelPaths, GRAPH_OPTIMIZATION_CHOICES, EXECUTION_MODE_CHOICES, LOG_SEVERITY_CHOICES


def _bench_helpers():
    _ensure_repo_root_on_path()

    from tools.bench.run_benchmarks import BenchmarkCase, _make_reference_wav

    return BenchmarkCase, _make_reference_wav


def _profile_parser():
    _ensure_repo_root_on_path()

    from tools.profile.parse_ort_profile import parse_profiles, _write_json, _write_markdown

    return parse_profiles, _write_json, _write_markdown


def _cases() -> dict[str, object]:
    BenchmarkCase, _ = _bench_helpers()
    return {
        "text_only_short": BenchmarkCase(
            case_id="text_only_short",
            mode="text_only",
            text="Hello from VoxCPM2.",
        ),
        "voice_design_short": BenchmarkCase(
            case_id="voice_design_short",
            mode="voice_design",
            text="Hello from VoxCPM2.",
            voice_design="pretty girl with sugar voice, slow",
        ),
        "controllable_clone_short": BenchmarkCase(
            case_id="controllable_clone_short",
            mode="controllable_clone",
            text="Hello from VoxCPM2.",
            needs_reference=True,
        ),
    }


def _onnx_paths(args: argparse.Namespace):
    _, OnnxModelPaths, *_ = _runtime_imports()
    defaults = OnnxModelPaths()
    return OnnxModelPaths(
        audio_encoder=args.audio_encoder_onnx or defaults.audio_encoder,
        audio_decoder=args.audio_decoder_onnx or defaults.audio_decoder,
        prefill=args.prefill_onnx or defaults.prefill,
        decode_chunk=args.decode_chunk_onnx or defaults.decode_chunk,
    )


def _create_pipeline(args: argparse.Namespace, profile_dir: Path):
    VoxCPM2OnnxPipeline, *_ = _runtime_imports()
    return VoxCPM2OnnxPipeline.from_default_artifacts(
        model_path=args.model_path,
        local_files_only=args.local_files_only,
        onnx_paths=_onnx_paths(args),
        graph_optimization_level=args.onnx_graph_optimization,
        execution_mode=args.onnx_execution_mode,
        log_severity_level=args.onnx_log_severity,
        intra_op_num_threads=args.onnx_intra_op_threads,
        inter_op_num_threads=args.onnx_inter_op_threads,
        enable_profiling=True,
        profile_file_prefix=profile_dir,
        max_audio_encoder_samples=args.max_audio_encoder_samples,
        max_decoder_latent_steps=args.max_decoder_latent_steps,
        max_prefill_seq_len=args.max_prefill_seq_len,
        max_decode_cache_seq=args.max_decode_cache_seq,
    )


def _preload_all_sessions(pipeline: object) -> None:
    _ = pipeline.sessions.audio_encoder
    _ = pipeline.sessions.prefill
    _ = pipeline.sessions.decode_chunk
    _ = pipeline.sessions.audio_decoder


def _run_case(args: argparse.Namespace, pipeline: object, case: object, reference_wav: Path) -> dict[str, object]:
    output_wav = args.output_dir / "wavs" / f"onnx_profile_{case.case_id}.wav"
    start = time.perf_counter()
    result = pipeline.synthesize_with_metadata(
        case.text,
        mode=case.mode,
        voice_design=case.voice_design,
        reference_wav_path=reference_wav if case.needs_reference else None,
        prompt_wav_path=None,
        prompt_text=None,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        cfg_value=args.cfg_value,
        seed=args.seed,
    )
    synth_seconds = time.perf_counter() - start
    pipeline.write_wav(output_wav, result.waveform)
    return {
        "case_id": case.case_id,
        "mode": case.mode,
        "output_wav": str(output_wav),
        "synth_seconds": round(synth_seconds, 6),
        "decode_steps": result.metadata.decode_steps,
        "stop_reason": result.metadata.stop_reason,
        "samples": int(result.waveform.shape[0]),
        "duration_seconds": round(int(result.waveform.shape[0]) / pipeline.config.decode_sample_rate, 6),
        "peak": round(float(np.max(np.abs(result.waveform))) if result.waveform.size else 0.0, 8),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir = args.output_dir.expanduser()
    profile_dir = args.profile_dir.expanduser() if args.profile_dir else args.output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, make_reference_wav = _bench_helpers()
    reference_wav = make_reference_wav(
        args.reference_wav.expanduser() if args.reference_wav else args.output_dir / "reference_16k.wav"
    )
    selected_cases = [_cases()[case_id] for case_id in args.cases]

    print("=" * 72, flush=True)
    print("VoxCPM2 ORT profiled bench", flush=True)
    print("=" * 72, flush=True)
    print(f"cases      : {', '.join(args.cases)}", flush=True)
    print(f"output_dir : {args.output_dir}", flush=True)
    print(f"profile_dir: {profile_dir}", flush=True)
    print(
        "ONNX ORT   : "
        f"graph_opt={args.onnx_graph_optimization}, execution={args.onnx_execution_mode}, "
        f"intra={args.onnx_intra_op_threads or 'default'}, inter={args.onnx_inter_op_threads or 'default'}",
        flush=True,
    )
    print(flush=True)

    load_start = time.perf_counter()
    pipeline = _create_pipeline(args, profile_dir)
    pipeline.validate()
    if args.preload_sessions:
        _preload_all_sessions(pipeline)
    load_seconds = time.perf_counter() - load_start
    print(f"load: {load_seconds:.3f}s", flush=True)

    runs = []
    for case in selected_cases:
        print(f"running {case.case_id}...", flush=True)
        run_record = _run_case(args, pipeline, case, reference_wav)
        runs.append(run_record)
        print(
            f"  synth={run_record['synth_seconds']:.3f}s "
            f"steps={run_record['decode_steps']} wav={run_record['output_wav']}",
            flush=True,
        )

    profile_paths = pipeline.sessions.end_profiling()
    print("profiles:", flush=True)
    for name, path in sorted(profile_paths.items()):
        print(f"  {name}: {path}", flush=True)

    parse_profiles, write_json, write_markdown = _profile_parser()
    hotspot_report = parse_profiles(list(profile_paths.values()), top_n=args.top_n)
    result = {
        "schema_version": 1,
        "load_seconds": round(load_seconds, 6),
        "runs": runs,
        "profile_paths": {name: str(path) for name, path in profile_paths.items()},
        "hotspots": hotspot_report,
    }
    json_report = (args.json_report or (args.output_dir / "profiled_bench.json")).expanduser()
    markdown_report = (args.markdown_report or (args.output_dir / "hotspots.md")).expanduser()
    write_json(json_report, result)
    write_markdown(markdown_report, hotspot_report, top_n=args.top_n)
    print(flush=True)
    print(f"json saved    : {json_report}", flush=True)
    print(f"markdown saved: {markdown_report}", flush=True)
    return result


def _parser() -> argparse.ArgumentParser:
    _, _, graph_choices, execution_choices, log_choices = _runtime_imports()
    parser = argparse.ArgumentParser(
        description="Run ONNX VoxCPM2 synthesis with ORT profiling and emit hotspot reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/profile"), help="Output directory.")
    parser.add_argument("--profile-dir", type=Path, help="ORT profile directory. Defaults to <output-dir>/profiles.")
    parser.add_argument(
        "--json-report", type=Path, help="JSON report path. Defaults to <output-dir>/profiled_bench.json."
    )
    parser.add_argument(
        "--markdown-report", type=Path, help="Markdown report path. Defaults to <output-dir>/hotspots.md."
    )
    parser.add_argument(
        "--cases", nargs="+", choices=CASE_IDS, default=["controllable_clone_short"], help="Cases to run."
    )
    parser.add_argument("--top-n", type=int, default=20, help="Ranked hotspot count per section.")
    parser.add_argument("--seed", type=int, default=0, help="Host diffusion-noise RNG seed.")
    parser.add_argument("--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or HF id.")
    parser.add_argument("--reference-wav", type=Path, help="Optional reference WAV for controllable clone profiling.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value.")
    parser.add_argument("--max-steps", type=int, default=0, help="ONNX max decode steps. 0 means until stop logits.")
    parser.add_argument("--min-steps", type=int, default=8, help="ONNX min decode steps before stop logits.")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Require local model files.")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only", help="Allow HF downloads.")
    parser.add_argument(
        "--preload-sessions", action=argparse.BooleanOptionalAction, default=True, help="Preload sessions."
    )
    parser.add_argument("--onnx-graph-optimization", choices=graph_choices, default="all", help="ORT graph opt.")
    parser.add_argument("--onnx-execution-mode", choices=execution_choices, default="sequential", help="ORT mode.")
    parser.add_argument("--onnx-log-severity", choices=log_choices, default="error", help="ORT log severity.")
    parser.add_argument("--onnx-intra-op-threads", type=int, help="ORT intra-op threads. Omit for default.")
    parser.add_argument("--onnx-inter-op-threads", type=int, help="ORT inter-op threads. Omit for default.")
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
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
