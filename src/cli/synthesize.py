#!/usr/bin/env python3
"""Synthesize speech with the CPU-only VoxCPM2 ONNX Runtime pipeline"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _runtime_classes():
    _ensure_repo_root_on_path()

    from src.runtime.pipeline import VoxCPM2OnnxPipeline
    from src.runtime.session_factory import (
        EXECUTION_MODE_CHOICES,
        GRAPH_OPTIMIZATION_CHOICES,
        LOG_SEVERITY_CHOICES,
        OnnxModelPaths,
    )

    return VoxCPM2OnnxPipeline, OnnxModelPaths, GRAPH_OPTIMIZATION_CHOICES, EXECUTION_MODE_CHOICES, LOG_SEVERITY_CHOICES


def _parser() -> argparse.ArgumentParser:
    _, OnnxModelPaths, graph_optimization_choices, execution_mode_choices, log_severity_choices = _runtime_classes()
    defaults = OnnxModelPaths()
    parser = argparse.ArgumentParser(
        description="Run the VoxCPM2 CPU-only ONNX Runtime synthesis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/cli/synthesize.py --text 'Hello from VoxCPM2.' "
            "--output artifacts/samples/runtime_sample.wav --mode text_only"
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
    parser.add_argument(
        "--reference-wav", type=Path, help="Reference WAV path for controllable_clone or ultimate_clone."
    )
    parser.add_argument("--prompt-wav", type=Path, help="Prompt/continuation WAV path for ultimate_clone.")
    parser.add_argument("--prompt-text", help="Prompt/continuation text paired with --prompt-wav for ultimate_clone.")
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument(
        "--audio-encoder-onnx", type=Path, default=defaults.audio_encoder, help="AudioVAEEncoder ONNX file."
    )
    parser.add_argument(
        "--audio-decoder-onnx", type=Path, default=defaults.audio_decoder, help="AudioVAEDecoder ONNX file."
    )
    parser.add_argument("--prefill-onnx", type=Path, default=defaults.prefill, help="VoxCPM2Prefill ONNX file.")
    parser.add_argument(
        "--decode-chunk-onnx",
        type=Path,
        default=defaults.decode_chunk,
        help="Production VoxCPM2DecodeChunk ONNX file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum host decode-loop iterations. 0 means run until stop logits with an internal safety cap.",
    )
    parser.add_argument(
        "--min-steps", type=int, default=8, help="Minimum decode steps before stop logits may end synthesis."
    )
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value passed to decode.")
    parser.add_argument("--seed", type=int, default=0, help="NumPy RNG seed for host-supplied diffusion noise.")
    parser.add_argument(
        "--max-audio-encoder-samples",
        type=int,
        help="Runtime bound matching the AudioVAEEncoder export shape profile.",
    )
    parser.add_argument(
        "--max-decoder-latent-steps",
        type=int,
        help="Runtime bound matching the AudioVAEDecoder export shape profile.",
    )
    parser.add_argument(
        "--max-prefill-seq-len",
        type=int,
        help="Runtime bound matching the Prefill export shape profile.",
    )
    parser.add_argument(
        "--max-decode-cache-seq",
        type=int,
        help="Runtime bound matching the DecodeChunk export shape profile.",
    )
    parser.add_argument(
        "--ort-graph-optimization",
        choices=graph_optimization_choices,
        default="all",
        help="ONNX Runtime graph optimization level for all CPU sessions.",
    )
    parser.add_argument(
        "--ort-execution-mode",
        choices=execution_mode_choices,
        default="sequential",
        help="ONNX Runtime execution mode for all CPU sessions.",
    )
    parser.add_argument(
        "--ort-log-severity",
        choices=log_severity_choices,
        default="error",
        help="Minimum ONNX Runtime log severity emitted by CPU sessions.",
    )
    parser.add_argument(
        "--ort-intra-op-threads",
        type=int,
        default=8,
        help="ONNX Runtime intra-op thread count. Use 0 to request ORT default scheduling.",
    )
    parser.add_argument(
        "--ort-inter-op-threads",
        type=int,
        default=1,
        help="ONNX Runtime inter-op thread count. Use 0 to request ORT default scheduling.",
    )
    parser.add_argument(
        "--ort-enable-mem-pattern",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ORT memory pattern planning for CPU sessions.",
    )
    parser.add_argument(
        "--ort-enable-cpu-mem-arena",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ORT CPU memory arena for repeated allocations.",
    )
    parser.add_argument(
        "--ort-enable-mem-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ORT memory reuse inside CPU sessions.",
    )
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Require local model files.")
    parser.add_argument(
        "--allow-download", action="store_false", dest="local_files_only", help="Allow Hugging Face downloads."
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    VoxCPM2OnnxPipeline, OnnxModelPaths, _, _, _ = _runtime_classes()
    paths = OnnxModelPaths(
        audio_encoder=args.audio_encoder_onnx,
        audio_decoder=args.audio_decoder_onnx,
        prefill=args.prefill_onnx,
        decode_chunk=args.decode_chunk_onnx,
    )
    pipeline = VoxCPM2OnnxPipeline.from_default_artifacts(
        model_path=args.model_path,
        local_files_only=args.local_files_only,
        onnx_paths=paths,
        graph_optimization_level=args.ort_graph_optimization,
        execution_mode=args.ort_execution_mode,
        log_severity_level=args.ort_log_severity,
        intra_op_num_threads=args.ort_intra_op_threads,
        inter_op_num_threads=args.ort_inter_op_threads,
        enable_mem_pattern=args.ort_enable_mem_pattern,
        enable_cpu_mem_arena=args.ort_enable_cpu_mem_arena,
        enable_mem_reuse=args.ort_enable_mem_reuse,
        max_audio_encoder_samples=args.max_audio_encoder_samples,
        max_decoder_latent_steps=args.max_decoder_latent_steps,
        max_prefill_seq_len=args.max_prefill_seq_len,
        max_decode_cache_seq=args.max_decode_cache_seq,
    )
    pipeline.validate()
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
    )
    waveform = result.waveform
    pipeline.write_wav(args.output, waveform)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_rate": pipeline.config.decode_sample_rate,
                "samples": int(waveform.shape[0]),
                "decode_steps": result.metadata.decode_steps,
                "stop_reason": result.metadata.stop_reason,
                "sessions": list(pipeline.sessions.created_session_names),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
