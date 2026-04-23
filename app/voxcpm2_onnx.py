"""Small embeddable API around the production VoxCPM2 ONNX CPU runtime.

This module is intentionally thin. It demonstrates the public integration
surface an application can copy: choose FP32/BF16 artifacts, create one
CPU-only pipeline, call `synthesize()`, and optionally write a WAV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.runtime.pipeline import Mode, SynthesisResult, VoxCPM2OnnxPipeline
from src.runtime.session_factory import OnnxModelPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX_ROOT = REPO_ROOT / "models" / "onnx"


def paths_for_precision(precision: str, *, root: str | Path = ONNX_ROOT) -> OnnxModelPaths:
    """Return the standard model paths for one production precision family"""

    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be 'fp32' or 'bf16'")
    base = Path(root).expanduser() / precision
    return OnnxModelPaths(
        audio_encoder=base / "audio_vae_encoder" / "audio_vae_encoder.onnx",
        audio_decoder=base / "audio_vae_decoder" / "audio_vae_decoder.onnx",
        prefill=base / "prefill" / "voxcpm2_prefill.onnx",
        decode_chunk=base / "decode_chunk" / "voxcpm2_decode_chunk.onnx",
    )


@dataclass(frozen=True)
class VoxCPM2OnnxConfig:
    """Application-facing runtime config with production defaults"""

    precision: str = "bf16"
    model_path: str | Path = "openbmb/VoxCPM2"
    local_files_only: bool = True
    onnx_root: str | Path = ONNX_ROOT
    graph_optimization_level: str = "all"
    execution_mode: str = "sequential"
    log_severity_level: str = "error"
    intra_op_num_threads: int | None = 8
    inter_op_num_threads: int | None = 1
    enable_mem_pattern: bool | None = True
    enable_cpu_mem_arena: bool | None = True
    enable_mem_reuse: bool | None = True
    prefer_optimized_onnx: bool = False
    enable_decode_chunk_iobinding: bool = False
    max_audio_encoder_samples: int | None = None
    max_decoder_latent_steps: int | None = None
    max_prefill_seq_len: int | None = None
    max_decode_cache_seq: int | None = None


class VoxCPM2Onnx:
    """Minimal reusable API for CPU-only VoxCPM2 ONNX synthesis"""

    def __init__(self, config: VoxCPM2OnnxConfig | None = None) -> None:
        self.config = config or VoxCPM2OnnxConfig()
        self.pipeline = VoxCPM2OnnxPipeline.from_default_artifacts(
            model_path=self.config.model_path,
            local_files_only=self.config.local_files_only,
            onnx_paths=paths_for_precision(self.config.precision, root=self.config.onnx_root),
            graph_optimization_level=self.config.graph_optimization_level,
            execution_mode=self.config.execution_mode,
            log_severity_level=self.config.log_severity_level,
            intra_op_num_threads=self.config.intra_op_num_threads,
            inter_op_num_threads=self.config.inter_op_num_threads,
            enable_mem_pattern=self.config.enable_mem_pattern,
            enable_cpu_mem_arena=self.config.enable_cpu_mem_arena,
            enable_mem_reuse=self.config.enable_mem_reuse,
            prefer_optimized_onnx=self.config.prefer_optimized_onnx,
            enable_decode_chunk_iobinding=self.config.enable_decode_chunk_iobinding,
            max_audio_encoder_samples=self.config.max_audio_encoder_samples,
            max_decoder_latent_steps=self.config.max_decoder_latent_steps,
            max_prefill_seq_len=self.config.max_prefill_seq_len,
            max_decode_cache_seq=self.config.max_decode_cache_seq,
        )

    def validate(self) -> dict[str, Path]:
        """Verify ONNX artifacts and official tokenizer/config files exist"""

        return self.pipeline.validate()

    def synthesize(
        self,
        text: str,
        *,
        mode: Mode = "text_only",
        voice_design: str | None = None,
        reference_wav_path: str | Path | None = None,
        prompt_wav_path: str | Path | None = None,
        prompt_text: str | None = None,
        output_wav: str | Path | None = None,
        max_steps: int = 0,
        min_steps: int = 8,
        cfg_value: float = 2.0,
        seed: int = 0,
    ) -> SynthesisResult:
        """Run one synthesis request and optionally write the waveform to disk"""

        result = self.pipeline.synthesize_with_metadata(
            text,
            mode=mode,
            voice_design=voice_design,
            reference_wav_path=reference_wav_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            max_steps=max_steps,
            min_steps=min_steps,
            cfg_value=cfg_value,
            seed=seed,
        )
        if output_wav is not None:
            self.pipeline.write_wav(output_wav, result.waveform)
        return result

    def write_wav(self, path: str | Path, waveform: np.ndarray) -> None:
        """Write a waveform with the runtime decode sample rate"""

        self.pipeline.write_wav(path, waveform)
