"""CPU-only VoxCPM2 ONNX Runtime synthesis pipeline"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from huggingface_hub import snapshot_download
from scipy.io import wavfile
from scipy.signal import resample_poly
from tokenizers import Tokenizer

from src.runtime.session_factory import OnnxModelPaths, OrtSessionFactory


Mode = Literal["text_only", "voice_design", "controllable_clone", "ultimate_clone"]
StopReason = Literal["stop_logits", "max_steps", "safety_max_steps"]

PREFILL_INPUTS = ["text_tokens", "text_mask", "audio_features", "audio_mask"]
PREFILL_OUTPUTS = [
    "lm_hidden",
    "residual_hidden",
    "prefix_feat_cond",
    "base_k_cache",
    "base_v_cache",
    "base_cache_length",
    "residual_k_cache",
    "residual_v_cache",
    "residual_cache_length",
]
DECODE_INPUTS = [
    "lm_hidden",
    "residual_hidden",
    "prefix_feat_cond",
    "base_k_cache",
    "base_v_cache",
    "base_current_length",
    "residual_k_cache",
    "residual_v_cache",
    "residual_current_length",
    "diffusion_noise",
    "cfg_value",
]
DECODE_OUTPUTS = [
    "pred_audio_feature",
    "decoder_latent",
    "stop_logits",
    "next_lm_hidden",
    "next_residual_hidden",
    "next_prefix_feat_cond",
    "base_k_update",
    "base_v_update",
    "next_base_current_length",
    "residual_k_update",
    "residual_v_update",
    "next_residual_current_length",
]


class CharTokenizerWrapper:
    """Minimal tokenizer adapter that avoids importing Transformers in runtime.

    The official path splits multi-character Chinese tokens into individual
    characters before inference. Keeping that behavior here preserves the
    multilingual path while avoiding the heavier Transformers import, which can
    transitively load PyTorch.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.multichar_tokens = {
            token
            for token in tokenizer.get_vocab().keys()
            if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
        }

    def __call__(self, text: str) -> list[int]:
        ids = []
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        for token, token_id in zip(encoded.tokens, encoded.ids, strict=True):
            clean_token = token.replace("▁", "")
            if clean_token in self.multichar_tokens:
                for char in clean_token:
                    char_id = self.tokenizer.token_to_id(char)
                    if char_id is None:
                        raise ValueError(f"Tokenizer has no id for split Chinese character: {char!r}")
                    ids.append(char_id)
            else:
                ids.append(token_id)
        return ids


@dataclass(frozen=True)
class VoxCPM2RuntimeConfig:
    model_path: str | Path = "openbmb/VoxCPM2"
    local_files_only: bool = True
    patch_size: int = 4
    feat_dim: int = 64
    audio_start_token: int = 101
    ref_audio_start_token: int = 103
    ref_audio_end_token: int = 104
    encode_sample_rate: int = 16000
    decode_sample_rate: int = 48000
    audio_chunk_size: int = 640
    decode_chunk_size: int = 4
    decode_safety_max_steps: int = 4096
    decode_auto_initial_steps: int = 16
    decode_cache_growth_steps: int = 64
    max_audio_encoder_samples: int = 960_000
    max_decoder_latent_steps: int = 16_384
    max_prefill_seq_len: int = 1_024
    max_decode_cache_seq: int = 6_144


@dataclass(frozen=True)
class SynthesisMetadata:
    decode_steps: int
    stop_reason: StopReason
    requested_max_steps: int
    effective_max_steps: int
    min_steps: int


@dataclass(frozen=True)
class SynthesisResult:
    waveform: np.ndarray
    metadata: SynthesisMetadata


@dataclass
class VoxCPM2OnnxPipeline:
    sessions: OrtSessionFactory = field(default_factory=OrtSessionFactory)
    config: VoxCPM2RuntimeConfig = field(default_factory=VoxCPM2RuntimeConfig)
    _model_dir: Path | None = field(default=None, init=False, repr=False)
    _tokenizer: CharTokenizerWrapper | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_default_artifacts(
        cls,
        *,
        model_path: str | Path = "openbmb/VoxCPM2",
        local_files_only: bool = True,
        onnx_paths: OnnxModelPaths | None = None,
        graph_optimization_level: str = "all",
        execution_mode: str = "sequential",
        log_severity_level: str = "error",
        intra_op_num_threads: int | None = 8,
        inter_op_num_threads: int | None = 1,
        enable_mem_pattern: bool | None = True,
        enable_cpu_mem_arena: bool | None = True,
        enable_mem_reuse: bool | None = True,
        enable_profiling: bool = False,
        profile_file_prefix: Path | None = None,
        prefer_ort_format: bool = True,
        max_audio_encoder_samples: int | None = None,
        max_decoder_latent_steps: int | None = None,
        max_prefill_seq_len: int | None = None,
        max_decode_cache_seq: int | None = None,
    ) -> "VoxCPM2OnnxPipeline":
        default_config = VoxCPM2RuntimeConfig(model_path=model_path, local_files_only=local_files_only)
        config = VoxCPM2RuntimeConfig(
            model_path=model_path,
            local_files_only=local_files_only,
            max_audio_encoder_samples=max_audio_encoder_samples or default_config.max_audio_encoder_samples,
            max_decoder_latent_steps=max_decoder_latent_steps or default_config.max_decoder_latent_steps,
            max_prefill_seq_len=max_prefill_seq_len or default_config.max_prefill_seq_len,
            max_decode_cache_seq=max_decode_cache_seq or default_config.max_decode_cache_seq,
        )
        sessions = OrtSessionFactory(
            paths=onnx_paths or OnnxModelPaths(),
            graph_optimization_level=graph_optimization_level,
            execution_mode=execution_mode,
            log_severity_level=log_severity_level,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
            enable_mem_pattern=enable_mem_pattern,
            enable_cpu_mem_arena=enable_cpu_mem_arena,
            enable_mem_reuse=enable_mem_reuse,
            enable_profiling=enable_profiling,
            profile_file_prefix=profile_file_prefix,
            prefer_ort_format=prefer_ort_format,
        )
        return cls(sessions=sessions, config=config)

    @property
    def model_dir(self) -> Path:
        if self._model_dir is None:
            path = Path(self.config.model_path).expanduser()
            if path.is_dir():
                self._model_dir = path.resolve()
            else:
                self._model_dir = Path(
                    snapshot_download(str(self.config.model_path), local_files_only=self.config.local_files_only)
                ).resolve()
            self._load_model_config()
        return self._model_dir

    @property
    def tokenizer(self) -> CharTokenizerWrapper:
        if self._tokenizer is None:
            self._tokenizer = CharTokenizerWrapper(Tokenizer.from_file(str(self.model_dir / "tokenizer.json")))
        return self._tokenizer

    def validate(self) -> dict[str, Path]:
        paths = self.sessions.validate_paths()
        _ = self.model_dir
        return paths

    def synthesize(
        self,
        text: str,
        *,
        mode: Mode = "text_only",
        voice_design: str | None = None,
        reference_wav_path: str | Path | None = None,
        prompt_wav_path: str | Path | None = None,
        prompt_text: str | None = None,
        max_steps: int = 0,
        min_steps: int = 8,
        cfg_value: float = 2.0,
        seed: int = 0,
    ) -> np.ndarray:
        return self.synthesize_with_metadata(
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
        ).waveform

    def synthesize_with_metadata(
        self,
        text: str,
        *,
        mode: Mode = "text_only",
        voice_design: str | None = None,
        reference_wav_path: str | Path | None = None,
        prompt_wav_path: str | Path | None = None,
        prompt_text: str | None = None,
        max_steps: int = 0,
        min_steps: int = 8,
        cfg_value: float = 2.0,
        seed: int = 0,
        progress_callback: Callable[[int, StopReason | None], None] | None = None,
    ) -> SynthesisResult:
        if max_steps < 0:
            raise ValueError("max_steps must be >= 0; use 0 for auto-until-stop")
        if min_steps < 0:
            raise ValueError("min_steps must be >= 0")

        # max_steps=0 is the production-friendly path: run until the model emits
        # an end-of-audio stop logit. The safety cap prevents an infinite loop if
        # an exported decode chunk graph or state contract is broken.
        effective_max_steps = self.config.decode_safety_max_steps if max_steps == 0 else max_steps
        if effective_max_steps < 1:
            raise ValueError("effective max decode steps must be >= 1")
        # Host code owns mode-specific sequence assembly. The prefill ONNX
        # graph receives only tensors, so text/reference/prompt policy remains
        # inspectable and testable outside the neural graph.
        sequence = self.build_prefill_inputs(
            text,
            mode=mode,
            voice_design=voice_design,
            reference_wav_path=reference_wav_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
        )

        prefill_outputs = self.sessions.prefill.run(PREFILL_OUTPUTS, sequence)
        prefill_state = dict(zip(PREFILL_OUTPUTS, prefill_outputs, strict=True))
        initial_cache_steps = self._initial_decode_cache_steps(
            requested_max_steps=max_steps,
            effective_max_steps=effective_max_steps,
        )
        self._validate_decode_cache_capacity(prefill_state, max_decode_steps=initial_cache_steps)
        state = self._init_fixed_capacity_decode_state(
            prefill_state,
            max_decode_steps=initial_cache_steps,
        )
        rng = np.random.default_rng(seed)
        chunk_size = self.config.decode_chunk_size
        diffusion_noise = np.empty((chunk_size, 1, self.config.feat_dim, self.config.patch_size), dtype=np.float32)
        cfg_tensor = np.array([cfg_value], dtype=np.float32)
        feature_buffer = np.empty(
            (1, effective_max_steps, self.config.patch_size, self.config.feat_dim), dtype=np.float32
        )
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
            "diffusion_noise": diffusion_noise,
            "cfg_value": cfg_tensor,
        }
        decode_outputs: dict[str, np.ndarray] = {}
        decode_session = self.sessions.decode_chunk
        stop_reason: StopReason | None = None
        completed_steps = 0

        while completed_steps < effective_max_steps:
            # One production ONNX call executes a small fixed chunk of exact
            # autoregressive decode steps. Host code still owns stop policy,
            # cache mutation, and the outer loop.
            remaining_steps = effective_max_steps - completed_steps
            accepted_steps = min(chunk_size, remaining_steps)
            self._ensure_decode_cache_capacity(state, required_update_steps=chunk_size)
            decode_inputs["base_k_cache"] = state["base_k_cache"]
            decode_inputs["base_v_cache"] = state["base_v_cache"]
            decode_inputs["residual_k_cache"] = state["residual_k_cache"]
            decode_inputs["residual_v_cache"] = state["residual_v_cache"]
            self._fill_standard_normal(rng, diffusion_noise)
            run_outputs = decode_session.run(DECODE_OUTPUTS, decode_inputs)
            for name, value in zip(DECODE_OUTPUTS, run_outputs, strict=True):
                decode_outputs[name] = value
            accepted_steps = self._accept_decode_chunk_outputs(
                decode_outputs,
                feature_buffer=feature_buffer,
                completed_steps=completed_steps,
                candidate_steps=accepted_steps,
                effective_max_steps=effective_max_steps,
                requested_max_steps=max_steps,
                min_steps=min_steps,
                progress_callback=progress_callback,
            )
            completed_steps += accepted_steps
            stop_reason = self._chunk_stop_reason(
                decode_outputs,
                accepted_steps=accepted_steps,
                completed_steps=completed_steps,
                effective_max_steps=effective_max_steps,
                requested_max_steps=max_steps,
                min_steps=min_steps,
            )
            if stop_reason is not None:
                break
            self._apply_decode_chunk_cache_updates(state, decode_outputs, update_steps=accepted_steps)
            state["lm_hidden"] = decode_outputs["next_lm_hidden"]
            state["residual_hidden"] = decode_outputs["next_residual_hidden"]
            state["prefix_feat_cond"] = decode_outputs["next_prefix_feat_cond"]
            decode_inputs["lm_hidden"] = state["lm_hidden"]
            decode_inputs["residual_hidden"] = state["residual_hidden"]
            decode_inputs["prefix_feat_cond"] = state["prefix_feat_cond"]

        feature_seq = feature_buffer[:, :completed_steps, :, :]
        # AudioVAEDecoder expects [B, latent_dim, latent_steps]. The decode
        # graph emits patch features, so host code performs the reversible layout
        # transform before the final ONNX decoder call.
        decoder_latent = np.transpose(feature_seq, (0, 3, 1, 2)).reshape(1, self.config.feat_dim, -1)
        if decoder_latent.shape[-1] > self.config.max_decoder_latent_steps:
            raise ValueError(
                f"decoder latent length {decoder_latent.shape[-1]} exceeds production bound "
                f"{self.config.max_decoder_latent_steps}; re-export with a larger --max-latent-steps"
            )
        sr_cond = np.array([self.config.decode_sample_rate], dtype=np.int32)
        waveform = self.sessions.audio_decoder.run(["waveform"], {"latent": decoder_latent, "sr_cond": sr_cond})[0]
        return SynthesisResult(
            waveform=waveform[0, 0].astype(np.float32, copy=False),
            metadata=SynthesisMetadata(
                decode_steps=completed_steps,
                stop_reason=stop_reason or ("safety_max_steps" if max_steps == 0 else "max_steps"),
                requested_max_steps=max_steps,
                effective_max_steps=effective_max_steps,
                min_steps=min_steps,
            ),
        )

    @staticmethod
    def _init_fixed_capacity_decode_state(
        prefill_state: dict[str, np.ndarray],
        *,
        max_decode_steps: int,
    ) -> dict[str, np.ndarray]:
        """Allocate fixed-capacity KV caches once for the host decode loop"""

        base_current_length = prefill_state["base_cache_length"].astype(np.int64, copy=False)
        residual_current_length = prefill_state["residual_cache_length"].astype(np.int64, copy=False)
        base_capacity = int(base_current_length[0]) + max_decode_steps
        residual_capacity = int(residual_current_length[0]) + max_decode_steps

        def make_cache(cache: np.ndarray, capacity: int) -> np.ndarray:
            fixed = np.zeros((*cache.shape[:3], capacity, cache.shape[4]), dtype=cache.dtype)
            fixed[:, :, :, : cache.shape[3], :] = cache
            return fixed

        return {
            "lm_hidden": prefill_state["lm_hidden"],
            "residual_hidden": prefill_state["residual_hidden"],
            "prefix_feat_cond": prefill_state["prefix_feat_cond"],
            "base_k_cache": make_cache(prefill_state["base_k_cache"], base_capacity),
            "base_v_cache": make_cache(prefill_state["base_v_cache"], base_capacity),
            "base_current_length": base_current_length,
            "residual_k_cache": make_cache(prefill_state["residual_k_cache"], residual_capacity),
            "residual_v_cache": make_cache(prefill_state["residual_v_cache"], residual_capacity),
            "residual_current_length": residual_current_length,
        }

    def _initial_decode_cache_steps(self, *, requested_max_steps: int, effective_max_steps: int) -> int:
        """Choose initial cache capacity without treating auto-stop as max output length.

        `max_steps=0` means "run until stop logits", but the safety cap is only
        a loop guard. Allocating KV cache for the whole safety cap makes every
        decode_chunk call move thousands of unused positions for short text.
        """

        if requested_max_steps == 0:
            return max(
                self.config.decode_chunk_size,
                min(effective_max_steps, self.config.decode_auto_initial_steps),
            )
        return max(self.config.decode_chunk_size, effective_max_steps)

    def _ensure_decode_cache_capacity(self, state: dict[str, np.ndarray], *, required_update_steps: int) -> None:
        """Grow fixed-capacity cache by blocks only when auto-stop runs long"""

        base_required = int(state["base_current_length"][0]) + required_update_steps
        residual_required = int(state["residual_current_length"][0]) + required_update_steps
        required = max(base_required, residual_required)
        current_capacity = int(state["base_k_cache"].shape[3])
        if required <= current_capacity:
            return
        if required > self.config.max_decode_cache_seq:
            raise ValueError(
                f"decode cache length {required} exceeds production bound {self.config.max_decode_cache_seq}; "
                "lower --max-steps or re-export with a larger --max-cache-seq-bound"
            )
        growth_target = max(required, current_capacity + self.config.decode_cache_growth_steps)
        new_capacity = min(self.config.max_decode_cache_seq, growth_target)
        for key in ("base_k_cache", "base_v_cache", "residual_k_cache", "residual_v_cache"):
            state[key] = self._grow_cache_tensor(state[key], new_capacity)

    @staticmethod
    def _grow_cache_tensor(cache: np.ndarray, new_capacity: int) -> np.ndarray:
        grown = np.zeros((*cache.shape[:3], new_capacity, cache.shape[4]), dtype=cache.dtype)
        grown[:, :, :, : cache.shape[3], :] = cache
        return grown

    @staticmethod
    def _fill_standard_normal(rng: np.random.Generator, target: np.ndarray) -> None:
        """Fill an existing FP32 buffer with Gaussian noise for one decode chunk"""

        try:
            rng.standard_normal(size=target.shape, dtype=target.dtype, out=target)
        except TypeError:
            target[...] = rng.standard_normal(target.shape, dtype=target.dtype)

    @staticmethod
    def _apply_decode_cache_updates(state: dict[str, np.ndarray], outputs: dict[str, np.ndarray]) -> None:
        """Apply one-position K/V updates returned by the fixed-cache graph"""

        base_index = int(state["base_current_length"][0])
        residual_index = int(state["residual_current_length"][0])
        state["base_k_cache"][:, :, :, base_index : base_index + 1, :] = outputs["base_k_update"]
        state["base_v_cache"][:, :, :, base_index : base_index + 1, :] = outputs["base_v_update"]
        state["residual_k_cache"][:, :, :, residual_index : residual_index + 1, :] = outputs["residual_k_update"]
        state["residual_v_cache"][:, :, :, residual_index : residual_index + 1, :] = outputs["residual_v_update"]
        state["base_current_length"] = outputs["next_base_current_length"].astype(np.int64, copy=False)
        state["residual_current_length"] = outputs["next_residual_current_length"].astype(np.int64, copy=False)

    @staticmethod
    def _accept_decode_chunk_outputs(
        outputs: dict[str, np.ndarray],
        *,
        feature_buffer: np.ndarray,
        completed_steps: int,
        candidate_steps: int,
        effective_max_steps: int,
        requested_max_steps: int,
        min_steps: int,
        progress_callback: Callable[[int, StopReason | None], None] | None,
    ) -> int:
        accepted_steps = 0
        for chunk_index in range(candidate_steps):
            next_completed_steps = completed_steps + chunk_index + 1
            step_stop_reason = VoxCPM2OnnxPipeline._step_stop_reason(
                outputs["stop_logits"],
                chunk_index=chunk_index,
                completed_steps=next_completed_steps,
                effective_max_steps=effective_max_steps,
                requested_max_steps=requested_max_steps,
                min_steps=min_steps,
            )
            accepted_steps += 1
            if progress_callback is not None:
                progress_callback(next_completed_steps, step_stop_reason)
            if step_stop_reason is not None:
                break
        feature_buffer[:, completed_steps : completed_steps + accepted_steps, :, :] = outputs["pred_audio_feature"][
            :, :accepted_steps, :, :
        ]
        return accepted_steps

    @staticmethod
    def _chunk_stop_reason(
        outputs: dict[str, np.ndarray],
        *,
        accepted_steps: int,
        completed_steps: int,
        effective_max_steps: int,
        requested_max_steps: int,
        min_steps: int,
    ) -> StopReason | None:
        if accepted_steps < 1:
            return None
        return VoxCPM2OnnxPipeline._step_stop_reason(
            outputs["stop_logits"],
            chunk_index=accepted_steps - 1,
            completed_steps=completed_steps,
            effective_max_steps=effective_max_steps,
            requested_max_steps=requested_max_steps,
            min_steps=min_steps,
        )

    @staticmethod
    def _step_stop_reason(
        stop_logits: np.ndarray,
        *,
        chunk_index: int,
        completed_steps: int,
        effective_max_steps: int,
        requested_max_steps: int,
        min_steps: int,
    ) -> StopReason | None:
        if completed_steps >= min_steps and int(stop_logits[0, chunk_index].argmax()) == 1:
            return "stop_logits"
        if completed_steps >= effective_max_steps:
            return "safety_max_steps" if requested_max_steps == 0 else "max_steps"
        return None

    @staticmethod
    def _apply_decode_chunk_cache_updates(
        state: dict[str, np.ndarray],
        outputs: dict[str, np.ndarray],
        *,
        update_steps: int,
    ) -> None:
        """Apply fixed-capacity K/V updates returned by the chunked graph"""

        base_index = int(state["base_current_length"][0])
        residual_index = int(state["residual_current_length"][0])
        state["base_k_cache"][:, :, :, base_index : base_index + update_steps, :] = outputs["base_k_update"][
            :, :, :, :update_steps, :
        ]
        state["base_v_cache"][:, :, :, base_index : base_index + update_steps, :] = outputs["base_v_update"][
            :, :, :, :update_steps, :
        ]
        state["residual_k_cache"][:, :, :, residual_index : residual_index + update_steps, :] = outputs[
            "residual_k_update"
        ][:, :, :, :update_steps, :]
        state["residual_v_cache"][:, :, :, residual_index : residual_index + update_steps, :] = outputs[
            "residual_v_update"
        ][:, :, :, :update_steps, :]
        state["base_current_length"][0] = base_index + update_steps
        state["residual_current_length"][0] = residual_index + update_steps

    def write_wav(self, path: str | Path, waveform: np.ndarray) -> None:
        # WAV writing is intentionally host code, not ONNX. int16 output keeps
        # the CLI artifact widely readable across platforms.
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        clean = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
        clipped = np.clip(clean, -1.0, 1.0)
        pcm16 = (clipped * np.iinfo(np.int16).max).astype(np.int16)
        wavfile.write(str(target), self.config.decode_sample_rate, pcm16)

    def build_prefill_inputs(
        self,
        text: str,
        *,
        mode: Mode,
        voice_design: str | None,
        reference_wav_path: str | Path | None,
        prompt_wav_path: str | Path | None,
        prompt_text: str | None,
    ) -> dict[str, np.ndarray]:
        if mode == "voice_design" and voice_design:
            text = f"({voice_design}){text}"
        if mode == "controllable_clone" and not reference_wav_path:
            raise ValueError("controllable_clone requires reference_wav_path")
        if mode == "ultimate_clone" and (not prompt_wav_path or prompt_text is None):
            raise ValueError("ultimate_clone requires prompt_wav_path and prompt_text")
        if mode in ("text_only", "voice_design") and (reference_wav_path or prompt_wav_path or prompt_text):
            raise ValueError(f"{mode} does not use reference or prompt audio")

        if mode == "ultimate_clone":
            text_for_tokens = f"{prompt_text}{text}"
        else:
            text_for_tokens = text
        # The prefill graph consumes a single aligned sequence. Text tokens,
        # text_mask, audio_features, and audio_mask must therefore be assembled
        # with identical sequence length before entering ONNX.
        text_tokens = np.array(self.tokenizer(text_for_tokens) + [self.config.audio_start_token], dtype=np.int64)
        text_len = int(text_tokens.shape[0])
        text_pad_feat = np.zeros((text_len, self.config.patch_size, self.config.feat_dim), dtype=np.float32)

        if mode == "controllable_clone":
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(
                self._encode_wav(reference_wav_path, padding_mode="right")
            )
            tokens = np.concatenate([ref_tokens, text_tokens])
            audio_features = np.concatenate([ref_feats, text_pad_feat], axis=0)
            text_mask = np.concatenate([ref_t_mask, np.ones(text_len, dtype=np.float32)])
            audio_mask = np.concatenate([ref_a_mask, np.zeros(text_len, dtype=np.float32)])
        elif mode == "ultimate_clone":
            prompt_feat = self._encode_wav(prompt_wav_path, padding_mode="left")
            prompt_len = int(prompt_feat.shape[0])
            prompt_pad_tokens = np.zeros(prompt_len, dtype=np.int64)
            if reference_wav_path:
                ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(
                    self._encode_wav(reference_wav_path, padding_mode="right")
                )
                tokens = np.concatenate([ref_tokens, text_tokens, prompt_pad_tokens])
                audio_features = np.concatenate([ref_feats, text_pad_feat, prompt_feat], axis=0)
                text_mask = np.concatenate(
                    [ref_t_mask, np.ones(text_len, dtype=np.float32), np.zeros(prompt_len, dtype=np.float32)]
                )
                audio_mask = np.concatenate(
                    [ref_a_mask, np.zeros(text_len, dtype=np.float32), np.ones(prompt_len, dtype=np.float32)]
                )
            else:
                tokens = np.concatenate([text_tokens, prompt_pad_tokens])
                audio_features = np.concatenate([text_pad_feat, prompt_feat], axis=0)
                text_mask = np.concatenate(
                    [np.ones(text_len, dtype=np.float32), np.zeros(prompt_len, dtype=np.float32)]
                )
                audio_mask = np.concatenate(
                    [np.zeros(text_len, dtype=np.float32), np.ones(prompt_len, dtype=np.float32)]
                )
        else:
            tokens = text_tokens
            audio_features = text_pad_feat
            text_mask = np.ones(text_len, dtype=np.float32)
            audio_mask = np.zeros(text_len, dtype=np.float32)

        self._validate_prefill_sequence_length(tokens.shape[0])
        return {
            "text_tokens": tokens[None, :].astype(np.int64, copy=False),
            "text_mask": text_mask[None, :].astype(np.float32, copy=False),
            "audio_features": audio_features[None, :, :, :].astype(np.float32, copy=False),
            "audio_mask": audio_mask[None, :].astype(np.float32, copy=False),
        }

    def _make_ref_prefix(self, ref_feat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ref_len = int(ref_feat.shape[0])
        z1 = np.zeros((1, self.config.patch_size, self.config.feat_dim), dtype=np.float32)
        tokens = np.concatenate(
            [
                np.array([self.config.ref_audio_start_token], dtype=np.int64),
                np.zeros(ref_len, dtype=np.int64),
                np.array([self.config.ref_audio_end_token], dtype=np.int64),
            ]
        )
        feats = np.concatenate([z1, ref_feat, z1], axis=0)
        text_mask = np.concatenate(
            [np.ones(1, dtype=np.float32), np.zeros(ref_len, dtype=np.float32), np.ones(1, dtype=np.float32)]
        )
        audio_mask = np.concatenate(
            [np.zeros(1, dtype=np.float32), np.ones(ref_len, dtype=np.float32), np.zeros(1, dtype=np.float32)]
        )
        return tokens, feats, text_mask, audio_mask

    def _encode_wav(self, wav_path: str | Path | None, *, padding_mode: Literal["left", "right"]) -> np.ndarray:
        if wav_path is None:
            raise ValueError("wav_path is required")
        # Audio file I/O and resampling stay outside ONNX. The AudioVAEEncoder
        # graph receives rank-3 mono FP32 audio padded to a patch-compatible
        # length by host code.
        sample_rate, audio = wavfile.read(str(wav_path))
        mono = self._to_float32_mono(audio)
        if sample_rate != self.config.encode_sample_rate:
            gcd = math.gcd(int(sample_rate), int(self.config.encode_sample_rate))
            mono = resample_poly(
                mono,
                up=self.config.encode_sample_rate // gcd,
                down=sample_rate // gcd,
            ).astype(np.float32)
        patch_len = self.config.patch_size * self.config.audio_chunk_size
        remainder = mono.shape[0] % patch_len
        if remainder:
            pad = patch_len - remainder
            mono = np.pad(mono, (pad, 0) if padding_mode == "left" else (0, pad))
        if mono.shape[0] > self.config.max_audio_encoder_samples:
            raise ValueError(
                f"encoded audio has {mono.shape[0]} samples after padding, exceeding production bound "
                f"{self.config.max_audio_encoder_samples}; trim audio or re-export with a larger --max-samples"
            )
        waveform = mono.reshape(1, 1, -1).astype(np.float32, copy=False)
        latent = self.sessions.audio_encoder.run(["latent"], {"waveform": waveform})[0][0]
        return (
            latent.reshape(self.config.feat_dim, -1, self.config.patch_size)
            .transpose(1, 2, 0)
            .astype(np.float32, copy=False)
        )

    @staticmethod
    def _to_float32_mono(audio: np.ndarray) -> np.ndarray:
        array = np.asarray(audio)
        if array.ndim == 2:
            array = array.mean(axis=1)
        if np.issubdtype(array.dtype, np.floating):
            return array.astype(np.float32, copy=False)
        if array.dtype == np.uint8:
            return ((array.astype(np.float32) - 128.0) / 128.0).astype(np.float32, copy=False)
        info = np.iinfo(array.dtype)
        scale = max(abs(info.min), abs(info.max))
        return (array.astype(np.float32) / float(scale)).astype(np.float32, copy=False)

    def _load_model_config(self) -> None:
        config_path = self.model_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        audio_vae_config = config.get("audio_vae_config") or {}
        object.__setattr__(
            self,
            "config",
            VoxCPM2RuntimeConfig(
                model_path=self.config.model_path,
                local_files_only=self.config.local_files_only,
                patch_size=int(config.get("patch_size", self.config.patch_size)),
                feat_dim=int(config.get("feat_dim", self.config.feat_dim)),
                encode_sample_rate=int(audio_vae_config.get("sample_rate", self.config.encode_sample_rate)),
                decode_sample_rate=int(audio_vae_config.get("out_sample_rate", self.config.decode_sample_rate)),
                audio_chunk_size=int(np.prod(audio_vae_config.get("encoder_rates", [2, 5, 8, 8]))),
                decode_chunk_size=self.config.decode_chunk_size,
                decode_safety_max_steps=self.config.decode_safety_max_steps,
                decode_auto_initial_steps=self.config.decode_auto_initial_steps,
                decode_cache_growth_steps=self.config.decode_cache_growth_steps,
                max_audio_encoder_samples=self.config.max_audio_encoder_samples,
                max_decoder_latent_steps=self.config.max_decoder_latent_steps,
                max_prefill_seq_len=self.config.max_prefill_seq_len,
                max_decode_cache_seq=self.config.max_decode_cache_seq,
            ),
        )

    def _validate_prefill_sequence_length(self, seq_len: int) -> None:
        if seq_len > self.config.max_prefill_seq_len:
            raise ValueError(
                f"prefill sequence length {seq_len} exceeds production bound {self.config.max_prefill_seq_len}; "
                "shorten text/reference/prompt inputs or re-export with a larger --max-seq-len"
            )

    def _validate_decode_cache_capacity(self, prefill_state: dict[str, np.ndarray], *, max_decode_steps: int) -> None:
        base_capacity = int(prefill_state["base_cache_length"][0]) + max_decode_steps
        residual_capacity = int(prefill_state["residual_cache_length"][0]) + max_decode_steps
        required_capacity = max(base_capacity, residual_capacity)
        if required_capacity > self.config.max_decode_cache_seq:
            raise ValueError(
                f"decode cache capacity {required_capacity} exceeds production bound "
                f"{self.config.max_decode_cache_seq}; lower --max-steps or re-export with a larger "
                "--max-cache-seq-bound"
            )
