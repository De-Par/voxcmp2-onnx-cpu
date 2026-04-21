"""CPU-only VoxCPM2 ONNX Runtime synthesis pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from src.runtime.session_factory import OnnxModelPaths, OrtSessionFactory


Mode = Literal["text_only", "voice_design", "controllable_clone", "ultimate_clone"]

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
    "base_cache_length",
    "residual_k_cache",
    "residual_v_cache",
    "residual_cache_length",
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
    "next_base_k_cache",
    "next_base_v_cache",
    "next_base_cache_length",
    "next_residual_k_cache",
    "next_residual_v_cache",
    "next_residual_cache_length",
]


class CharTokenizerWrapper:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.multichar_tokens = {
            token for token in tokenizer.get_vocab().keys() if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
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
    ) -> "VoxCPM2OnnxPipeline":
        config = VoxCPM2RuntimeConfig(model_path=model_path, local_files_only=local_files_only)
        sessions = OrtSessionFactory(paths=onnx_paths or OnnxModelPaths())
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
        max_steps: int = 1,
        min_steps: int = 0,
        cfg_value: float = 2.0,
        seed: int = 0,
    ) -> np.ndarray:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        sequence = self.build_prefill_inputs(
            text,
            mode=mode,
            voice_design=voice_design,
            reference_wav_path=reference_wav_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
        )

        prefill_outputs = self.sessions.prefill.run(PREFILL_OUTPUTS, sequence)
        state = dict(zip(PREFILL_OUTPUTS, prefill_outputs, strict=True))
        rng = np.random.default_rng(seed)
        generated: list[np.ndarray] = []

        for step in range(max_steps):
            decode_inputs = {
                "lm_hidden": state["lm_hidden"],
                "residual_hidden": state["residual_hidden"],
                "prefix_feat_cond": state["prefix_feat_cond"],
                "base_k_cache": state["base_k_cache"],
                "base_v_cache": state["base_v_cache"],
                "base_cache_length": state["base_cache_length"],
                "residual_k_cache": state["residual_k_cache"],
                "residual_v_cache": state["residual_v_cache"],
                "residual_cache_length": state["residual_cache_length"],
                "diffusion_noise": rng.standard_normal((1, self.config.feat_dim, self.config.patch_size), dtype=np.float32),
                "cfg_value": np.array([cfg_value], dtype=np.float32),
            }
            outputs = dict(zip(DECODE_OUTPUTS, self.sessions.decode_step.run(DECODE_OUTPUTS, decode_inputs), strict=True))
            generated.append(outputs["pred_audio_feature"])
            if step >= min_steps and int(np.argmax(outputs["stop_logits"], axis=-1)[0]) == 1:
                break
            state = {
                "lm_hidden": outputs["next_lm_hidden"],
                "residual_hidden": outputs["next_residual_hidden"],
                "prefix_feat_cond": outputs["next_prefix_feat_cond"],
                "base_k_cache": outputs["next_base_k_cache"],
                "base_v_cache": outputs["next_base_v_cache"],
                "base_cache_length": outputs["next_base_cache_length"],
                "residual_k_cache": outputs["next_residual_k_cache"],
                "residual_v_cache": outputs["next_residual_v_cache"],
                "residual_cache_length": outputs["next_residual_cache_length"],
            }

        feature_seq = np.concatenate(generated, axis=1)
        decoder_latent = np.transpose(feature_seq, (0, 3, 1, 2)).reshape(1, self.config.feat_dim, -1)
        sr_cond = np.array([self.config.decode_sample_rate], dtype=np.int32)
        waveform = self.sessions.audio_decoder.run(["waveform"], {"latent": decoder_latent, "sr_cond": sr_cond})[0]
        return waveform[0, 0].astype(np.float32, copy=False)

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
                text_mask = np.concatenate([ref_t_mask, np.ones(text_len, dtype=np.float32), np.zeros(prompt_len, dtype=np.float32)])
                audio_mask = np.concatenate([ref_a_mask, np.zeros(text_len, dtype=np.float32), np.ones(prompt_len, dtype=np.float32)])
            else:
                tokens = np.concatenate([text_tokens, prompt_pad_tokens])
                audio_features = np.concatenate([text_pad_feat, prompt_feat], axis=0)
                text_mask = np.concatenate([np.ones(text_len, dtype=np.float32), np.zeros(prompt_len, dtype=np.float32)])
                audio_mask = np.concatenate([np.zeros(text_len, dtype=np.float32), np.ones(prompt_len, dtype=np.float32)])
        else:
            tokens = text_tokens
            audio_features = text_pad_feat
            text_mask = np.ones(text_len, dtype=np.float32)
            audio_mask = np.zeros(text_len, dtype=np.float32)

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
        text_mask = np.concatenate([np.ones(1, dtype=np.float32), np.zeros(ref_len, dtype=np.float32), np.ones(1, dtype=np.float32)])
        audio_mask = np.concatenate([np.zeros(1, dtype=np.float32), np.ones(ref_len, dtype=np.float32), np.zeros(1, dtype=np.float32)])
        return tokens, feats, text_mask, audio_mask

    def _encode_wav(self, wav_path: str | Path | None, *, padding_mode: Literal["left", "right"]) -> np.ndarray:
        if wav_path is None:
            raise ValueError("wav_path is required")
        audio, sample_rate = sf.read(str(wav_path), always_2d=True, dtype="float32")
        mono = audio.mean(axis=1)
        if sample_rate != self.config.encode_sample_rate:
            mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=self.config.encode_sample_rate).astype(np.float32)
        patch_len = self.config.patch_size * self.config.audio_chunk_size
        remainder = mono.shape[0] % patch_len
        if remainder:
            pad = patch_len - remainder
            mono = np.pad(mono, (pad, 0) if padding_mode == "left" else (0, pad))
        waveform = mono.reshape(1, 1, -1).astype(np.float32, copy=False)
        latent = self.sessions.audio_encoder.run(["latent"], {"waveform": waveform})[0][0]
        return latent.reshape(self.config.feat_dim, -1, self.config.patch_size).transpose(1, 2, 0).astype(np.float32, copy=False)

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
            ),
        )
