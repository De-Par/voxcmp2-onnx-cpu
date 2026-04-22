"""Static I/O schemas for proposed VoxCPM2 ONNX module boundaries.

These schemas describe tensor contracts; they are not runtime wrappers.
Shapes use symbolic dimensions where export must read exact values from the
VoxCPM2 config/checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


DType = Literal["float32", "int32", "int64"]


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: DType
    shape: tuple[str, ...]
    description: str


class TransformerCacheSpec(TypedDict):
    key: TensorSpec
    value: TensorSpec
    cache_length: TensorSpec


class TransformerCacheUpdateSpec(TypedDict):
    key: TensorSpec
    value: TensorSpec
    next_length: TensorSpec


class AudioVAEEncoderInputs(TypedDict):
    waveform: TensorSpec


class AudioVAEEncoderOutputs(TypedDict):
    latent: TensorSpec


class AudioVAEDecoderInputs(TypedDict):
    latent: TensorSpec
    sr_cond: TensorSpec


class AudioVAEDecoderOutputs(TypedDict):
    waveform: TensorSpec


class VoxCPM2PrefillInputs(TypedDict):
    text_tokens: TensorSpec
    text_mask: TensorSpec
    audio_features: TensorSpec
    audio_mask: TensorSpec


class VoxCPM2PrefillOutputs(TypedDict):
    lm_hidden: TensorSpec
    residual_hidden: TensorSpec
    prefix_feat_cond: TensorSpec
    base_cache: TransformerCacheSpec
    residual_cache: TransformerCacheSpec


class VoxCPM2DecodeStepInputs(TypedDict):
    lm_hidden: TensorSpec
    residual_hidden: TensorSpec
    prefix_feat_cond: TensorSpec
    base_cache: TransformerCacheSpec
    residual_cache: TransformerCacheSpec
    diffusion_noise: TensorSpec
    cfg_value: TensorSpec


class VoxCPM2DecodeStepOutputs(TypedDict):
    pred_audio_feature: TensorSpec
    decoder_latent: TensorSpec
    stop_logits: TensorSpec
    next_lm_hidden: TensorSpec
    next_residual_hidden: TensorSpec
    next_prefix_feat_cond: TensorSpec
    base_cache_update: TransformerCacheUpdateSpec
    residual_cache_update: TransformerCacheUpdateSpec


def _cache(prefix: str, layers: str, *, seq_label: str = "cache_seq", length_name: str | None = None) -> TransformerCacheSpec:
    length_tensor_name = length_name or f"{prefix}_cache_length"
    return {
        "key": TensorSpec(
            name=f"{prefix}_k_cache",
            dtype="float32",
            shape=(layers, "batch", "kv_heads", seq_label, "head_dim"),
            description="Transformer key cache carried between prefill and decode steps.",
        ),
        "value": TensorSpec(
            name=f"{prefix}_v_cache",
            dtype="float32",
            shape=(layers, "batch", "kv_heads", seq_label, "head_dim"),
            description="Transformer value cache carried between prefill and decode steps.",
        ),
        "cache_length": TensorSpec(
            name=length_tensor_name,
            dtype="int64",
            shape=("1",),
            description="Number of valid cache positions.",
        ),
    }


def _cache_update(prefix: str, layers: str) -> TransformerCacheUpdateSpec:
    return {
        "key": TensorSpec(
            name=f"{prefix}_k_update",
            dtype="float32",
            shape=(layers, "batch", "kv_heads", "1", "head_dim"),
            description="One newly generated key position to write at current_length in host code.",
        ),
        "value": TensorSpec(
            name=f"{prefix}_v_update",
            dtype="float32",
            shape=(layers, "batch", "kv_heads", "1", "head_dim"),
            description="One newly generated value position to write at current_length in host code.",
        ),
        "next_length": TensorSpec(
            name=f"next_{prefix}_current_length",
            dtype="int64",
            shape=("1",),
            description="Updated valid cache length after applying the one-position update.",
        ),
    }


AUDIO_VAE_ENCODER_INPUTS: AudioVAEEncoderInputs = {
    "waveform": TensorSpec(
        name="waveform",
        dtype="float32",
        shape=("batch", "1", "samples"),
        description="Mono waveform already loaded and resampled by host code.",
    )
}

AUDIO_VAE_ENCODER_OUTPUTS: AudioVAEEncoderOutputs = {
    "latent": TensorSpec(
        name="latent",
        dtype="float32",
        shape=("batch", "latent_dim_64", "latent_steps"),
        description="AudioVAE continuous latent; host reshapes it into VoxCPM2 patch features.",
    )
}

AUDIO_VAE_DECODER_INPUTS: AudioVAEDecoderInputs = {
    "latent": TensorSpec(
        name="latent",
        dtype="float32",
        shape=("batch", "latent_dim_64", "latent_steps"),
        description="Generated latent sequence after host concatenates patch features.",
    ),
    "sr_cond": TensorSpec(
        name="sr_cond",
        dtype="int32",
        shape=("batch",),
        description="Explicit output sample-rate condition for AudioVAE V2 checkpoints that use it.",
    ),
}

AUDIO_VAE_DECODER_OUTPUTS: AudioVAEDecoderOutputs = {
    "waveform": TensorSpec(
        name="waveform",
        dtype="float32",
        shape=("batch", "1", "samples"),
        description="Decoded waveform; host performs final trimming and WAV writing.",
    )
}

VOXCPM2_PREFILL_INPUTS: VoxCPM2PrefillInputs = {
    "text_tokens": TensorSpec(
        name="text_tokens",
        dtype="int64",
        shape=("batch", "seq"),
        description="Token IDs produced by host tokenizer, including VoxCPM2 audio/ref marker tokens.",
    ),
    "text_mask": TensorSpec(
        name="text_mask",
        dtype="float32",
        shape=("batch", "seq"),
        description="1.0 where the sequence position is text/control token, else 0.0.",
    ),
    "audio_features": TensorSpec(
        name="audio_features",
        dtype="float32",
        shape=("batch", "seq", "patch_size_4", "latent_dim_64"),
        description="Reference/prompt audio features aligned with token sequence; zeros at text positions.",
    ),
    "audio_mask": TensorSpec(
        name="audio_mask",
        dtype="float32",
        shape=("batch", "seq"),
        description="1.0 where the sequence position carries audio features, else 0.0.",
    ),
}

VOXCPM2_PREFILL_OUTPUTS: VoxCPM2PrefillOutputs = {
    "lm_hidden": TensorSpec(
        name="lm_hidden",
        dtype="float32",
        shape=("batch", "hidden_2048"),
        description="Last base LM hidden state used to start decode_step.",
    ),
    "residual_hidden": TensorSpec(
        name="residual_hidden",
        dtype="float32",
        shape=("batch", "hidden_2048"),
        description="Last residual LM hidden state used to start decode_step.",
    ),
    "prefix_feat_cond": TensorSpec(
        name="prefix_feat_cond",
        dtype="float32",
        shape=("batch", "patch_size_4", "latent_dim_64"),
        description="Last audio feature condition for the first diffusion decode step.",
    ),
    "base_cache": _cache("base", "base_layers_28", seq_label="max_cache_seq", length_name="base_current_length"),
    "residual_cache": _cache(
        "residual", "residual_layers_8", seq_label="max_cache_seq", length_name="residual_current_length"
    ),
}

VOXCPM2_DECODE_STEP_INPUTS: VoxCPM2DecodeStepInputs = {
    "lm_hidden": VOXCPM2_PREFILL_OUTPUTS["lm_hidden"],
    "residual_hidden": VOXCPM2_PREFILL_OUTPUTS["residual_hidden"],
    "prefix_feat_cond": VOXCPM2_PREFILL_OUTPUTS["prefix_feat_cond"],
    "base_cache": _cache("base", "base_layers_28"),
    "residual_cache": _cache("residual", "residual_layers_8"),
    "diffusion_noise": TensorSpec(
        name="diffusion_noise",
        dtype="float32",
        shape=("batch", "latent_dim_64", "patch_size_4"),
        description="Host-supplied initial noise for deterministic CFM sampling.",
    ),
    "cfg_value": TensorSpec(
        name="cfg_value",
        dtype="float32",
        shape=("1",),
        description="Classifier-free guidance value used by LocDiT/CFM sampling.",
    ),
}

VOXCPM2_DECODE_STEP_OUTPUTS: VoxCPM2DecodeStepOutputs = {
    "pred_audio_feature": TensorSpec(
        name="pred_audio_feature",
        dtype="float32",
        shape=("batch", "1", "patch_size_4", "latent_dim_64"),
        description="One generated audio-feature patch to append in host code.",
    ),
    "decoder_latent": TensorSpec(
        name="decoder_latent",
        dtype="float32",
        shape=("batch", "latent_dim_64", "patch_size_4"),
        description="Same generated patch arranged for AudioVAEDecoder input.",
    ),
    "stop_logits": TensorSpec(
        name="stop_logits",
        dtype="float32",
        shape=("batch", "2"),
        description="Stop predictor logits; host applies argmax and min-length policy.",
    ),
    "next_lm_hidden": TensorSpec(
        name="next_lm_hidden",
        dtype="float32",
        shape=("batch", "hidden_2048"),
        description="Updated base LM hidden state for the next decode step.",
    ),
    "next_residual_hidden": TensorSpec(
        name="next_residual_hidden",
        dtype="float32",
        shape=("batch", "hidden_2048"),
        description="Updated residual LM hidden state for the next decode step.",
    ),
    "next_prefix_feat_cond": TensorSpec(
        name="next_prefix_feat_cond",
        dtype="float32",
        shape=("batch", "patch_size_4", "latent_dim_64"),
        description="Generated patch reused as the next diffusion condition.",
    ),
    "base_cache_update": _cache_update("base", "base_layers_28"),
    "residual_cache_update": _cache_update("residual", "residual_layers_8"),
}
