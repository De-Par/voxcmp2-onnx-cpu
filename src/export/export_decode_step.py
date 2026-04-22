#!/usr/bin/env python3
"""Export one VoxCPM2 decode step to ONNX"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .common import (
        MODULE_EXPORT_CONTRACTS,
        PrecisionProfile,
        add_precision_argument,
        cast_tensor_if_needed,
        ensure_output_dir,
        export_onnx_graph,
        get_precision_profile,
        print_export_plan,
        resolve_output_path,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        MODULE_EXPORT_CONTRACTS,
        PrecisionProfile,
        add_precision_argument,
        cast_tensor_if_needed,
        ensure_output_dir,
        export_onnx_graph,
        get_precision_profile,
        print_export_plan,
        resolve_output_path,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _export_helpers():
    _ensure_repo_root_on_path()

    from src.export.export_audio_vae_decoder import _resolve_model_path
    from src.export.export_prefill import load_voxcpm2_prefill_model

    return _resolve_model_path, load_voxcpm2_prefill_model


MODULE_KEY = "decode_step"
INPUT_NAMES = list(MODULE_EXPORT_CONTRACTS[MODULE_KEY].input_names)
OUTPUT_NAMES = list(MODULE_EXPORT_CONTRACTS[MODULE_KEY].output_names)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    q = cast_tensor_if_needed(q, torch.float32)
    k = cast_tensor_if_needed(k, torch.float32)
    cos = cast_tensor_if_needed(cos, torch.float32)
    sin = cast_tensor_if_needed(sin, torch.float32)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return cast_tensor_if_needed(q_embed, orig_dtype), cast_tensor_if_needed(k_embed, orig_dtype)


class VoxCPM2DecodeStepWrapper(torch.nn.Module):
    """ONNX-facing one-step decode wrapper with explicit tensor state.

    The official implementation stores KV state in Python ``StaticKVCache`` and
    mutates it inside ``MiniCPMModel.forward_step``. This wrapper keeps the same
    per-layer math but accepts fixed-capacity cache tensors and returns
    one-position K/V updates, so host code owns the autoregressive loop and
    cache mutation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        inference_timesteps: int = 10,
        precision: PrecisionProfile | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.inference_timesteps = inference_timesteps
        self.precision = precision or get_precision_profile("fp32")
        self.compute_dtype = self.precision.torch_compute_dtype()
        self.host_float_dtype = self.precision.torch_host_float_dtype()
        t_span = torch.linspace(1, 0, inference_timesteps + 1, dtype=torch.float32)
        t_span = t_span + (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        self.register_buffer("t_span", t_span, persistent=False)

    @staticmethod
    def _attention_step(
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor] | None,
        layer_k_cache: torch.Tensor,
        layer_v_cache: torch.Tensor,
        current_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, _ = hidden_states.size()
        attn = layer.self_attn

        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        positions = torch.arange(layer_k_cache.shape[2], device=layer_k_cache.device, dtype=torch.long)
        write_mask = (positions == current_length).reshape(1, 1, -1, 1)
        valid_mask = (positions <= current_length).reshape(1, 1, 1, -1)
        attention_k = torch.where(write_mask, key_states, layer_k_cache)
        attention_v = torch.where(write_mask, value_states, layer_v_cache)

        attn_output = F.scaled_dot_product_attention(
            query_states.contiguous(),
            attention_k.contiguous(),
            attention_v.contiguous(),
            attn_mask=valid_mask,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, attn.num_heads * attn.head_dim)
        attn_output = attn.o_proj(attn_output)
        return attn_output, key_states, value_states

    def _transformer_step(
        self,
        lm: torch.nn.Module,
        inputs_embeds: torch.Tensor,
        current_length: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        position_id = cast_tensor_if_needed(current_length, torch.long)
        if lm.rope_emb is not None:
            position_emb = lm.rope_emb(position_id)
        else:
            position_emb = None

        hidden_states = inputs_embeds
        next_k_layers = []
        next_v_layers = []
        # Reimplement only the official single-step transformer path needed to
        # replace Python-side mutable cache with tensor-in/tensor-out state.
        for layer_idx, decoder_layer in enumerate(lm.layers):
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            hidden_states, layer_k, layer_v = self._attention_step(
                decoder_layer,
                hidden_states,
                position_emb,
                k_cache[layer_idx],
                v_cache[layer_idx],
                current_length,
            )
            if decoder_layer.use_mup:
                hidden_states = residual + hidden_states * (
                    decoder_layer.scale_depth / math.sqrt(decoder_layer.num_hidden_layers)
                )
            else:
                hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            if decoder_layer.use_mup:
                hidden_states = residual + hidden_states * (
                    decoder_layer.scale_depth / math.sqrt(decoder_layer.num_hidden_layers)
                )
            else:
                hidden_states = residual + hidden_states

            next_k_layers.append(layer_k)
            next_v_layers.append(layer_v)

        hidden_states = lm.norm(hidden_states)
        return hidden_states, torch.stack(next_k_layers, dim=0), torch.stack(next_v_layers, dim=0), current_length + 1

    def forward(
        self,
        lm_hidden: torch.Tensor,
        residual_hidden: torch.Tensor,
        prefix_feat_cond: torch.Tensor,
        base_k_cache: torch.Tensor,
        base_v_cache: torch.Tensor,
        base_current_length: torch.Tensor,
        residual_k_cache: torch.Tensor,
        residual_v_cache: torch.Tensor,
        residual_current_length: torch.Tensor,
        diffusion_noise: torch.Tensor,
        cfg_value: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self._forward_compute(
            lm_hidden,
            residual_hidden,
            prefix_feat_cond,
            base_k_cache,
            base_v_cache,
            base_current_length,
            residual_k_cache,
            residual_v_cache,
            residual_current_length,
            diffusion_noise,
            cfg_value,
        )
        (
            pred_audio_feature,
            decoder_latent,
            stop_logits,
            next_lm_hidden,
            next_residual_hidden,
            pred_feat,
            base_k_update,
            base_v_update,
            next_base_current_length,
            residual_k_update,
            residual_v_update,
            next_residual_current_length,
        ) = outputs

        return (
            cast_tensor_if_needed(pred_audio_feature, self.host_float_dtype),
            cast_tensor_if_needed(decoder_latent, self.host_float_dtype),
            cast_tensor_if_needed(stop_logits, self.host_float_dtype),
            cast_tensor_if_needed(next_lm_hidden, self.host_float_dtype),
            cast_tensor_if_needed(next_residual_hidden, self.host_float_dtype),
            cast_tensor_if_needed(pred_feat, self.host_float_dtype),
            cast_tensor_if_needed(base_k_update, self.host_float_dtype),
            cast_tensor_if_needed(base_v_update, self.host_float_dtype),
            next_base_current_length,
            cast_tensor_if_needed(residual_k_update, self.host_float_dtype),
            cast_tensor_if_needed(residual_v_update, self.host_float_dtype),
            next_residual_current_length,
        )

    def _forward_compute(
        self,
        lm_hidden: torch.Tensor,
        residual_hidden: torch.Tensor,
        prefix_feat_cond: torch.Tensor,
        base_k_cache: torch.Tensor,
        base_v_cache: torch.Tensor,
        base_current_length: torch.Tensor,
        residual_k_cache: torch.Tensor,
        residual_v_cache: torch.Tensor,
        residual_current_length: torch.Tensor,
        diffusion_noise: torch.Tensor,
        cfg_value: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        model = self.model
        # Host code supplies FP32 tensors for both production profiles. The
        # wrapper localizes profile-specific compute dtype at the graph edge.
        lm_hidden = cast_tensor_if_needed(lm_hidden, self.compute_dtype)
        residual_hidden = cast_tensor_if_needed(residual_hidden, self.compute_dtype)
        prefix_feat_cond = cast_tensor_if_needed(prefix_feat_cond, self.compute_dtype)
        base_k_cache = cast_tensor_if_needed(base_k_cache, self.compute_dtype)
        base_v_cache = cast_tensor_if_needed(base_v_cache, self.compute_dtype)
        residual_k_cache = cast_tensor_if_needed(residual_k_cache, self.compute_dtype)
        residual_v_cache = cast_tensor_if_needed(residual_v_cache, self.compute_dtype)
        diffusion_noise = cast_tensor_if_needed(diffusion_noise, self.compute_dtype)
        cfg_value = cast_tensor_if_needed(cfg_value, self.compute_dtype)

        dit_hidden_1 = model.lm_to_dit_proj(lm_hidden)
        dit_hidden_2 = model.res_to_dit_proj(residual_hidden)
        dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)

        decoder_latent = model.feat_decoder.solve_euler(
            x=diffusion_noise,
            t_span=cast_tensor_if_needed(self.t_span, diffusion_noise.dtype),
            mu=dit_hidden,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            cfg_value=cfg_value.reshape(()),
            use_cfg_zero_star=True,
        )
        pred_feat = decoder_latent.transpose(1, 2)
        pred_audio_feature = pred_feat.unsqueeze(1)

        curr_embed = model.feat_encoder(pred_audio_feature)
        curr_embed = model.enc_to_lm_proj(curr_embed)[:, 0, :]

        stop_logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))

        next_lm_hidden, base_k_update, base_v_update, next_base_current_length = self._transformer_step(
            model.base_lm,
            curr_embed,
            base_current_length,
            base_k_cache,
            base_v_cache,
        )
        next_lm_hidden = model.fsq_layer(next_lm_hidden)
        curr_residual_input = model.fusion_concat_proj(torch.cat((next_lm_hidden, curr_embed), dim=-1))
        (
            next_residual_hidden,
            residual_k_update,
            residual_v_update,
            next_residual_current_length,
        ) = self._transformer_step(
            model.residual_lm,
            curr_residual_input,
            residual_current_length,
            residual_k_cache,
            residual_v_cache,
        )

        return (
            pred_audio_feature,
            decoder_latent,
            stop_logits,
            next_lm_hidden,
            next_residual_hidden,
            pred_feat,
            base_k_update,
            base_v_update,
            next_base_current_length,
            residual_k_update,
            residual_v_update,
            next_residual_current_length,
        )


def make_synthetic_decode_step_inputs(
    *,
    batch_size: int,
    current_length: int,
    max_cache_seq: int,
    hidden_size: int,
    patch_size: int,
    feat_dim: int,
    base_layers: int,
    residual_layers: int,
    kv_heads: int,
    head_dim: int,
    cfg_value: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    if current_length < 1:
        raise ValueError("--current-length must be >= 1")
    if max_cache_seq <= current_length:
        raise ValueError("--max-cache-seq must be greater than --current-length")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "lm_hidden": torch.randn(batch_size, hidden_size, dtype=torch.float32, generator=generator),
        "residual_hidden": torch.randn(batch_size, hidden_size, dtype=torch.float32, generator=generator),
        "prefix_feat_cond": torch.randn(batch_size, patch_size, feat_dim, dtype=torch.float32, generator=generator),
        "base_k_cache": torch.randn(
            base_layers, batch_size, kv_heads, max_cache_seq, head_dim, dtype=torch.float32, generator=generator
        ),
        "base_v_cache": torch.randn(
            base_layers, batch_size, kv_heads, max_cache_seq, head_dim, dtype=torch.float32, generator=generator
        ),
        "base_current_length": torch.tensor([current_length], dtype=torch.long),
        "residual_k_cache": torch.randn(
            residual_layers, batch_size, kv_heads, max_cache_seq, head_dim, dtype=torch.float32, generator=generator
        ),
        "residual_v_cache": torch.randn(
            residual_layers, batch_size, kv_heads, max_cache_seq, head_dim, dtype=torch.float32, generator=generator
        ),
        "residual_current_length": torch.tensor([current_length], dtype=torch.long),
        "diffusion_noise": torch.randn(batch_size, feat_dim, patch_size, dtype=torch.float32, generator=generator),
        "cfg_value": torch.tensor([cfg_value], dtype=torch.float32),
    }


def _model_dims(model: torch.nn.Module) -> dict[str, int]:
    lm_config = model.config.lm_config
    return {
        "hidden_size": lm_config.hidden_size,
        "patch_size": model.patch_size,
        "feat_dim": model.feat_dim,
        "base_layers": lm_config.num_hidden_layers,
        "residual_layers": model.config.residual_lm_num_layers,
        "kv_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.kv_channels or (lm_config.hidden_size // lm_config.num_attention_heads),
    }


def _dynamic_shapes() -> dict[str, dict[int, Any]]:
    max_cache_seq = torch.export.Dim("max_cache_seq", min=2)
    return {
        "lm_hidden": {},
        "residual_hidden": {},
        "prefix_feat_cond": {},
        "base_k_cache": {3: max_cache_seq},
        "base_v_cache": {3: max_cache_seq},
        "base_current_length": {},
        "residual_k_cache": {3: max_cache_seq},
        "residual_v_cache": {3: max_cache_seq},
        "residual_current_length": {},
        "diffusion_noise": {},
        "cfg_value": {},
    }


def _shape_report(
    model: torch.nn.Module,
    batch_size: int,
    current_length: int,
    max_cache_seq: int,
    inference_timesteps: int,
) -> dict[str, Any]:
    dims = _model_dims(model)
    return {
        "inference_timesteps": inference_timesteps,
        "inputs": {
            "lm_hidden": {"dtype": "float32", "dims": [f"static:{batch_size}", f"static:{dims['hidden_size']}"]},
            "residual_hidden": {"dtype": "float32", "dims": [f"static:{batch_size}", f"static:{dims['hidden_size']}"]},
            "prefix_feat_cond": {
                "dtype": "float32",
                "dims": [f"static:{batch_size}", f"static:{dims['patch_size']}", f"static:{dims['feat_dim']}"],
            },
            "base_k_cache": {
                "dtype": "float32",
                "dims": [
                    f"static:{dims['base_layers']}",
                    f"static:{batch_size}",
                    f"static:{dims['kv_heads']}",
                    "dynamic:max_cache_seq",
                    f"static:{dims['head_dim']}",
                ],
            },
            "base_v_cache": {
                "dtype": "float32",
                "dims": [
                    f"static:{dims['base_layers']}",
                    f"static:{batch_size}",
                    f"static:{dims['kv_heads']}",
                    "dynamic:max_cache_seq",
                    f"static:{dims['head_dim']}",
                ],
            },
            "base_current_length": {"dtype": "int64", "dims": ["static:1"], "example_value": current_length},
            "residual_k_cache": {
                "dtype": "float32",
                "dims": [
                    f"static:{dims['residual_layers']}",
                    f"static:{batch_size}",
                    f"static:{dims['kv_heads']}",
                    "dynamic:max_cache_seq",
                    f"static:{dims['head_dim']}",
                ],
            },
            "residual_v_cache": {
                "dtype": "float32",
                "dims": [
                    f"static:{dims['residual_layers']}",
                    f"static:{batch_size}",
                    f"static:{dims['kv_heads']}",
                    "dynamic:max_cache_seq",
                    f"static:{dims['head_dim']}",
                ],
            },
            "residual_current_length": {"dtype": "int64", "dims": ["static:1"], "example_value": current_length},
            "diffusion_noise": {
                "dtype": "float32",
                "dims": [f"static:{batch_size}", f"static:{dims['feat_dim']}", f"static:{dims['patch_size']}"],
            },
            "cfg_value": {"dtype": "float32", "dims": ["static:1"]},
        },
        "outputs": {
            "pred_audio_feature": [
                f"static:{batch_size}",
                "static:1",
                f"static:{dims['patch_size']}",
                f"static:{dims['feat_dim']}",
            ],
            "decoder_latent": [f"static:{batch_size}", f"static:{dims['feat_dim']}", f"static:{dims['patch_size']}"],
            "stop_logits": [f"static:{batch_size}", "static:2"],
            "next_lm_hidden": [f"static:{batch_size}", f"static:{dims['hidden_size']}"],
            "next_residual_hidden": [f"static:{batch_size}", f"static:{dims['hidden_size']}"],
            "next_prefix_feat_cond": [
                f"static:{batch_size}",
                f"static:{dims['patch_size']}",
                f"static:{dims['feat_dim']}",
            ],
            "base_k_update": [
                f"static:{dims['base_layers']}",
                f"static:{batch_size}",
                f"static:{dims['kv_heads']}",
                "static:1",
                f"static:{dims['head_dim']}",
            ],
            "base_v_update": [
                f"static:{dims['base_layers']}",
                f"static:{batch_size}",
                f"static:{dims['kv_heads']}",
                "static:1",
                f"static:{dims['head_dim']}",
            ],
            "next_base_current_length": ["static:1"],
            "residual_k_update": [
                f"static:{dims['residual_layers']}",
                f"static:{batch_size}",
                f"static:{dims['kv_heads']}",
                "static:1",
                f"static:{dims['head_dim']}",
            ],
            "residual_v_update": [
                f"static:{dims['residual_layers']}",
                f"static:{batch_size}",
                f"static:{dims['kv_heads']}",
                "static:1",
                f"static:{dims['head_dim']}",
            ],
            "next_residual_current_length": ["static:1"],
        },
        "example": {"batch_size": batch_size, "current_length": current_length, "max_cache_seq": max_cache_seq},
    }


def export_decode_step(args: argparse.Namespace) -> None:
    _resolve_model_path, load_voxcpm2_prefill_model = _export_helpers()
    precision = get_precision_profile(args.precision)
    model_dir = _resolve_model_path(args.model_path, args.local_files_only)
    output_path = resolve_output_path(args.output, MODULE_KEY, precision)
    ensure_output_dir(output_path)

    model = load_voxcpm2_prefill_model(model_dir, precision)
    wrapper = VoxCPM2DecodeStepWrapper(model, inference_timesteps=args.inference_timesteps, precision=precision).eval()
    dims = _model_dims(model)
    inputs = make_synthetic_decode_step_inputs(
        batch_size=args.batch_size,
        current_length=args.current_length,
        max_cache_seq=args.max_cache_seq,
        cfg_value=args.cfg_value,
        seed=args.seed,
        **dims,
    )

    print_export_plan(
        module_key=MODULE_KEY,
        precision=precision,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        shape_report=_shape_report(
            model,
            args.batch_size,
            args.current_length,
            args.max_cache_seq,
            args.inference_timesteps,
        ),
        output_path=output_path,
    )
    export_onnx_graph(
        wrapper=wrapper,
        inputs=tuple(inputs[name] for name in INPUT_NAMES),
        output_path=output_path,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset=args.opset,
        dynamic_shapes=_dynamic_shapes(),
    )

    print(f"exported={output_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one VoxCPM2 autoregressive decode step to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/export/export_decode_step.py "
            "--precision fp32 --current-length 16 --max-cache-seq 64"
        ),
    )
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ONNX output path. Defaults to models/onnx/<precision>/decode_step/voxcpm2_decode_step.onnx.",
    )
    add_precision_argument(parser)
    parser.add_argument("--batch-size", type=int, default=1, help="Example batch dimension used during export.")
    parser.add_argument(
        "--current-length", type=int, default=16, help="Example valid KV-cache length entering the decode step."
    )
    parser.add_argument(
        "--max-cache-seq",
        type=int,
        default=64,
        help="Example fixed KV-cache capacity used during export; must exceed --current-length.",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Fixed CFM/LocDiT solver steps embedded in this one-step graph.",
    )
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Example classifier-free guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="PyTorch RNG seed for synthetic export inputs.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version for torch.onnx.export.")
    parser.add_argument(
        "--local-files-only", action="store_true", default=True, help="Require local Hugging Face cache/model files."
    )
    parser.add_argument(
        "--allow-download",
        action="store_false",
        dest="local_files_only",
        help="Allow snapshot_download to fetch missing files.",
    )
    return parser


def main() -> int:
    export_decode_step(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
