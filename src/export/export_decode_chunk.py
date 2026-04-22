#!/usr/bin/env python3
"""Export a chunk of VoxCPM2 autoregressive decode steps to ONNX"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

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
    from .export_decode_step import (
        VoxCPM2DecodeStepWrapper,
        _export_helpers,
        _model_dims,
        make_synthetic_decode_step_inputs,
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
    from export_decode_step import (  # type: ignore[no-redef]
        VoxCPM2DecodeStepWrapper,
        _export_helpers,
        _model_dims,
        make_synthetic_decode_step_inputs,
    )


MODULE_KEY = "decode_chunk"
INPUT_NAMES = list(MODULE_EXPORT_CONTRACTS[MODULE_KEY].input_names)
OUTPUT_NAMES = list(MODULE_EXPORT_CONTRACTS[MODULE_KEY].output_names)
DEFAULT_CHUNK_SIZE = 4


class VoxCPM2DecodeChunkWrapper(VoxCPM2DecodeStepWrapper):
    """ONNX-facing chunked decode wrapper.

    The wrapper executes ``chunk_size`` exact one-step decode computations in a
    statically unrolled loop. Between internal steps it applies the same
    fixed-capacity cache update that host code applies for the one-step utility
    path. This reduces ONNX Runtime session boundary crossings without moving
    stop policy or the outer orchestration loop into ONNX.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        inference_timesteps: int = 10,
        precision: PrecisionProfile | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        super().__init__(model, inference_timesteps=inference_timesteps, precision=precision)
        self.chunk_size = chunk_size

    @staticmethod
    def _write_cache_position(cache: torch.Tensor, update: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(cache.shape[3], device=cache.device, dtype=torch.long)
        write_mask = (positions == current_length).reshape(1, 1, 1, -1, 1)
        return torch.where(write_mask, update, cache)

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
        diffusion_noise = cast_tensor_if_needed(diffusion_noise, self.compute_dtype)

        pred_audio_features = []
        decoder_latents = []
        stop_logits_values = []
        base_k_updates = []
        base_v_updates = []
        residual_k_updates = []
        residual_v_updates = []

        for chunk_index in range(self.chunk_size):
            step_outputs = self._forward_compute(
                lm_hidden,
                residual_hidden,
                prefix_feat_cond,
                base_k_cache,
                base_v_cache,
                base_current_length,
                residual_k_cache,
                residual_v_cache,
                residual_current_length,
                diffusion_noise[chunk_index],
                cfg_value,
            )
            (
                pred_audio_feature,
                decoder_latent,
                stop_logits,
                lm_hidden,
                residual_hidden,
                prefix_feat_cond,
                base_k_update,
                base_v_update,
                next_base_current_length,
                residual_k_update,
                residual_v_update,
                next_residual_current_length,
            ) = step_outputs

            pred_audio_features.append(pred_audio_feature)
            decoder_latents.append(decoder_latent)
            stop_logits_values.append(stop_logits)
            base_k_updates.append(base_k_update)
            base_v_updates.append(base_v_update)
            residual_k_updates.append(residual_k_update)
            residual_v_updates.append(residual_v_update)

            if chunk_index + 1 < self.chunk_size:
                base_k_cache = self._write_cache_position(base_k_cache, base_k_update, base_current_length)
                base_v_cache = self._write_cache_position(base_v_cache, base_v_update, base_current_length)
                residual_k_cache = self._write_cache_position(
                    residual_k_cache, residual_k_update, residual_current_length
                )
                residual_v_cache = self._write_cache_position(
                    residual_v_cache, residual_v_update, residual_current_length
                )
            base_current_length = next_base_current_length
            residual_current_length = next_residual_current_length

        return (
            cast_tensor_if_needed(torch.cat(pred_audio_features, dim=1), self.host_float_dtype),
            cast_tensor_if_needed(torch.cat(decoder_latents, dim=2), self.host_float_dtype),
            cast_tensor_if_needed(torch.stack(stop_logits_values, dim=1), self.host_float_dtype),
            cast_tensor_if_needed(lm_hidden, self.host_float_dtype),
            cast_tensor_if_needed(residual_hidden, self.host_float_dtype),
            cast_tensor_if_needed(prefix_feat_cond, self.host_float_dtype),
            cast_tensor_if_needed(torch.cat(base_k_updates, dim=3), self.host_float_dtype),
            cast_tensor_if_needed(torch.cat(base_v_updates, dim=3), self.host_float_dtype),
            base_current_length,
            cast_tensor_if_needed(torch.cat(residual_k_updates, dim=3), self.host_float_dtype),
            cast_tensor_if_needed(torch.cat(residual_v_updates, dim=3), self.host_float_dtype),
            residual_current_length,
        )


def make_synthetic_decode_chunk_inputs(
    *,
    chunk_size: int,
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
    inputs = make_synthetic_decode_step_inputs(
        batch_size=batch_size,
        current_length=current_length,
        max_cache_seq=max_cache_seq,
        hidden_size=hidden_size,
        patch_size=patch_size,
        feat_dim=feat_dim,
        base_layers=base_layers,
        residual_layers=residual_layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        cfg_value=cfg_value,
        seed=seed,
    )
    if max_cache_seq < current_length + chunk_size:
        raise ValueError("--max-cache-seq must be at least --current-length + --chunk-size")
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    inputs["diffusion_noise"] = torch.randn(
        chunk_size,
        batch_size,
        feat_dim,
        patch_size,
        dtype=torch.float32,
        generator=generator,
    )
    return inputs


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
    chunk_size: int,
    inference_timesteps: int,
) -> dict[str, Any]:
    dims = _model_dims(model)
    return {
        "chunk_size": chunk_size,
        "inference_timesteps": inference_timesteps,
        "inputs": {
            "lm_hidden": ["static:batch", f"static:{dims['hidden_size']}"],
            "residual_hidden": ["static:batch", f"static:{dims['hidden_size']}"],
            "prefix_feat_cond": ["static:batch", f"static:{dims['patch_size']}", f"static:{dims['feat_dim']}"],
            "base_k_cache": [
                f"static:{dims['base_layers']}",
                "static:batch",
                f"static:{dims['kv_heads']}",
                "dynamic:max_cache_seq",
                f"static:{dims['head_dim']}",
            ],
            "base_v_cache": [
                f"static:{dims['base_layers']}",
                "static:batch",
                f"static:{dims['kv_heads']}",
                "dynamic:max_cache_seq",
                f"static:{dims['head_dim']}",
            ],
            "base_current_length": ["static:1"],
            "residual_k_cache": [
                f"static:{dims['residual_layers']}",
                "static:batch",
                f"static:{dims['kv_heads']}",
                "dynamic:max_cache_seq",
                f"static:{dims['head_dim']}",
            ],
            "residual_v_cache": [
                f"static:{dims['residual_layers']}",
                "static:batch",
                f"static:{dims['kv_heads']}",
                "dynamic:max_cache_seq",
                f"static:{dims['head_dim']}",
            ],
            "residual_current_length": ["static:1"],
            "diffusion_noise": [
                f"static:{chunk_size}",
                f"static:{batch_size}",
                f"static:{dims['feat_dim']}",
                f"static:{dims['patch_size']}",
            ],
            "cfg_value": ["static:1"],
        },
        "outputs": {
            "pred_audio_feature": [
                f"static:{batch_size}",
                f"static:{chunk_size}",
                f"static:{dims['patch_size']}",
                f"static:{dims['feat_dim']}",
            ],
            "decoder_latent": [
                f"static:{batch_size}",
                f"static:{dims['feat_dim']}",
                f"static:{chunk_size * dims['patch_size']}",
            ],
            "stop_logits": [f"static:{batch_size}", f"static:{chunk_size}", "static:2"],
            "base_k_update": [
                f"static:{dims['base_layers']}",
                f"static:{batch_size}",
                f"static:{dims['kv_heads']}",
                f"static:{chunk_size}",
                f"static:{dims['head_dim']}",
            ],
            "residual_k_update": [
                f"static:{dims['residual_layers']}",
                f"static:{batch_size}",
                f"static:{dims['kv_heads']}",
                f"static:{chunk_size}",
                f"static:{dims['head_dim']}",
            ],
        },
        "example": {
            "batch_size": batch_size,
            "current_length": current_length,
            "max_cache_seq": max_cache_seq,
        },
    }


def export_decode_chunk(args: argparse.Namespace) -> None:
    _resolve_model_path, load_voxcpm2_prefill_model = _export_helpers()
    precision = get_precision_profile(args.precision)
    model_dir = _resolve_model_path(args.model_path, args.local_files_only)
    output_path = resolve_output_path(args.output, MODULE_KEY, precision)
    ensure_output_dir(output_path)

    model = load_voxcpm2_prefill_model(model_dir, precision)
    wrapper = VoxCPM2DecodeChunkWrapper(
        model,
        chunk_size=args.chunk_size,
        inference_timesteps=args.inference_timesteps,
        precision=precision,
    ).eval()
    dims = _model_dims(model)
    inputs = make_synthetic_decode_chunk_inputs(
        batch_size=args.batch_size,
        current_length=args.current_length,
        max_cache_seq=args.max_cache_seq,
        chunk_size=args.chunk_size,
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
            args.chunk_size,
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
        description="Export a production VoxCPM2 chunked autoregressive decode graph to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/export/export_decode_chunk.py "
            "--precision fp32 --chunk-size 4 --current-length 16 --max-cache-seq 64"
        ),
    )
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ONNX output path. Defaults to models/onnx/<precision>/decode_chunk/voxcpm2_decode_chunk.onnx.",
    )
    add_precision_argument(parser)
    parser.add_argument("--batch-size", type=int, default=1, help="Example batch dimension used during export.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Decode steps per ONNX session.run.")
    parser.add_argument(
        "--current-length", type=int, default=16, help="Example valid KV-cache length entering the chunk."
    )
    parser.add_argument(
        "--max-cache-seq",
        type=int,
        default=64,
        help="Example fixed KV-cache capacity; must be at least --current-length + --chunk-size.",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Fixed CFM/LocDiT solver steps embedded in each internal decode step.",
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
    export_decode_chunk(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
