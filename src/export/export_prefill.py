#!/usr/bin/env python3
"""Export the VoxCPM2 prefill path to ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.export.export_audio_vae_decoder import _resolve_model_path


INPUT_NAMES = ["text_tokens", "text_mask", "audio_features", "audio_mask"]
OUTPUT_NAMES = [
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

PrefillMode = Literal["plain_tts", "voice_design", "controllable_clone", "ultimate_clone"]


class VoxCPM2PrefillWrapper(torch.nn.Module):
    """ONNX-facing wrapper for VoxCPM2 prefill neural work.

    Host code owns tokenization, multilingual text handling, reference/prompt
    sequence assembly, and autoregressive orchestration. This wrapper mirrors
    the non-iterative neural section at the start of ``VoxCPM2Model._inference``.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _stack_cache(cache_tuple: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        k_cache = torch.stack([layer_cache[0] for layer_cache in cache_tuple], dim=0)
        v_cache = torch.stack([layer_cache[1] for layer_cache in cache_tuple], dim=0)
        return k_cache, v_cache

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_features: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        model = self.model
        # The export boundary is intentionally tensor-only. Host code has
        # already handled tokenization, language text, reference/prompt audio
        # alignment, and masks before this wrapper is called.
        text_tokens = text_tokens.to(dtype=torch.long)
        text_mask = text_mask.to(dtype=torch.float32)
        audio_features = audio_features.to(dtype=torch.float32)
        audio_mask = audio_mask.to(dtype=torch.float32)

        prefill_encoder = getattr(model, "_feat_encoder_raw", model.feat_encoder)
        feat_embed = prefill_encoder(audio_features)
        feat_embed = model.enc_to_lm_proj(feat_embed)

        if model.config.lm_config.use_mup:
            scale_emb = model.config.lm_config.scale_emb
        else:
            scale_emb = 1.0
        text_embed = model.base_lm.embed_tokens(text_tokens) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed

        prefix_feat_cond = audio_features[:, -1, :, :]

        enc_outputs, base_cache_tuple = model.base_lm(inputs_embeds=combined_embed, is_causal=True)
        enc_outputs = model.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

        residual_enc_inputs = model.fusion_concat_proj(
            torch.cat((enc_outputs, audio_mask.unsqueeze(-1) * feat_embed), dim=-1)
        )
        residual_enc_outputs, residual_cache_tuple = model.residual_lm(
            inputs_embeds=residual_enc_inputs,
            is_causal=True,
        )
        residual_hidden = residual_enc_outputs[:, -1, :]

        base_k_cache, base_v_cache = self._stack_cache(base_cache_tuple)
        residual_k_cache, residual_v_cache = self._stack_cache(residual_cache_tuple)

        # MiniCPM returns Python cache tuples. ONNX Runtime cannot pass those
        # through a session boundary, so the wrapper exposes named tensor caches
        # plus explicit valid lengths for the following decode_step graph.
        cache_length = torch.arange(text_tokens.shape[1], device=text_tokens.device, dtype=torch.long)[-1:] + 1
        residual_cache_length = cache_length.clone()

        return (
            lm_hidden,
            residual_hidden,
            prefix_feat_cond,
            base_k_cache,
            base_v_cache,
            cache_length,
            residual_k_cache,
            residual_v_cache,
            residual_cache_length,
        )


def _install_upstream_import_path() -> None:
    upstream_src = REPO_ROOT / "third_party" / "VoxCPM" / "src"
    if not upstream_src.exists():
        raise FileNotFoundError(f"missing upstream VoxCPM source: {upstream_src}")
    sys.path.insert(0, str(upstream_src))


def load_voxcpm2_prefill_model(model_dir: Path) -> torch.nn.Module:
    _install_upstream_import_path()

    from voxcpm.model.voxcpm2 import VoxCPM2Model

    model = VoxCPM2Model.from_local(str(model_dir), optimize=False, device="cpu")
    model = model.to(device="cpu", dtype=torch.float32).eval()
    model.config.dtype = "float32"
    print(f"loaded_voxcpm2={model_dir}")
    print(f"device=cpu dtype=float32")
    print(f"patch_size={model.patch_size} feat_dim={model.feat_dim}")
    print(
        "lm_config="
        + json.dumps(
            {
                "hidden_size": model.config.lm_config.hidden_size,
                "base_layers": model.config.lm_config.num_hidden_layers,
                "residual_layers": model.config.residual_lm_num_layers,
                "num_attention_heads": model.config.lm_config.num_attention_heads,
                "num_key_value_heads": model.config.lm_config.num_key_value_heads,
                "kv_channels": model.config.lm_config.kv_channels,
                "vocab_size": model.config.lm_config.vocab_size,
                "use_mup": model.config.lm_config.use_mup,
            },
            sort_keys=True,
        )
    )
    return model


def make_synthetic_prefill_inputs(
    *,
    batch_size: int,
    seq_len: int,
    patch_size: int,
    feat_dim: int,
    vocab_size: int,
    mode: PrefillMode,
    seed: int,
    reference_steps: int,
    prompt_steps: int,
) -> dict[str, torch.Tensor]:
    if seq_len < 1:
        raise ValueError("--seq-len must be >= 1")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    text_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, generator=generator)
    text_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    audio_mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)
    audio_features = torch.zeros(batch_size, seq_len, patch_size, feat_dim, dtype=torch.float32)

    def fill_audio_span(start: int, length: int) -> None:
        if length <= 0 or start >= seq_len:
            return
        end = min(seq_len, start + length)
        if end <= start:
            return
        text_tokens[:, start:end] = 0
        text_mask[:, start:end] = 0.0
        audio_mask[:, start:end] = 1.0
        audio_features[:, start:end, :, :] = torch.randn(
            batch_size,
            end - start,
            patch_size,
            feat_dim,
            dtype=torch.float32,
            generator=generator,
        )

    if mode in ("plain_tts", "voice_design"):
        pass
    elif mode == "controllable_clone":
        ref_len = min(reference_steps, max(seq_len - 2, 0))
        fill_audio_span(1, ref_len)
    elif mode == "ultimate_clone":
        ref_len = min(reference_steps, max(seq_len - prompt_steps - 3, 0))
        fill_audio_span(1, ref_len)
        prompt_len = min(prompt_steps, max(seq_len - (ref_len + 3), 0))
        fill_audio_span(seq_len - prompt_len, prompt_len)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    return {
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "audio_features": audio_features,
        "audio_mask": audio_mask,
    }


def _dynamic_shapes() -> dict[str, dict[int, Any]]:
    batch = torch.export.Dim("batch", min=1)
    seq = torch.export.Dim("seq", min=1)
    return {
        "text_tokens": {0: batch, 1: seq},
        "text_mask": {0: batch, 1: seq},
        "audio_features": {0: batch, 1: seq},
        "audio_mask": {0: batch, 1: seq},
    }


def _shape_report(model: torch.nn.Module, batch_size: int, seq_len: int, mode: str) -> dict[str, Any]:
    lm_config = model.config.lm_config
    head_dim = lm_config.kv_channels or (lm_config.hidden_size // lm_config.num_attention_heads)
    return {
        "mode": mode,
        "inputs": {
            "text_tokens": {
                "dtype": "int64",
                "dims": ["dynamic:batch", "dynamic:seq"],
                "example_shape": [batch_size, seq_len],
            },
            "text_mask": {
                "dtype": "float32",
                "dims": ["dynamic:batch", "dynamic:seq"],
                "example_shape": [batch_size, seq_len],
            },
            "audio_features": {
                "dtype": "float32",
                "dims": ["dynamic:batch", "dynamic:seq", f"static:{model.patch_size}", f"static:{model.feat_dim}"],
                "example_shape": [batch_size, seq_len, model.patch_size, model.feat_dim],
            },
            "audio_mask": {
                "dtype": "float32",
                "dims": ["dynamic:batch", "dynamic:seq"],
                "example_shape": [batch_size, seq_len],
            },
        },
        "outputs": {
            "lm_hidden": ["dynamic:batch", f"static:{lm_config.hidden_size}"],
            "residual_hidden": ["dynamic:batch", f"static:{lm_config.hidden_size}"],
            "prefix_feat_cond": ["dynamic:batch", f"static:{model.patch_size}", f"static:{model.feat_dim}"],
            "base_k_cache": [
                f"static:{lm_config.num_hidden_layers}",
                "dynamic:batch",
                f"static:{lm_config.num_key_value_heads}",
                "dynamic:seq",
                f"static:{head_dim}",
            ],
            "base_v_cache": [
                f"static:{lm_config.num_hidden_layers}",
                "dynamic:batch",
                f"static:{lm_config.num_key_value_heads}",
                "dynamic:seq",
                f"static:{head_dim}",
            ],
            "base_cache_length": ["static:1"],
            "residual_k_cache": [
                f"static:{model.config.residual_lm_num_layers}",
                "dynamic:batch",
                f"static:{lm_config.num_key_value_heads}",
                "dynamic:seq",
                f"static:{head_dim}",
            ],
            "residual_v_cache": [
                f"static:{model.config.residual_lm_num_layers}",
                "dynamic:batch",
                f"static:{lm_config.num_key_value_heads}",
                "dynamic:seq",
                f"static:{head_dim}",
            ],
            "residual_cache_length": ["static:1"],
        },
    }


def export_prefill(args: argparse.Namespace) -> None:
    model_dir = _resolve_model_path(args.model_path, args.local_files_only)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_voxcpm2_prefill_model(model_dir)
    wrapper = VoxCPM2PrefillWrapper(model).eval()
    inputs = make_synthetic_prefill_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        patch_size=model.patch_size,
        feat_dim=model.feat_dim,
        vocab_size=model.config.lm_config.vocab_size,
        mode=args.mode,
        seed=args.seed,
        reference_steps=args.reference_steps,
        prompt_steps=args.prompt_steps,
    )

    report = _shape_report(model, args.batch_size, args.seq_len, args.mode)
    print("input_names=" + ",".join(INPUT_NAMES))
    print("output_names=" + ",".join(OUTPUT_NAMES))
    print("shape_report=" + json.dumps(report, sort_keys=True))
    print(f"output_path={output_path}")

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            args=tuple(inputs[name] for name in INPUT_NAMES),
            f=str(output_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=args.opset,
            dynamo=True,
            external_data=True,
            dynamic_shapes=_dynamic_shapes(),
            optimize=False,
            do_constant_folding=False,
            verify=False,
        )

    print(f"exported={output_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the VoxCPM2 prefill neural boundary to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/export/export_prefill.py "
            "--output artifacts/prefill/voxcpm2_prefill.onnx --mode plain_tts"
        ),
    )
    parser.add_argument("--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/prefill/voxcpm2_prefill.onnx"), help="ONNX output path.")
    parser.add_argument("--batch-size", type=int, default=1, help="Example batch dimension used during export.")
    parser.add_argument("--seq-len", type=int, default=16, help="Example full prompt sequence length used during export.")
    parser.add_argument(
        "--mode",
        choices=["plain_tts", "voice_design", "controllable_clone", "ultimate_clone"],
        default="plain_tts",
        help="Synthetic input layout used to exercise the prefill boundary during export.",
    )
    parser.add_argument("--reference-steps", type=int, default=3, help="Synthetic reference-audio feature steps for clone modes.")
    parser.add_argument("--prompt-steps", type=int, default=3, help="Synthetic prompt-audio feature steps for ultimate_clone.")
    parser.add_argument("--seed", type=int, default=0, help="PyTorch RNG seed for synthetic export inputs.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version for torch.onnx.export.")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Require local Hugging Face cache/model files.")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only", help="Allow snapshot_download to fetch missing files.")
    return parser


def main() -> int:
    export_prefill(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
