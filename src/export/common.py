"""Shared ONNX export contracts and precision-profile helpers"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PrecisionName = Literal["fp32", "bf16"]

REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_ROOT = REPO_ROOT / "models" / "onnx"
PRECISION_CHOICES: tuple[PrecisionName, ...] = ("fp32", "bf16")


@dataclass(frozen=True)
class PrecisionProfile:
    """Export-time precision policy.

    Host-visible tensors stay FP32 for both profiles so one CPU-only runtime can
    load either artifact family. BF16 changes the model compute dtype inside the
    export wrapper and records boundary casts as intentional mixed precision.
    """

    name: PrecisionName
    compute_dtype: str
    host_float_dtype: str
    model_config_dtype: str
    production_compute: bool
    storage_only: bool
    description: str

    def torch_compute_dtype(self):
        import torch

        return getattr(torch, self.compute_dtype)

    def torch_host_float_dtype(self):
        import torch

        return getattr(torch, self.host_float_dtype)


PRECISION_PROFILES: dict[PrecisionName, PrecisionProfile] = {
    "fp32": PrecisionProfile(
        name="fp32",
        compute_dtype="float32",
        host_float_dtype="float32",
        model_config_dtype="float32",
        production_compute=True,
        storage_only=False,
        description="Production correctness anchor with FP32 model compute and FP32 public tensor contract.",
    ),
    "bf16": PrecisionProfile(
        name="bf16",
        compute_dtype="bfloat16",
        host_float_dtype="float32",
        model_config_dtype="bfloat16",
        production_compute=True,
        storage_only=False,
        description=(
            "Production BF16 compute target with BF16 weights/activations where feasible and the same runtime path."
        ),
    ),
}


@dataclass(frozen=True)
class ModulePrecisionPolicy:
    """BF16 compute regions and intentional FP32 islands for one module"""

    bf16_compute_regions: tuple[str, ...]
    fp32_islands: tuple[str, ...]
    boundary_casts: tuple[str, ...]


BF16_MODULE_POLICIES: dict[str, ModulePrecisionPolicy] = {
    "audio_vae_encoder": ModulePrecisionPolicy(
        bf16_compute_regions=("AudioVAE encoder convolution/residual stack",),
        fp32_islands=(),
        boundary_casts=("waveform fp32->bf16", "latent bf16->fp32"),
    ),
    "audio_vae_decoder": ModulePrecisionPolicy(
        bf16_compute_regions=("AudioVAE decoder convolution/residual stack",),
        fp32_islands=(),
        boundary_casts=("latent fp32->bf16", "waveform bf16->fp32"),
    ),
    "prefill": ModulePrecisionPolicy(
        bf16_compute_regions=(
            "feature encoder",
            "text embeddings",
            "base LM prefill",
            "FSQ/fusion projection",
            "residual LM prefill",
        ),
        fp32_islands=(),
        boundary_casts=(
            "text/audio masks fp32->bf16",
            "audio features fp32->bf16",
            "hidden/cache outputs bf16->fp32",
        ),
    ),
    "decode_step": ModulePrecisionPolicy(
        bf16_compute_regions=(
            "DiT conditioning projections",
            "LocDiT/CFM solve",
            "feature encoder",
            "base LM decode step",
            "residual LM decode step",
            "stop head",
        ),
        fp32_islands=("rotary position embedding multiply/add",),
        boundary_casts=(
            "hidden/cache/noise/cfg fp32->bf16",
            "feature/hidden/cache-update outputs bf16->fp32",
        ),
    ),
    "decode_chunk": ModulePrecisionPolicy(
        bf16_compute_regions=(
            "DiT conditioning projections",
            "LocDiT/CFM solve",
            "feature encoder",
            "base LM decode steps",
            "residual LM decode steps",
            "stop head",
        ),
        fp32_islands=("rotary position embedding multiply/add",),
        boundary_casts=(
            "hidden/cache/noise/cfg fp32->bf16",
            "chunk feature/hidden/cache-update outputs bf16->fp32",
        ),
    ),
}


@dataclass(frozen=True)
class ModuleOutputLayout:
    """Default artifact location for one exported module"""

    directory: str
    filename: str


MODULE_OUTPUT_LAYOUTS: dict[str, ModuleOutputLayout] = {
    "audio_vae_encoder": ModuleOutputLayout("audio_vae_encoder", "audio_vae_encoder.onnx"),
    "audio_vae_decoder": ModuleOutputLayout("audio_vae_decoder", "audio_vae_decoder.onnx"),
    "prefill": ModuleOutputLayout("prefill", "voxcpm2_prefill.onnx"),
    "decode_step": ModuleOutputLayout("decode_step", "voxcpm2_decode_step.onnx"),
    "decode_chunk": ModuleOutputLayout("decode_chunk", "voxcpm2_decode_chunk.onnx"),
}


@dataclass(frozen=True)
class ModuleExportContract:
    """Stable public graph contract shared by FP32 and BF16 artifacts"""

    module_key: str
    display_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    state_semantics: str


MODULE_EXPORT_CONTRACTS: dict[str, ModuleExportContract] = {
    "audio_vae_encoder": ModuleExportContract(
        module_key="audio_vae_encoder",
        display_name="AudioVAEEncoder",
        input_names=("waveform",),
        output_names=("latent",),
        state_semantics="stateless",
    ),
    "audio_vae_decoder": ModuleExportContract(
        module_key="audio_vae_decoder",
        display_name="AudioVAEDecoder",
        input_names=("latent", "sr_cond"),
        output_names=("waveform",),
        state_semantics="stateless",
    ),
    "prefill": ModuleExportContract(
        module_key="prefill",
        display_name="VoxCPM2Prefill",
        input_names=("text_tokens", "text_mask", "audio_features", "audio_mask"),
        output_names=(
            "lm_hidden",
            "residual_hidden",
            "prefix_feat_cond",
            "base_k_cache",
            "base_v_cache",
            "base_cache_length",
            "residual_k_cache",
            "residual_v_cache",
            "residual_cache_length",
        ),
        state_semantics="creates initial hidden states and explicit KV-cache tensors",
    ),
    "decode_step": ModuleExportContract(
        module_key="decode_step",
        display_name="VoxCPM2DecodeStep",
        input_names=(
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
        ),
        output_names=(
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
        ),
        state_semantics="consumes fixed-capacity caches and returns one-position cache updates plus new lengths",
    ),
    "decode_chunk": ModuleExportContract(
        module_key="decode_chunk",
        display_name="VoxCPM2DecodeChunk",
        input_names=(
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
        ),
        output_names=(
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
        ),
        state_semantics=(
            "consumes fixed-capacity caches and returns chunked feature, stop-logit, "
            "and cache-update tensors plus final lengths"
        ),
    ),
}


def add_precision_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--precision",
        choices=PRECISION_CHOICES,
        default="fp32",
        help="Production precision profile. BF16 keeps the same public graph contract and changes compute dtype.",
    )


def get_precision_profile(name: str) -> PrecisionProfile:
    if name not in PRECISION_PROFILES:
        choices = ", ".join(PRECISION_CHOICES)
        raise ValueError(f"unsupported precision profile {name!r}; expected one of: {choices}")
    return PRECISION_PROFILES[name]  # type: ignore[index]


def default_output_path(module_key: str, precision: PrecisionProfile) -> Path:
    layout = MODULE_OUTPUT_LAYOUTS[module_key]
    return ONNX_ROOT / precision.name / layout.directory / layout.filename


def output_path_under_root(output_root: Path, module_key: str, precision: PrecisionProfile) -> Path:
    layout = MODULE_OUTPUT_LAYOUTS[module_key]
    return output_root.expanduser() / precision.name / layout.directory / layout.filename


def resolve_output_path(output: Path | None, module_key: str, precision: PrecisionProfile) -> Path:
    if output is not None:
        return output.expanduser()
    return default_output_path(module_key, precision)


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def configure_module_precision(module: Any, precision: PrecisionProfile) -> Any:
    if precision.storage_only:
        raise ValueError("storage-only precision profiles are not valid production export profiles")
    return module.to(device="cpu", dtype=precision.torch_compute_dtype()).eval()


def cast_tensor_if_needed(tensor: Any, dtype: Any) -> Any:
    """Return ``tensor`` unchanged when it already has ``dtype``.

    Export wrappers use this instead of unconditional ``Tensor.to(dtype=...)``
    so FP32 exports do not pick up no-op Cast nodes. BF16 exports still emit the
    intentional graph-boundary and FP32-island casts required by the precision
    policy.
    """

    if getattr(tensor, "dtype", None) == dtype:
        return tensor
    return tensor.to(dtype=dtype)


def add_precision_metadata(
    report: dict[str, Any],
    precision: PrecisionProfile,
    *,
    module_key: str | None = None,
) -> dict[str, Any]:
    enriched = deepcopy(report)
    enriched["precision_profile"] = {
        "name": precision.name,
        "compute_dtype": precision.compute_dtype,
        "host_float_dtype": precision.host_float_dtype,
        "model_config_dtype": precision.model_config_dtype,
        "production_compute": precision.production_compute,
        "storage_only": precision.storage_only,
        "boundary_policy": "host_float32_contract_with_profile_compute_dtype",
    }
    if module_key is not None and precision.name == "bf16":
        module_policy = BF16_MODULE_POLICIES[module_key]
        enriched["bf16_compute_policy"] = {
            "bf16_compute_regions": list(module_policy.bf16_compute_regions),
            "fp32_islands": list(module_policy.fp32_islands),
            "boundary_casts": list(module_policy.boundary_casts),
            "forbidden_pattern": "BF16 initializer immediately cast back to FLOAT as the primary compute path",
        }
    return enriched


def print_export_plan(
    *,
    module_key: str,
    precision: PrecisionProfile,
    input_names: list[str],
    output_names: list[str],
    shape_report: dict[str, Any],
    output_path: Path,
) -> None:
    print(f"module={MODULE_EXPORT_CONTRACTS[module_key].display_name}")
    print(f"precision={precision.name}")
    print(f"compute_dtype={precision.compute_dtype}")
    print(f"host_float_dtype={precision.host_float_dtype}")
    print("input_names=" + ",".join(input_names))
    print("output_names=" + ",".join(output_names))
    print(
        "shape_report="
        + json.dumps(add_precision_metadata(shape_report, precision, module_key=module_key), sort_keys=True)
    )
    print(f"output_path={output_path}")


def export_onnx_graph(
    *,
    wrapper: Any,
    inputs: tuple[Any, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    dynamic_shapes: dict[str, dict[int, Any]],
) -> None:
    import torch

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            args=inputs,
            f=str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamo=True,
            external_data=True,
            dynamic_shapes=dynamic_shapes,
            optimize=False,
            do_constant_folding=False,
            verify=False,
        )
