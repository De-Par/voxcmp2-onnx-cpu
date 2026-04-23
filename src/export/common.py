"""Shared ONNX export contracts and precision-profile helpers"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

PrecisionName = Literal["fp32", "bf16"]
ShapeProfileName = Literal["production", "flex"]

REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_ROOT = REPO_ROOT / "models" / "onnx"
PRECISION_CHOICES: tuple[PrecisionName, ...] = ("fp32", "bf16")
SHAPE_PROFILE_CHOICES: tuple[ShapeProfileName, ...] = ("production", "flex")
BF16_ORT_CPU_FP32_ISLAND_OPS: tuple[str, ...] = (
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Sqrt",
    "Reciprocal",
    "Exp",
    "Tanh",
    "Sigmoid",
    "Softmax",
    "ReduceMean",
    "Expand",
    "Pad",
    "Conv",
    "ConvTranspose",
    "MatMul",
    "Gemm",
    "Where",
    "IsNaN",
    "SplitToSequence",
    "Cos",
    "Sin",
    "Round",
)
BF16_ORT_CPU_FLOAT_INPUT_INDEXES: dict[str, tuple[int, ...] | None] = {
    # ReduceMean input 1 is axes in opset 18 and must remain INT64.
    "ReduceMean": (0,),
    # Pad input 1 is pads and must remain INT64; input 2 is optional data-type constant value.
    "Pad": (0, 2),
    # Expand input 1 is the output shape and must remain INT64.
    "Expand": (0,),
    # SplitToSequence input 1 is split sizes and must remain INT64.
    "SplitToSequence": (0,),
    # Where input 0 is the BOOL condition; only the value branches are floating tensors.
    "Where": (1, 2),
}
BF16_ORT_CPU_SKIP_OUTPUT_CAST_OPS: set[str] = {"IsNaN", "SplitToSequence"}
BF16_ORT_CPU_PROPAGATE_FLOAT_FROM_INPUT0_OPS: set[str] = {
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Flatten",
}


@dataclass(frozen=True)
class ShapeProfile:
    """Export-time shape policy shared by FP32 and BF16 artifacts"""

    name: ShapeProfileName
    static_batch: bool
    batch_size: int
    max_audio_samples: int | None
    max_decoder_latent_steps: int | None
    max_prefill_seq: int | None
    max_decode_cache_seq: int | None
    description: str


SHAPE_PROFILES: dict[ShapeProfileName, ShapeProfile] = {
    "production": ShapeProfile(
        name="production",
        static_batch=True,
        batch_size=1,
        max_audio_samples=960_000,
        max_decoder_latent_steps=16_384,
        max_prefill_seq=1_024,
        max_decode_cache_seq=6_144,
        description=(
            "Production CPU runtime profile: batch=1 with bounded prompt/audio/decode dimensions. "
            "The same profile is used for FP32 and BF16 exports."
        ),
    ),
    "flex": ShapeProfile(
        name="flex",
        static_batch=False,
        batch_size=1,
        max_audio_samples=None,
        max_decoder_latent_steps=None,
        max_prefill_seq=None,
        max_decode_cache_seq=None,
        description="Internal/debug profile preserving the previous broadly dynamic export shapes.",
    ),
}


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
        bf16_compute_regions=("AudioVAE encoder non-convolution activation/mixing regions",),
        fp32_islands=("Pad", "Conv", "Sin", "elementwise ops required by ORT CPU kernel coverage"),
        boundary_casts=("waveform fp32->bf16", "latent bf16->fp32", "BF16<->FP32 around ORT CPU islands"),
    ),
    "audio_vae_decoder": ModulePrecisionPolicy(
        bf16_compute_regions=("AudioVAE decoder non-convolution activation/mixing regions",),
        fp32_islands=("Pad", "ConvTranspose", "Conv", "Sin", "elementwise ops required by ORT CPU kernel coverage"),
        boundary_casts=("latent fp32->bf16", "waveform bf16->fp32", "BF16<->FP32 around ORT CPU islands"),
    ),
    "prefill": ModulePrecisionPolicy(
        bf16_compute_regions=(
            "feature encoder",
            "text embeddings",
            "base LM prefill",
            "FSQ/fusion projection",
            "residual LM prefill",
        ),
        fp32_islands=("MatMul, Gemm, Expand, Round, Sigmoid, and elementwise ops required by ORT CPU kernel coverage",),
        boundary_casts=(
            "text/audio masks fp32->bf16",
            "audio features fp32->bf16",
            "hidden/cache outputs bf16->fp32",
            "Round BF16<->FP32 ORT CPU island",
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
        fp32_islands=(
            "rotary position embedding multiply/add",
            "MatMul/Gemm/Where/IsNaN/Expand/elementwise/Cos/Sin/Round ORT CPU islands",
        ),
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
        fp32_islands=(
            "rotary position embedding multiply/add",
            "MatMul/Gemm/Where/IsNaN/Expand/elementwise/Cos/Sin/Round ORT CPU islands",
        ),
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


def add_shape_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shape-profile",
        choices=SHAPE_PROFILE_CHOICES,
        default="production",
        help=(
            "Export shape policy. production specializes batch=1 and bounded dimensions; "
            "flex keeps broader dynamic shapes for internal debugging."
        ),
    )


def get_shape_profile(name: str) -> ShapeProfile:
    if name not in SHAPE_PROFILES:
        choices = ", ".join(SHAPE_PROFILE_CHOICES)
        raise ValueError(f"unsupported shape profile {name!r}; expected one of: {choices}")
    return SHAPE_PROFILES[name]  # type: ignore[index]


def resolve_shape_profile(args: argparse.Namespace) -> ShapeProfile:
    profile = get_shape_profile(args.shape_profile)
    overrides = {
        "max_audio_samples": getattr(args, "max_samples", None),
        "max_decoder_latent_steps": getattr(args, "max_latent_steps", None),
        "max_prefill_seq": getattr(args, "max_seq_len", None),
        "max_decode_cache_seq": getattr(args, "max_cache_seq_bound", None),
    }
    return replace(profile, **{key: value for key, value in overrides.items() if value is not None})


def bounded_dim(name: str, *, minimum: int = 1, maximum: int | None = None) -> Any:
    import torch

    if maximum is None:
        return torch.export.Dim(name, min=minimum)
    return torch.export.Dim(name, min=minimum, max=maximum)


def validate_static_batch(batch_size: int, shape_profile: ShapeProfile) -> None:
    if shape_profile.static_batch and batch_size != shape_profile.batch_size:
        raise ValueError(
            f"shape profile {shape_profile.name!r} requires --batch-size {shape_profile.batch_size}; "
            "use --shape-profile flex for non-production batch experiments"
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


def _tensor_type_map(model: Any) -> dict[str, int]:
    type_map: dict[str, int] = {}
    graph = model.graph
    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not value_info.type.HasField("tensor_type"):
            continue
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type:
            type_map[value_info.name] = tensor_type.elem_type
    for initializer in graph.initializer:
        type_map[initializer.name] = initializer.data_type
    for sparse_initializer in graph.sparse_initializer:
        type_map[sparse_initializer.values.name] = sparse_initializer.values.data_type
    return type_map


def _inferred_tensor_type_map(model: Any) -> dict[str, int]:
    import onnx

    type_map = _tensor_type_map(model)
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    except Exception:
        return type_map
    inferred_map = _tensor_type_map(inferred)
    type_map.update(inferred_map)
    return type_map


def _set_value_info_elem_type(graph: Any, name: str, elem_type: int) -> None:
    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        if value_info.name == name and value_info.type.HasField("tensor_type"):
            value_info.type.tensor_type.elem_type = elem_type


def apply_bf16_ort_cpu_compatibility_pass(
    model_path: Path,
    *,
    op_types: tuple[str, ...] = BF16_ORT_CPU_FP32_ISLAND_OPS,
    check_model: bool = True,
) -> dict[str, Any]:
    """Insert FP32 islands around BF16 ops unsupported by ORT CPU.

    ONNX checker accepts BF16 graphs that ONNX Runtime CPU later rejects during
    session creation. The production BF16 path keeps BF16 where ORT CPU can run
    it and casts only specific unsupported operators to FP32, then casts their
    BF16-typed outputs back so the surrounding graph contract does not change.
    This edits graph edges only; it does not alter public input/output names,
    shapes, or state semantics.
    """

    import onnx
    from onnx import TensorProto, helper

    model_path = model_path.expanduser()
    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph
    type_map = _inferred_tensor_type_map(model)
    island_ops = set(op_types)

    existing_cast_outputs = {node.output[0] for node in graph.node if node.op_type == "Cast" and node.output}
    graph_output_names = {output.name for output in graph.output}
    new_nodes = []
    inserted_input_casts = 0
    inserted_output_casts = 0
    restored_nonfloat_inputs = 0
    updated_sequence_at_outputs = 0
    updated_float_propagated_outputs = 0
    patched_nodes = 0
    patched_by_op: dict[str, int] = {}
    fp32_sequence_outputs = {
        node.output[0]
        for node in graph.node
        if node.op_type == "SplitToSequence" and node.input and node.output and node.input[0].endswith("__bf16_to_fp32")
    }

    for node in graph.node:
        if node.op_type == "SequenceAt" and node.input and node.input[0] in fp32_sequence_outputs:
            for output_name in node.output:
                _set_value_info_elem_type(graph, output_name, TensorProto.FLOAT)
                type_map[output_name] = TensorProto.FLOAT
                updated_sequence_at_outputs += 1
            new_nodes.append(node)
            continue

        if (
            node.op_type in BF16_ORT_CPU_PROPAGATE_FLOAT_FROM_INPUT0_OPS
            and node.input
            and type_map.get(node.input[0]) == TensorProto.FLOAT
        ):
            for output_name in node.output:
                _set_value_info_elem_type(graph, output_name, TensorProto.FLOAT)
                type_map[output_name] = TensorProto.FLOAT
                updated_float_propagated_outputs += 1

        if node.op_type not in island_ops:
            new_nodes.append(node)
            continue

        allowed_input_indexes = BF16_ORT_CPU_FLOAT_INPUT_INDEXES.get(node.op_type)
        replacement_inputs = list(node.input)
        node_name = node.name or node.op_type
        for index, input_name in enumerate(node.input):
            if allowed_input_indexes is None or index in allowed_input_indexes:
                continue
            suffix = f"__{node_name}_{index}__bf16_to_fp32"
            if input_name.endswith(suffix):
                replacement_inputs[index] = input_name[: -len(suffix)]
                restored_nonfloat_inputs += 1

        # Exporter shape/type metadata is incomplete for some Transpose/MatMul
        # initializer paths. Unknown inputs are cast only when at least one
        # eligible input is already known BF16; pure shape/index subgraphs stay
        # untouched.
        eligible_indexes = [
            index
            for index, input_name in enumerate(replacement_inputs)
            if input_name and (allowed_input_indexes is None or index in allowed_input_indexes)
        ]
        known_bf16_input = any(
            type_map.get(replacement_inputs[index]) == TensorProto.BFLOAT16 for index in eligible_indexes
        )
        bf16_input_indexes = []
        for index in eligible_indexes:
            input_name = replacement_inputs[index]
            if not input_name or input_name.endswith("__bf16_to_fp32"):
                continue
            input_type = type_map.get(input_name)
            if input_type == TensorProto.BFLOAT16 or (known_bf16_input and input_type not in (TensorProto.FLOAT,)):
                bf16_input_indexes.append(index)

        bf16_output_indexes = []
        if node.op_type not in BF16_ORT_CPU_SKIP_OUTPUT_CAST_OPS:
            for index, output_name in enumerate(node.output):
                if not output_name or output_name.endswith("__fp32"):
                    continue
                output_type = type_map.get(output_name)
                should_restore_bf16 = known_bf16_input and output_name not in graph_output_names
                if output_type == TensorProto.BFLOAT16 or (should_restore_bf16 and output_type != TensorProto.FLOAT):
                    bf16_output_indexes.append(index)

        if replacement_inputs != list(node.input):
            patched_nodes += 1
            patched_by_op[node.op_type] = patched_by_op.get(node.op_type, 0) + 1

        if not bf16_input_indexes and not bf16_output_indexes:
            if replacement_inputs != list(node.input):
                repaired_node = deepcopy(node)
                del repaired_node.input[:]
                repaired_node.input.extend(replacement_inputs)
                new_nodes.append(repaired_node)
            else:
                new_nodes.append(node)
            continue

        if replacement_inputs == list(node.input):
            patched_nodes += 1
            patched_by_op[node.op_type] = patched_by_op.get(node.op_type, 0) + 1
        for index in bf16_input_indexes:
            input_name = replacement_inputs[index]
            cast_output = f"{input_name}__{node_name}_{index}__bf16_to_fp32"
            if cast_output not in existing_cast_outputs:
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        [input_name],
                        [cast_output],
                        name=f"Cast_BF16ToFP32_{node_name}_{index}",
                        to=TensorProto.FLOAT,
                    )
                )
                existing_cast_outputs.add(cast_output)
                type_map[cast_output] = TensorProto.FLOAT
                inserted_input_casts += 1
            replacement_inputs[index] = cast_output

        replacement_outputs = list(node.output)
        output_casts = []
        for index in bf16_output_indexes:
            output_name = node.output[index]
            fp32_output = f"{output_name}__{node_name}_{index}__fp32"
            replacement_outputs[index] = fp32_output
            type_map[fp32_output] = TensorProto.FLOAT
            output_casts.append(
                helper.make_node(
                    "Cast",
                    [fp32_output],
                    [output_name],
                    name=f"Cast_FP32ToBF16_{node_name}_{index}",
                    to=TensorProto.BFLOAT16,
                )
            )
            _set_value_info_elem_type(graph, output_name, TensorProto.BFLOAT16)
            type_map[output_name] = TensorProto.BFLOAT16
            inserted_output_casts += 1

        patched_node = deepcopy(node)
        del patched_node.input[:]
        patched_node.input.extend(replacement_inputs)
        del patched_node.output[:]
        patched_node.output.extend(replacement_outputs)
        new_nodes.append(patched_node)
        if node.op_type == "SplitToSequence" and patched_node.output:
            fp32_sequence_outputs.add(patched_node.output[0])
        new_nodes.extend(output_casts)

    if patched_nodes or updated_sequence_at_outputs or updated_float_propagated_outputs:
        del graph.node[:]
        graph.node.extend(new_nodes)
        # Keep existing external-data initializers untouched; the pass only
        # rewrites graph edges and adds Cast nodes.
        onnx.save_model(model, str(model_path))
        if check_model:
            onnx.checker.check_model(str(model_path))

    return {
        "model_path": str(model_path),
        "patched_nodes": patched_nodes,
        "patched_by_op": patched_by_op,
        "inserted_input_casts": inserted_input_casts,
        "inserted_output_casts": inserted_output_casts,
        "restored_nonfloat_inputs": restored_nonfloat_inputs,
        "updated_sequence_at_outputs": updated_sequence_at_outputs,
        "updated_float_propagated_outputs": updated_float_propagated_outputs,
        "op_types": list(op_types),
    }


def finalize_exported_graph(output_path: Path, precision: PrecisionProfile) -> dict[str, Any] | None:
    """Apply production post-export fixes that are specific to one precision"""

    if precision.name != "bf16":
        return None
    report = apply_bf16_ort_cpu_compatibility_pass(output_path)
    print("bf16_ort_cpu_compatibility=" + json.dumps(report, sort_keys=True))
    return report


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
            "forbidden_pattern": "unscoped storage-only BF16 conversion as the primary compute path",
        }
    return enriched


def add_shape_metadata(report: dict[str, Any], shape_profile: ShapeProfile) -> dict[str, Any]:
    enriched = deepcopy(report)
    enriched["shape_profile"] = {
        "name": shape_profile.name,
        "static_batch": shape_profile.static_batch,
        "batch_size": shape_profile.batch_size if shape_profile.static_batch else "dynamic",
        "max_audio_samples": shape_profile.max_audio_samples,
        "max_decoder_latent_steps": shape_profile.max_decoder_latent_steps,
        "max_prefill_seq": shape_profile.max_prefill_seq,
        "max_decode_cache_seq": shape_profile.max_decode_cache_seq,
        "description": shape_profile.description,
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
    shape_profile: ShapeProfile | None = None,
) -> None:
    print(f"module={MODULE_EXPORT_CONTRACTS[module_key].display_name}")
    print(f"precision={precision.name}")
    print(f"compute_dtype={precision.compute_dtype}")
    print(f"host_float_dtype={precision.host_float_dtype}")
    print("input_names=" + ",".join(input_names))
    print("output_names=" + ",".join(output_names))
    enriched_report = add_precision_metadata(shape_report, precision, module_key=module_key)
    if shape_profile is not None:
        enriched_report = add_shape_metadata(enriched_report, shape_profile)
    print("shape_report=" + json.dumps(enriched_report, sort_keys=True))
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
