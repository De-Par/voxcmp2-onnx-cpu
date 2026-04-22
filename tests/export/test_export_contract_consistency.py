from __future__ import annotations

import importlib

import pytest

from src.export.common import (
    MODULE_EXPORT_CONTRACTS,
    MODULE_OUTPUT_LAYOUTS,
    PRECISION_CHOICES,
    default_output_path,
    get_precision_profile,
    get_shape_profile,
)


def _public_contract_snapshot(_precision_name: str) -> dict[str, tuple[tuple[str, ...], tuple[str, ...], str]]:
    return {
        key: (contract.input_names, contract.output_names, contract.state_semantics)
        for key, contract in MODULE_EXPORT_CONTRACTS.items()
    }


def test_precision_profiles_keep_public_contract_identical() -> None:
    fp32 = get_precision_profile("fp32")
    bf16 = get_precision_profile("bf16")

    assert fp32.host_float_dtype == bf16.host_float_dtype == "float32"
    assert fp32.compute_dtype == "float32"
    assert bf16.compute_dtype == "bfloat16"

    assert _public_contract_snapshot(fp32.name)
    assert _public_contract_snapshot(fp32.name) == _public_contract_snapshot(bf16.name)


def test_default_paths_are_precision_scoped() -> None:
    for precision_name in PRECISION_CHOICES:
        precision = get_precision_profile(precision_name)
        for module_key, layout in MODULE_OUTPUT_LAYOUTS.items():
            path = default_output_path(module_key, precision)
            assert path.parts[-4:] == ("onnx", precision_name, layout.directory, layout.filename)


def test_production_shape_profile_specializes_runtime_safe_axes() -> None:
    production = get_shape_profile("production")
    flex = get_shape_profile("flex")

    assert production.static_batch is True
    assert production.batch_size == 1
    assert production.max_prefill_seq is not None
    assert production.max_decode_cache_seq is not None
    assert production.max_audio_samples is not None
    assert production.max_decoder_latent_steps is not None

    assert flex.static_batch is False
    assert flex.max_prefill_seq is None
    assert flex.max_decode_cache_seq is None


def test_exporter_constants_match_shared_contracts() -> None:
    pytest.importorskip("torch")

    modules = {
        "audio_vae_encoder": "src.export.export_audio_vae_encoder",
        "audio_vae_decoder": "src.export.export_audio_vae_decoder",
        "prefill": "src.export.export_prefill",
        "decode_step": "src.export.export_decode_step",
        "decode_chunk": "src.export.export_decode_chunk",
    }

    for module_key, module_name in modules.items():
        module = importlib.import_module(module_name)
        contract = MODULE_EXPORT_CONTRACTS[module_key]
        assert tuple(module.INPUT_NAMES) == contract.input_names
        assert tuple(module.OUTPUT_NAMES) == contract.output_names
