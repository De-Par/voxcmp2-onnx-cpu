from __future__ import annotations

import os
from pathlib import Path

import onnx
import pytest
import torch
from onnx import TensorProto

from src.experiments.bf16_feasibility import (
    DEFAULT_STORAGE_ONLY_OUTPUT_DIR,
    _check_storage_only_output_policy,
    _parser as bf16_feasibility_parser,
)
from src.export.common import (
    BF16_MODULE_POLICIES,
    MODULE_EXPORT_CONTRACTS,
    configure_module_precision,
    get_precision_profile,
)
from src.export.export_audio_vae_decoder import AudioVAEDecoderWrapper
from src.export.export_audio_vae_encoder import AudioVAEEncoderWrapper


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_dtype: torch.dtype | None = None

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen_dtype = waveform.dtype
        return {"mu": waveform}


class _FakeAudioVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FakeEncoder()
        self.seen_decode_dtype: torch.dtype | None = None

    def decode(self, latent: torch.Tensor, _sr_cond: torch.Tensor) -> torch.Tensor:
        self.seen_decode_dtype = latent.dtype
        return latent[:, :1, :]


def _cast_to(node: onnx.NodeProto) -> int | None:
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _storage_only_bf16_to_float_casts(model_path: Path) -> list[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    bf16_initializers = {
        initializer.name for initializer in model.graph.initializer if initializer.data_type == TensorProto.BFLOAT16
    }
    return [
        node.name or node.output[0]
        for node in model.graph.node
        if node.op_type == "Cast"
        and node.input
        and node.input[0] in bf16_initializers
        and _cast_to(node) == TensorProto.FLOAT
    ]


def test_bf16_profile_is_real_compute_not_storage_only() -> None:
    bf16 = get_precision_profile("bf16")
    assert bf16.compute_dtype == "bfloat16"
    assert bf16.production_compute is True
    assert bf16.storage_only is False

    module = configure_module_precision(torch.nn.Linear(2, 2), bf16)
    assert next(module.parameters()).dtype == torch.bfloat16


def test_bf16_module_policies_are_explicit() -> None:
    assert set(BF16_MODULE_POLICIES) == set(MODULE_EXPORT_CONTRACTS)
    for module_key, policy in BF16_MODULE_POLICIES.items():
        assert policy.bf16_compute_regions, module_key
        assert all("storage" not in region.lower() for region in policy.bf16_compute_regions)


def test_audio_vae_wrappers_execute_bf16_inside_float32_boundary() -> None:
    bf16 = get_precision_profile("bf16")

    encoder_model = _FakeAudioVAE()
    encoder = AudioVAEEncoderWrapper(encoder_model, bf16)
    encoder_out = encoder(torch.ones(1, 1, 8, dtype=torch.float32))
    assert encoder_model.encoder.seen_dtype == torch.bfloat16
    assert encoder_out.dtype == torch.float32

    decoder_model = _FakeAudioVAE()
    decoder = AudioVAEDecoderWrapper(decoder_model, bf16)
    decoder_out = decoder(torch.ones(1, 64, 2, dtype=torch.float32), torch.tensor([48000], dtype=torch.int32))
    assert decoder_model.seen_decode_dtype == torch.bfloat16
    assert decoder_out.dtype == torch.float32


def test_storage_only_converter_cannot_default_to_production_bf16_dir() -> None:
    args = bf16_feasibility_parser().parse_args(["--mode", "convert", "--models", "audio_vae_encoder"])
    assert args.output_dir == DEFAULT_STORAGE_ONLY_OUTPUT_DIR

    args.output_dir = Path("models/onnx/bf16")
    with pytest.raises(ValueError, match="storage-only BF16"):
        _check_storage_only_output_policy(args)


def test_optional_bf16_onnx_graphs_have_no_initializer_cast_back_pattern() -> None:
    raw_paths = os.environ.get("VOXCPM2_BF16_ONNX_PATHS")
    if not raw_paths:
        pytest.skip("set VOXCPM2_BF16_ONNX_PATHS to inspect exported BF16 ONNX graphs")

    for raw_path in raw_paths.split(os.pathsep):
        model_path = Path(raw_path).expanduser()
        assert model_path.is_file(), model_path
        storage_only_casts = _storage_only_bf16_to_float_casts(model_path)
        assert storage_only_casts == [], {"model": str(model_path), "casts": storage_only_casts[:20]}
