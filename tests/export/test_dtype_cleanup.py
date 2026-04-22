from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from tools.profile.summarize_dtype_casts import analyze_casts


def _save_synthetic_cast_graph(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
    w_out = helper.make_tensor_value_info("w_out", TensorProto.FLOAT, [1])
    w = helper.make_tensor("w_bf16", TensorProto.BFLOAT16, [1], vals=b"\x00\x3f", raw=True)
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["x"], ["x_float"], to=TensorProto.FLOAT, name="redundant_input_cast"),
            helper.make_node("Cast", ["x_float"], ["z"], to=TensorProto.FLOAT, name="redundant_chain_cast"),
            helper.make_node("Cast", ["x"], ["x_bf16"], to=TensorProto.BFLOAT16, name="to_bf16"),
            helper.make_node("Cast", ["x_bf16"], ["y"], to=TensorProto.FLOAT, name="bf16_back_to_float"),
            helper.make_node("Cast", ["w_bf16"], ["w_out"], to=TensorProto.FLOAT, name="storage_only_weight"),
        ],
        "synthetic_casts",
        [x],
        [y, z, w_out],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    onnx.save(model, path)


def test_cast_summary_classifies_cleanup_targets(tmp_path: Path) -> None:
    model_path = tmp_path / "casts.onnx"
    _save_synthetic_cast_graph(model_path)

    report = analyze_casts(model_path)

    assert report["cast_nodes"] == 5
    assert report["redundant_casts"]["count"] >= 2
    assert report["direct_cast_chains"]["count"] == 2
    assert report["fp32_bf16_ping_pong"]["count"] == 1
    assert report["storage_only_bf16_to_fp32"]["count"] == 1
    assert report["unavoidable_precision_boundaries"]["count"] >= 1


def test_cast_tensor_if_needed_returns_same_tensor_for_matching_dtype() -> None:
    torch = pytest.importorskip("torch")

    from src.export.common import cast_tensor_if_needed

    tensor = torch.ones(2, dtype=torch.float32)
    assert cast_tensor_if_needed(tensor, torch.float32) is tensor
    assert cast_tensor_if_needed(tensor, torch.bfloat16).dtype == torch.bfloat16
