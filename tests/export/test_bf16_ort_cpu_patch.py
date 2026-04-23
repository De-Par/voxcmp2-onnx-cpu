from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from src.export.common import apply_bf16_ort_cpu_compatibility_pass


def _make_round_model(path: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Round", ["x"], ["y"], name="round_bf16"),
        ],
        "bf16_round",
        [
            helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [1]),
        ],
        [
            helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, str(path))


def test_bf16_ort_cpu_patch_inserts_documented_fp32_island(tmp_path: Path) -> None:
    model_path = tmp_path / "round.onnx"
    _make_round_model(model_path)

    report = apply_bf16_ort_cpu_compatibility_pass(model_path, op_types=("Round",))

    assert report["patched_nodes"] == 1
    assert report["patched_by_op"] == {"Round": 1}
    assert report["inserted_input_casts"] == 1
    assert report["inserted_output_casts"] == 1

    patched = onnx.load(str(model_path), load_external_data=False)
    assert [node.op_type for node in patched.graph.node] == ["Cast", "Round", "Cast"]
    assert patched.graph.node[0].attribute[0].i == TensorProto.FLOAT
    assert patched.graph.node[2].attribute[0].i == TensorProto.BFLOAT16
    assert patched.graph.output[0].name == "y"


def test_bf16_ort_cpu_patch_keeps_where_condition_bool(tmp_path: Path) -> None:
    model_path = tmp_path / "where.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Where", ["cond", "left", "right"], ["y"], name="where_bf16"),
        ],
        "bf16_where",
        [
            helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
            helper.make_tensor_value_info("left", TensorProto.BFLOAT16, [1]),
            helper.make_tensor_value_info("right", TensorProto.BFLOAT16, [1]),
        ],
        [
            helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, str(model_path))

    report = apply_bf16_ort_cpu_compatibility_pass(model_path, op_types=("Where",))

    assert report["patched_nodes"] == 1
    patched = onnx.load(str(model_path), load_external_data=False)
    where_node = next(node for node in patched.graph.node if node.op_type == "Where")
    assert where_node.input[0] == "cond"
    assert where_node.input[1].endswith("__bf16_to_fp32")
    assert where_node.input[2].endswith("__bf16_to_fp32")
