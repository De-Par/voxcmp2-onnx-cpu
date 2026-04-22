#!/usr/bin/env python3
"""Inspect and optionally create legacy BF16-storage ONNX copies.

This tool is not the production BF16 path. Conversion mode stores selected
FLOAT initializers as BFLOAT16 and immediately casts them back to FLOAT before
graph use. That measures storage feasibility and ORT CPU loader coverage, but
it intentionally provides no BF16 compute benefit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_BF16_ROOT = (REPO_ROOT / "models" / "onnx" / "bf16").resolve()
DEFAULT_STORAGE_ONLY_OUTPUT_DIR = Path("artifacts/experiments/bf16_storage_only")

DEFAULT_MODELS = {
    "audio_vae_encoder": REPO_ROOT / "models" / "onnx" / "fp32" / "audio_vae_encoder" / "audio_vae_encoder.onnx",
    "audio_vae_decoder": REPO_ROOT / "models" / "onnx" / "fp32" / "audio_vae_decoder" / "audio_vae_decoder.onnx",
    "prefill": REPO_ROOT / "models" / "onnx" / "fp32" / "prefill" / "voxcpm2_prefill.onnx",
    "decode_step": REPO_ROOT / "models" / "onnx" / "fp32" / "decode_step" / "voxcpm2_decode_step.onnx",
}
LARGE_MODELS = {"prefill", "decode_step"}

DTYPE_SIZE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.COMPLEX64: 8,
    TensorProto.COMPLEX128: 16,
    TensorProto.BFLOAT16: 2,
}


@dataclass(frozen=True)
class ModelTarget:
    name: str
    path: Path


def _dtype_name(data_type: int) -> str:
    try:
        return TensorProto.DataType.Name(data_type)
    except ValueError:
        return f"UNKNOWN_{data_type}"


def _initializer_numel(initializer: TensorProto) -> int:
    return math.prod(int(dim) for dim in initializer.dims) if initializer.dims else 1


def _initializer_nbytes(initializer: TensorProto) -> int:
    itemsize = DTYPE_SIZE_BYTES.get(initializer.data_type, 0)
    return _initializer_numel(initializer) * itemsize


def _external_data_files(model_path: Path, model: onnx.ModelProto) -> list[Path]:
    files: set[Path] = set()
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location":
                files.add((model_path.parent / entry.value).resolve())
    fallback = model_path.with_suffix(model_path.suffix + ".data")
    if fallback.exists():
        files.add(fallback.resolve())
    return sorted(files)


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _artifact_size(model_path: Path, model: onnx.ModelProto) -> dict[str, Any]:
    files = [model_path.resolve(), *_external_data_files(model_path, model)]
    return {
        "total_bytes": sum(_file_size(path) for path in files),
        "files": [{"path": str(path), "bytes": _file_size(path)} for path in files],
    }


def _cast_to(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "to":
            return _dtype_name(attr.i)
    return None


def _cast_report(model: onnx.ModelProto) -> dict[str, Any]:
    cast_outputs = {output for node in model.graph.node if node.op_type == "Cast" for output in node.output}
    cast_to_counts: Counter[str] = Counter()
    chain_edges = []
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        cast_to_counts[_cast_to(node) or "UNSPECIFIED"] += 1
        if node.input and node.input[0] in cast_outputs:
            chain_edges.append(
                {
                    "node": node.name or "<unnamed>",
                    "input": node.input[0],
                    "output": node.output[0] if node.output else None,
                    "to": _cast_to(node),
                }
            )
    return {
        "cast_nodes": sum(cast_to_counts.values()),
        "cast_to": dict(sorted(cast_to_counts.items())),
        "cast_chain_edges": len(chain_edges),
        "cast_chain_samples": chain_edges[:20],
    }


def analyze_model(target: ModelTarget) -> dict[str, Any]:
    model = onnx.load(str(target.path), load_external_data=False)
    dtype_counts: Counter[str] = Counter()
    dtype_bytes: defaultdict[str, int] = defaultdict(int)
    float_initializer_count = 0
    float_initializer_bytes = 0
    for initializer in model.graph.initializer:
        dtype = _dtype_name(initializer.data_type)
        nbytes = _initializer_nbytes(initializer)
        dtype_counts[dtype] += 1
        dtype_bytes[dtype] += nbytes
        if initializer.data_type == TensorProto.FLOAT:
            float_initializer_count += 1
            float_initializer_bytes += nbytes

    return {
        "name": target.name,
        "path": str(target.path),
        "exists": target.path.exists(),
        "ir_version": model.ir_version,
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "graph": {
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "inputs": [item.name for item in model.graph.input],
            "outputs": [item.name for item in model.graph.output],
        },
        "initializer_dtypes": {
            dtype: {"count": dtype_counts[dtype], "logical_bytes": dtype_bytes[dtype]} for dtype in sorted(dtype_counts)
        },
        "float_initializers": {
            "count": float_initializer_count,
            "logical_bytes": float_initializer_bytes,
            "estimated_bf16_logical_bytes": float_initializer_bytes // 2,
        },
        "casts": _cast_report(model),
        "artifact_size": _artifact_size(target.path, model),
    }


def _fp32_to_bf16_raw(array: np.ndarray) -> bytes:
    fp32 = np.asarray(array, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bf16 = (rounded >> np.uint32(16)).astype(np.uint16)
    return bf16.tobytes()


def _safe_node_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "initializer"


def _replace_initializer(graph: onnx.GraphProto, old_name: str, new_initializer: TensorProto) -> None:
    for index, initializer in enumerate(graph.initializer):
        if initializer.name == old_name:
            graph.initializer.remove(initializer)
            graph.initializer.insert(index, new_initializer)
            return
    raise KeyError(old_name)


def convert_float_initializers_to_bf16_with_fp32_casts(
    source_path: Path,
    output_path: Path,
    *,
    min_tensor_bytes: int,
) -> dict[str, Any]:
    model = onnx.load(str(source_path), load_external_data=True)
    graph_input_names = {item.name for item in model.graph.input}
    graph_output_names = {item.name for item in model.graph.output}
    converted: dict[str, str] = {}
    skipped: Counter[str] = Counter()

    for initializer in list(model.graph.initializer):
        if initializer.data_type != TensorProto.FLOAT:
            skipped["non_float"] += 1
            continue
        if initializer.name in graph_input_names or initializer.name in graph_output_names:
            skipped["graph_boundary"] += 1
            continue
        if _initializer_nbytes(initializer) < min_tensor_bytes:
            skipped["below_min_tensor_bytes"] += 1
            continue

        array = numpy_helper.to_array(initializer)
        bf16_initializer = helper.make_tensor(
            name=initializer.name,
            data_type=TensorProto.BFLOAT16,
            dims=list(initializer.dims),
            vals=_fp32_to_bf16_raw(array),
            raw=True,
        )
        bf16_initializer.doc_string = initializer.doc_string
        _replace_initializer(model.graph, initializer.name, bf16_initializer)
        converted[initializer.name] = f"{initializer.name}__bf16_to_fp32"

    cast_nodes = []
    for initializer_name, cast_output in converted.items():
        cast_nodes.append(
            helper.make_node(
                "Cast",
                inputs=[initializer_name],
                outputs=[cast_output],
                to=TensorProto.FLOAT,
                name=f"Cast_BF16ToFP32_{_safe_node_name(initializer_name)}",
            )
        )

    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in converted:
                node.input[index] = converted[input_name]

    if cast_nodes:
        for cast_node in reversed(cast_nodes):
            model.graph.node.insert(0, cast_node)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for stale_path in (output_path, output_path.with_suffix(output_path.suffix + ".data")):
        if stale_path.exists():
            stale_path.unlink()
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(output_path))
    return {
        "converted_float_initializers": len(converted),
        "skipped": dict(sorted(skipped.items())),
        "output_path": str(output_path),
    }


def _session_load_report(model_path: Path) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        return {"ok": True, "providers": session.get_providers()}
    except Exception as exc:  # noqa: BLE001 - report ORT feasibility without hiding provider errors.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _default_targets(names: list[str]) -> list[ModelTarget]:
    return [ModelTarget(name=name, path=DEFAULT_MODELS[name]) for name in names]


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    path.expanduser().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_report_json(args: argparse.Namespace) -> Path:
    if args.report_json is not None:
        return args.report_json
    model_part = "all" if set(args.models) == set(DEFAULT_MODELS) else "_".join(args.models)
    precision = "fp32" if args.mode == "analyze" else "bf16"
    return Path("artifacts/reports/bf16_feasibility") / f"{precision}_{model_part}_report.json"


def _check_large_model_policy(args: argparse.Namespace) -> None:
    requested_large_models = sorted(set(args.models).intersection(LARGE_MODELS))
    if args.mode == "convert" and requested_large_models and not args.include_large_models:
        raise ValueError(
            "Refusing to convert large models "
            + ", ".join(requested_large_models)
            + "; pass --include-large-models to allow multi-GB BF16 artifacts."
        )


def _check_storage_only_output_policy(args: argparse.Namespace) -> None:
    if args.mode != "convert":
        return
    output_dir = args.output_dir.expanduser()
    try:
        output_resolved = output_dir.resolve()
    except FileNotFoundError:
        output_resolved = output_dir.parent.resolve() / output_dir.name
    if output_resolved == PRODUCTION_BF16_ROOT or output_resolved.is_relative_to(PRODUCTION_BF16_ROOT):
        raise ValueError(
            "Refusing to write storage-only BF16 conversion into production models/onnx/bf16. "
            "Use src/export/export_all.py --precision bf16 for production BF16 compute artifacts, "
            "or choose an artifacts/experiments output directory for this legacy feasibility tool."
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _check_large_model_policy(args)
    _check_storage_only_output_policy(args)
    targets = _default_targets(args.models)
    report_json = _default_report_json(args)
    report: dict[str, Any] = {
        "mode": args.mode,
        "min_tensor_bytes": args.min_tensor_bytes,
        "report_json": str(report_json),
        "models": [],
    }

    for target in targets:
        if not target.path.is_file():
            report["models"].append({"name": target.name, "path": str(target.path), "exists": False})
            continue

        before = analyze_model(target)
        entry: dict[str, Any] = {"name": target.name, "before": before}
        if args.mode == "convert":
            print(f"bf16_feasibility: converting model={target.name}", file=sys.stderr)
            output_path = args.output_dir.expanduser() / target.name / target.path.name
            conversion = convert_float_initializers_to_bf16_with_fp32_casts(
                target.path,
                output_path,
                min_tensor_bytes=args.min_tensor_bytes,
            )
            after = analyze_model(ModelTarget(name=target.name, path=output_path))
            before_size = before["artifact_size"]["total_bytes"]
            after_size = after["artifact_size"]["total_bytes"]
            entry.update(
                {
                    "conversion": conversion,
                    "after": after,
                    "size_delta": {
                        "before_bytes": before_size,
                        "after_bytes": after_size,
                        "saved_bytes": before_size - after_size,
                        "after_over_before": round(after_size / before_size, 6) if before_size else None,
                    },
                }
            )
            if args.check_ort:
                entry["ort_session_load"] = _session_load_report(output_path)
        report["models"].append(entry)

    _write_report(report_json, report)
    return report


def _print_success(report: dict[str, Any]) -> None:
    converted = [
        {
            "name": entry["name"],
            "output_path": entry.get("conversion", {}).get("output_path"),
            "saved_bytes": entry.get("size_delta", {}).get("saved_bytes"),
            "ort_ok": entry.get("ort_session_load", {}).get("ok"),
        }
        for entry in report["models"]
        if "conversion" in entry
    ]
    payload: dict[str, Any] = {
        "bf16_feasibility": "ok",
        "mode": report["mode"],
        "models": [entry["name"] for entry in report["models"]],
        "report_json": report["report_json"],
    }
    if converted:
        payload["converted"] = converted
    print(json.dumps(payload, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze legacy storage-only BF16 feasibility without touching production BF16 compute exports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["analyze", "convert"],
        default="analyze",
        help="Analyze FP32 artifacts or write experimental BF16-initializer copies.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(DEFAULT_MODELS),
        default=sorted(DEFAULT_MODELS),
        help="Model artifacts to inspect or convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STORAGE_ONLY_OUTPUT_DIR,
        help="Directory for converted storage-only experiment artifacts; production BF16 uses export_all.py.",
    )
    parser.add_argument(
        "--report-json", type=Path, help="JSON report output path. Defaults to artifacts/reports/bf16_feasibility."
    )
    parser.add_argument(
        "--min-tensor-bytes",
        type=int,
        default=4096,
        help="Only FLOAT initializers at least this large are converted in convert mode.",
    )
    parser.add_argument("--check-ort", action="store_true", help="Try loading converted models with ORT CPU.")
    parser.add_argument(
        "--include-large-models",
        action="store_true",
        help="Allow conversion of prefill/decode_step. This can create multi-GB artifacts.",
    )
    return parser


def main() -> int:
    try:
        report = run(_parser().parse_args())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    _print_success(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
