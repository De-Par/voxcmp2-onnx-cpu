#!/usr/bin/env python3
"""Fast checks for configurable CPU ONNX Runtime session options"""

from __future__ import annotations

import sys
from pathlib import Path

import onnxruntime as ort
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _runtime_classes():
    _ensure_repo_root_on_path()

    from src.runtime.session_factory import OnnxModelPaths, OrtSessionFactory

    return OrtSessionFactory, OnnxModelPaths


def test_ort_session_options_are_configurable_without_creating_sessions() -> None:
    OrtSessionFactory, _ = _runtime_classes()
    factory = OrtSessionFactory(
        graph_optimization_level="extended",
        execution_mode="parallel",
        intra_op_num_threads=2,
        inter_op_num_threads=1,
    )

    assert factory.options_summary() == {
        "provider": "CPUExecutionProvider",
        "graph_optimization_level": "extended",
        "execution_mode": "parallel",
        "log_severity_level": "error",
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1,
        "enable_mem_pattern": True,
        "enable_cpu_mem_arena": True,
        "enable_mem_reuse": True,
        "enable_profiling": False,
        "profile_file_prefix": None,
        "prefer_ort_format": True,
        "prefer_optimized_onnx": False,
    }

    options = factory._session_options()
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    assert options.execution_mode == ort.ExecutionMode.ORT_PARALLEL
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert options.enable_mem_pattern is True
    assert options.enable_cpu_mem_arena is True
    assert options.enable_mem_reuse is True


def test_ort_session_options_reject_invalid_values() -> None:
    OrtSessionFactory, _ = _runtime_classes()

    with pytest.raises(ValueError, match="Unsupported ORT graph optimization level"):
        OrtSessionFactory(graph_optimization_level="fast")._session_options()

    with pytest.raises(ValueError, match="Unsupported ORT execution mode"):
        OrtSessionFactory(execution_mode="async")._session_options()

    with pytest.raises(ValueError, match="Unsupported ORT log severity"):
        OrtSessionFactory(log_severity_level="debug")._session_options()

    with pytest.raises(ValueError, match="intra_op_num_threads must be >= 0"):
        OrtSessionFactory(intra_op_num_threads=-1)._session_options()


def test_validate_paths_ignores_zero_byte_ort_files(tmp_path: Path) -> None:
    OrtSessionFactory, OnnxModelPaths = _runtime_classes()

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    onnx_path = model_dir / "module.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    onnx_path.with_suffix(".onnx.data").write_bytes(b"fake-data")
    onnx_path.with_suffix(".ort").write_bytes(b"")
    paths = OnnxModelPaths(
        audio_encoder=onnx_path,
        audio_decoder=onnx_path,
        prefill=onnx_path,
        decode_chunk=onnx_path,
    )

    resolved = OrtSessionFactory(paths=paths).validate_paths()
    assert all(path == onnx_path for path in resolved.values())


def test_validate_paths_prefers_optimized_onnx_when_ort_missing(tmp_path: Path) -> None:
    OrtSessionFactory, OnnxModelPaths = _runtime_classes()

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    onnx_path = model_dir / "module.onnx"
    optimized_path = model_dir / "module.optimized.onnx"
    onnx_path.write_bytes(b"raw-onnx")
    onnx_path.with_suffix(".onnx.data").write_bytes(b"raw-data")
    optimized_path.write_bytes(b"optimized-onnx")
    optimized_path.with_suffix(".onnx.data").write_bytes(b"optimized-data")
    paths = OnnxModelPaths(
        audio_encoder=onnx_path,
        audio_decoder=onnx_path,
        prefill=onnx_path,
        decode_chunk=onnx_path,
    )

    resolved = OrtSessionFactory(paths=paths, prefer_optimized_onnx=True).validate_paths()
    assert all(path == optimized_path for path in resolved.values())


def test_invalid_ort_falls_back_to_onnx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    OrtSessionFactory, OnnxModelPaths = _runtime_classes()

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    onnx_path = model_dir / "module.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    onnx_path.with_suffix(".onnx.data").write_bytes(b"fake-data")
    ort_path = onnx_path.with_suffix(".ort")
    ort_path.write_bytes(b"not-a-real-ort")
    paths = OnnxModelPaths(
        audio_encoder=onnx_path,
        audio_decoder=onnx_path,
        prefill=onnx_path,
        decode_chunk=onnx_path,
    )
    calls: list[Path] = []

    class _FakeSession:
        def __init__(self, path: Path) -> None:
            self._model_path = str(path)

        @staticmethod
        def get_providers() -> list[str]:
            return ["CPUExecutionProvider"]

    def _fake_inference_session(path: str, *, sess_options, providers):
        del sess_options, providers
        candidate = Path(path)
        calls.append(candidate)
        if candidate.suffix == ".ort":
            raise RuntimeError("broken ort")
        return _FakeSession(candidate)

    monkeypatch.setattr(ort, "InferenceSession", _fake_inference_session)
    factory = OrtSessionFactory(paths=paths)

    session = factory.audio_encoder
    assert Path(session._model_path) == onnx_path
    assert calls == [ort_path, onnx_path]


def test_invalid_ort_falls_back_to_optimized_onnx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    OrtSessionFactory, OnnxModelPaths = _runtime_classes()

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    onnx_path = model_dir / "module.onnx"
    optimized_path = model_dir / "module.optimized.onnx"
    ort_path = model_dir / "module.ort"
    onnx_path.write_bytes(b"fake-onnx")
    onnx_path.with_suffix(".onnx.data").write_bytes(b"fake-data")
    optimized_path.write_bytes(b"optimized-onnx")
    optimized_path.with_suffix(".onnx.data").write_bytes(b"optimized-data")
    ort_path.write_bytes(b"not-a-real-ort")
    paths = OnnxModelPaths(
        audio_encoder=onnx_path,
        audio_decoder=onnx_path,
        prefill=onnx_path,
        decode_chunk=onnx_path,
    )
    calls: list[tuple[Path, int]] = []

    class _FakeSession:
        def __init__(self, path: Path) -> None:
            self._model_path = str(path)

        @staticmethod
        def get_providers() -> list[str]:
            return ["CPUExecutionProvider"]

    def _fake_inference_session(path: str, *, sess_options, providers):
        del providers
        candidate = Path(path)
        calls.append((candidate, sess_options.graph_optimization_level))
        if candidate.suffix == ".ort":
            raise RuntimeError("broken ort")
        return _FakeSession(candidate)

    monkeypatch.setattr(ort, "InferenceSession", _fake_inference_session)
    factory = OrtSessionFactory(paths=paths, prefer_optimized_onnx=True)

    session = factory.audio_encoder
    assert Path(session._model_path) == optimized_path
    assert calls[0][0] == ort_path
    assert calls[1][0] == optimized_path
    assert calls[1][1] == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
