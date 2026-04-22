#!/usr/bin/env python3
"""Fast checks for configurable CPU ONNX Runtime session options."""

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

    from src.runtime.session_factory import OrtSessionFactory

    return OrtSessionFactory


def test_ort_session_options_are_configurable_without_creating_sessions() -> None:
    OrtSessionFactory = _runtime_classes()
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
        "log_severity_level": "warning",
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1,
    }

    options = factory._session_options()
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    assert options.execution_mode == ort.ExecutionMode.ORT_PARALLEL
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1


def test_ort_session_options_reject_invalid_values() -> None:
    OrtSessionFactory = _runtime_classes()

    with pytest.raises(ValueError, match="Unsupported ORT graph optimization level"):
        OrtSessionFactory(graph_optimization_level="fast")._session_options()

    with pytest.raises(ValueError, match="Unsupported ORT execution mode"):
        OrtSessionFactory(execution_mode="async")._session_options()

    with pytest.raises(ValueError, match="Unsupported ORT log severity"):
        OrtSessionFactory(log_severity_level="debug")._session_options()

    with pytest.raises(ValueError, match="intra_op_num_threads must be >= 0"):
        OrtSessionFactory(intra_op_num_threads=-1)._session_options()
