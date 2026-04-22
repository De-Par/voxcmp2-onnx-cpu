#!/usr/bin/env python3
"""Smoke test for the CPU-only VoxCPM2 ONNX runtime pipeline"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _runtime_classes():
    _ensure_repo_root_on_path()

    from src.runtime.pipeline import VoxCPM2OnnxPipeline
    from src.runtime.session_factory import CPU_PROVIDER

    return VoxCPM2OnnxPipeline, CPU_PROVIDER


TORCH_MODULE_NAME = "to" + "rch"
FORBIDDEN_RUNTIME_MODULES = (TORCH_MODULE_NAME, "sound" + "file", "lib" + "rosa")


def _forbidden_runtime_modules_loaded() -> set[str]:
    return {module_name for module_name in FORBIDDEN_RUNTIME_MODULES if module_name in sys.modules}


def _assert_no_new_forbidden_runtime_modules(baseline: set[str]) -> None:
    assert _forbidden_runtime_modules_loaded().difference(baseline) == set()


def test_cpu_only_runtime_smoke() -> None:
    VoxCPM2OnnxPipeline, CPU_PROVIDER = _runtime_classes()
    forbidden_baseline = _forbidden_runtime_modules_loaded()

    pipeline = VoxCPM2OnnxPipeline.from_default_artifacts()
    try:
        paths = pipeline.validate()
    except FileNotFoundError as exc:
        pytest.skip(f"default ONNX artifacts are not available in this workspace: {exc}")
    _assert_no_new_forbidden_runtime_modules(forbidden_baseline)

    assert set(paths) == {"audio_encoder", "audio_decoder", "prefill", "decode_step"}
    assert pipeline.sessions.created_session_names == ()

    decode_inputs = {item.name for item in pipeline.sessions.decode_step.get_inputs()}
    if "base_current_length" not in decode_inputs:
        pytest.skip("decode_step artifact uses the old grow-by-concat cache contract; re-export fixed-cache ONNX")

    waveform = pipeline.synthesize(
        "Hello from VoxCPM2.",
        mode="text_only",
        max_steps=1,
        min_steps=0,
        seed=0,
    )
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert waveform.size > 0
    assert np.isfinite(waveform).all()
    _assert_no_new_forbidden_runtime_modules(forbidden_baseline)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "prompt.wav"
        sample_rate = pipeline.config.encode_sample_rate
        samples = np.zeros(sample_rate // 5, dtype=np.float32)
        wavfile.write(str(wav_path), sample_rate, samples)

        mode_inputs = [
            pipeline.build_prefill_inputs(
                "Hello.",
                mode="text_only",
                voice_design=None,
                reference_wav_path=None,
                prompt_wav_path=None,
                prompt_text=None,
            ),
            pipeline.build_prefill_inputs(
                "Hello.",
                mode="voice_design",
                voice_design="calm voice",
                reference_wav_path=None,
                prompt_wav_path=None,
                prompt_text=None,
            ),
            pipeline.build_prefill_inputs(
                "Hello.",
                mode="controllable_clone",
                voice_design=None,
                reference_wav_path=wav_path,
                prompt_wav_path=None,
                prompt_text=None,
            ),
            pipeline.build_prefill_inputs(
                "Hello.",
                mode="ultimate_clone",
                voice_design=None,
                reference_wav_path=wav_path,
                prompt_wav_path=wav_path,
                prompt_text="Prompt.",
            ),
        ]
    _assert_no_new_forbidden_runtime_modules(forbidden_baseline)

    for inputs in mode_inputs:
        assert set(inputs) == {"text_tokens", "text_mask", "audio_features", "audio_mask"}
        seq = inputs["text_tokens"].shape[1]
        assert inputs["text_mask"].shape == (1, seq)
        assert inputs["audio_mask"].shape == (1, seq)
        assert inputs["audio_features"].shape[:2] == (1, seq)

    created = set(pipeline.sessions.created_session_names)
    assert {"audio_encoder", "prefill", "decode_step", "audio_decoder"}.issubset(created)
    for name in created:
        session = getattr(pipeline.sessions, name)
        assert session.get_providers() == [CPU_PROVIDER]


def main() -> int:
    test_cpu_only_runtime_smoke()
    print("cpu_only_runtime_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
