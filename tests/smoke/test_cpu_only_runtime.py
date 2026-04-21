#!/usr/bin/env python3
"""Smoke test for the CPU-only VoxCPM2 ONNX runtime pipeline."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.runtime.pipeline import VoxCPM2OnnxPipeline
from src.runtime.session_factory import CPU_PROVIDER


def test_cpu_only_runtime_smoke() -> None:
    pipeline = VoxCPM2OnnxPipeline.from_default_artifacts()
    paths = pipeline.validate()
    assert set(paths) == {"audio_encoder", "audio_decoder", "prefill", "decode_step"}
    assert pipeline.sessions.created_session_names == ()

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

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "prompt.wav"
        sample_rate = pipeline.config.encode_sample_rate
        samples = np.zeros(sample_rate // 5, dtype=np.float32)
        sf.write(str(wav_path), samples, sample_rate)

        mode_inputs = [
            pipeline.build_prefill_inputs("Hello.", mode="text_only", voice_design=None, reference_wav_path=None, prompt_wav_path=None, prompt_text=None),
            pipeline.build_prefill_inputs("Hello.", mode="voice_design", voice_design="calm voice", reference_wav_path=None, prompt_wav_path=None, prompt_text=None),
            pipeline.build_prefill_inputs("Hello.", mode="controllable_clone", voice_design=None, reference_wav_path=wav_path, prompt_wav_path=None, prompt_text=None),
            pipeline.build_prefill_inputs("Hello.", mode="ultimate_clone", voice_design=None, reference_wav_path=wav_path, prompt_wav_path=wav_path, prompt_text="Prompt."),
        ]
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
