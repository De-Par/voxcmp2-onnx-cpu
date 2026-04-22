#!/usr/bin/env python3
"""Host decode-loop stop policy tests.

These tests use tiny fake ONNX sessions. They verify orchestration behavior
without loading the real VoxCPM2 weights, so they can run quickly on every
platform.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _runtime_classes():
    _ensure_repo_root_on_path()

    from src.runtime.pipeline import VoxCPM2OnnxPipeline, VoxCPM2RuntimeConfig

    return VoxCPM2OnnxPipeline, VoxCPM2RuntimeConfig


class _FakeSession:
    def __init__(self, run_impl):
        self._run_impl = run_impl

    def run(self, output_names, inputs):
        return self._run_impl(output_names, inputs)


class _FakeSessions:
    def __init__(self, *, stop_after: int | None):
        self.decode_calls = 0
        self.chunk_size = 4
        self.stop_after = stop_after
        self.prefill = _FakeSession(self._prefill_run)
        self.decode_chunk = _FakeSession(self._decode_run)
        self.audio_decoder = _FakeSession(self._decoder_run)

    def _prefill_run(self, output_names, _inputs):
        values = {
            "lm_hidden": np.zeros((1, 1, 2), dtype=np.float32),
            "residual_hidden": np.zeros((1, 1, 2), dtype=np.float32),
            "prefix_feat_cond": np.zeros((1, 1, 2), dtype=np.float32),
            "base_k_cache": np.zeros((1, 1, 1, 0, 1), dtype=np.float32),
            "base_v_cache": np.zeros((1, 1, 1, 0, 1), dtype=np.float32),
            "base_cache_length": np.array([0], dtype=np.int64),
            "residual_k_cache": np.zeros((1, 1, 1, 0, 1), dtype=np.float32),
            "residual_v_cache": np.zeros((1, 1, 1, 0, 1), dtype=np.float32),
            "residual_cache_length": np.array([0], dtype=np.int64),
        }
        return [values[name] for name in output_names]

    def _decode_run(self, output_names, inputs):
        self.decode_calls += 1
        start_step = int(inputs["base_current_length"][0])
        step_numbers = np.arange(start_step + 1, start_step + self.chunk_size + 1)
        should_stop = (
            step_numbers >= self.stop_after if self.stop_after is not None else np.zeros(self.chunk_size, dtype=bool)
        )
        stop_logits = np.stack(
            [
                np.zeros(self.chunk_size, dtype=np.float32),
                np.where(should_stop, 1.0, -1.0).astype(np.float32),
            ],
            axis=-1,
        )[None, :, :]
        values = {
            "pred_audio_feature": step_numbers.reshape(1, self.chunk_size, 1, 1).astype(np.float32).repeat(2, axis=3),
            "decoder_latent": np.zeros((1, 2, self.chunk_size), dtype=np.float32),
            "stop_logits": stop_logits,
            "next_lm_hidden": inputs["lm_hidden"],
            "next_residual_hidden": inputs["residual_hidden"],
            "next_prefix_feat_cond": inputs["prefix_feat_cond"],
            "base_k_update": np.zeros((1, 1, 1, self.chunk_size, 1), dtype=np.float32),
            "base_v_update": np.zeros((1, 1, 1, self.chunk_size, 1), dtype=np.float32),
            "next_base_current_length": inputs["base_current_length"] + self.chunk_size,
            "residual_k_update": np.zeros((1, 1, 1, self.chunk_size, 1), dtype=np.float32),
            "residual_v_update": np.zeros((1, 1, 1, self.chunk_size, 1), dtype=np.float32),
            "next_residual_current_length": inputs["residual_current_length"] + self.chunk_size,
        }
        return [values[name] for name in output_names]

    def _decoder_run(self, output_names, inputs):
        assert output_names == ["waveform"]
        latent_steps = inputs["latent"].shape[-1]
        return [np.zeros((1, 1, latent_steps), dtype=np.float32)]


def _fake_pipeline(stop_after: int | None, *, safety_max_steps: int = 16) -> tuple[object, _FakeSessions]:
    VoxCPM2OnnxPipeline, VoxCPM2RuntimeConfig = _runtime_classes()
    sessions = _FakeSessions(stop_after=stop_after)
    config = VoxCPM2RuntimeConfig(feat_dim=2, patch_size=1, decode_safety_max_steps=safety_max_steps)
    pipeline = VoxCPM2OnnxPipeline(sessions=sessions, config=config)
    pipeline.build_prefill_inputs = lambda *args, **kwargs: {
        "text_tokens": np.array([[1]], dtype=np.int64),
        "text_mask": np.array([[1]], dtype=np.int64),
        "audio_features": np.zeros((1, 1, 2), dtype=np.float32),
        "audio_mask": np.array([[0]], dtype=np.int64),
    }
    return pipeline, sessions


def test_decode_loop_stops_before_large_upper_bound() -> None:
    pipeline, sessions = _fake_pipeline(stop_after=3)
    result = pipeline.synthesize_with_metadata("hello", max_steps=1000, min_steps=0)

    assert sessions.decode_calls == 1
    assert result.metadata.decode_steps == 3
    assert result.metadata.stop_reason == "stop_logits"
    assert result.metadata.effective_max_steps == 1000
    assert result.waveform.shape == (3,)


def test_auto_decode_loop_uses_safety_cap_without_stop() -> None:
    pipeline, sessions = _fake_pipeline(stop_after=None, safety_max_steps=5)
    result = pipeline.synthesize_with_metadata("hello", max_steps=0, min_steps=0)

    assert sessions.decode_calls == 2
    assert result.metadata.decode_steps == 5
    assert result.metadata.stop_reason == "safety_max_steps"
    assert result.metadata.effective_max_steps == 5
    assert result.waveform.shape == (5,)


def main() -> int:
    test_decode_loop_stops_before_large_upper_bound()
    test_auto_decode_loop_uses_safety_cap_without_stop()
    print("decode_stop_policy_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
