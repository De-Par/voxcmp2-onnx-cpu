"""Lazy ONNX Runtime CPU session factory for VoxCPM2 modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[2]
CPU_PROVIDER = "CPUExecutionProvider"
FORBIDDEN_PROVIDERS = {"CUDAExecutionProvider", "CoreMLExecutionProvider", "MPSExecutionProvider"}


@dataclass(frozen=True)
class OnnxModelPaths:
    audio_encoder: Path = REPO_ROOT / "artifacts" / "audio_vae_encoder" / "audio_vae_encoder.onnx"
    audio_decoder: Path = REPO_ROOT / "artifacts" / "audio_vae_decoder" / "audio_vae_decoder.onnx"
    prefill: Path = REPO_ROOT / "artifacts" / "prefill" / "voxcpm2_prefill.onnx"
    decode_step: Path = REPO_ROOT / "artifacts" / "decode_step" / "voxcpm2_decode_step.onnx"

    def expanded(self) -> "OnnxModelPaths":
        return OnnxModelPaths(
            audio_encoder=self.audio_encoder.expanduser().resolve(),
            audio_decoder=self.audio_decoder.expanduser().resolve(),
            prefill=self.prefill.expanduser().resolve(),
            decode_step=self.decode_step.expanduser().resolve(),
        )

    def items(self) -> Iterable[tuple[str, Path]]:
        yield "audio_encoder", self.audio_encoder
        yield "audio_decoder", self.audio_decoder
        yield "prefill", self.prefill
        yield "decode_step", self.decode_step


@dataclass
class OrtSessionFactory:
    """Create ONNX Runtime sessions lazily and CPU-only."""

    paths: OnnxModelPaths = field(default_factory=OnnxModelPaths)
    disable_graph_optimizations: bool = True
    intra_op_num_threads: int | None = None
    inter_op_num_threads: int | None = None
    _sessions: dict[str, ort.InferenceSession] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.paths = self.paths.expanded()

    def validate_paths(self) -> dict[str, Path]:
        missing: list[str] = []
        resolved: dict[str, Path] = {}
        for name, path in self.paths.items():
            resolved[name] = path
            if not path.is_file():
                missing.append(f"{name}: {path}")
                continue
            data_path = path.with_suffix(path.suffix + ".data")
            if not data_path.is_file():
                missing.append(f"{name}_external_data: {data_path}")
        if missing:
            raise FileNotFoundError("Missing ONNX model files:\n" + "\n".join(missing))
        return resolved

    @property
    def created_session_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._sessions))

    @property
    def audio_encoder(self) -> ort.InferenceSession:
        return self._get("audio_encoder", self.paths.audio_encoder)

    @property
    def audio_decoder(self) -> ort.InferenceSession:
        return self._get("audio_decoder", self.paths.audio_decoder)

    @property
    def prefill(self) -> ort.InferenceSession:
        return self._get("prefill", self.paths.prefill)

    @property
    def decode_step(self) -> ort.InferenceSession:
        return self._get("decode_step", self.paths.decode_step)

    def _session_options(self) -> ort.SessionOptions:
        options = ort.SessionOptions()
        if self.disable_graph_optimizations:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if self.intra_op_num_threads is not None:
            options.intra_op_num_threads = self.intra_op_num_threads
        if self.inter_op_num_threads is not None:
            options.inter_op_num_threads = self.inter_op_num_threads
        return options

    def _get(self, name: str, path: Path) -> ort.InferenceSession:
        if name not in self._sessions:
            self._assert_path(name, path)
            session = ort.InferenceSession(
                str(path),
                sess_options=self._session_options(),
                providers=[CPU_PROVIDER],
            )
            self._assert_cpu_only(session, name)
            self._sessions[name] = session
        return self._sessions[name]

    @staticmethod
    def _assert_path(name: str, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{name} ONNX file not found: {path}")
        data_path = path.with_suffix(path.suffix + ".data")
        if not data_path.is_file():
            raise FileNotFoundError(f"{name} external data file not found: {data_path}")

    @staticmethod
    def _assert_cpu_only(session: ort.InferenceSession, name: str) -> None:
        providers = session.get_providers()
        if providers != [CPU_PROVIDER]:
            raise RuntimeError(f"{name} must use only {CPU_PROVIDER}, got {providers}")
        forbidden = FORBIDDEN_PROVIDERS.intersection(providers)
        if forbidden:
            raise RuntimeError(f"{name} loaded forbidden execution providers: {sorted(forbidden)}")
