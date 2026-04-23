"""Lazy ONNX Runtime CPU session factory for VoxCPM2 modules"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_MODELS_ROOT = REPO_ROOT / "models" / "onnx"
CPU_PROVIDER = "CPUExecutionProvider"
FORBIDDEN_PROVIDERS = {"CUDAExecutionProvider", "CoreMLExecutionProvider", "MPSExecutionProvider"}
GRAPH_OPTIMIZATION_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
EXECUTION_MODES = {
    "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
    "parallel": ort.ExecutionMode.ORT_PARALLEL,
}
LOG_SEVERITY_LEVELS = {
    "verbose": 0,
    "info": 1,
    "warning": 2,
    "error": 3,
    "fatal": 4,
}
GRAPH_OPTIMIZATION_CHOICES = tuple(GRAPH_OPTIMIZATION_LEVELS)
EXECUTION_MODE_CHOICES = tuple(EXECUTION_MODES)
LOG_SEVERITY_CHOICES = tuple(LOG_SEVERITY_LEVELS)
ArtifactKind = Literal["ort", "optimized_onnx", "onnx"]


@dataclass(frozen=True)
class OnnxModelPaths:
    audio_encoder: Path = ONNX_MODELS_ROOT / "fp32" / "audio_vae_encoder" / "audio_vae_encoder.onnx"
    audio_decoder: Path = ONNX_MODELS_ROOT / "fp32" / "audio_vae_decoder" / "audio_vae_decoder.onnx"
    prefill: Path = ONNX_MODELS_ROOT / "fp32" / "prefill" / "voxcpm2_prefill.onnx"
    decode_chunk: Path = ONNX_MODELS_ROOT / "fp32" / "decode_chunk" / "voxcpm2_decode_chunk.onnx"

    def expanded(self) -> "OnnxModelPaths":
        return OnnxModelPaths(
            audio_encoder=self.audio_encoder.expanduser().resolve(),
            audio_decoder=self.audio_decoder.expanduser().resolve(),
            prefill=self.prefill.expanduser().resolve(),
            decode_chunk=self.decode_chunk.expanduser().resolve(),
        )

    def items(self) -> Iterable[tuple[str, Path]]:
        yield "audio_encoder", self.audio_encoder
        yield "audio_decoder", self.audio_decoder
        yield "prefill", self.prefill
        yield "decode_chunk", self.decode_chunk


@dataclass
class OrtSessionFactory:
    """Create ONNX Runtime sessions lazily and CPU-only"""

    paths: OnnxModelPaths = field(default_factory=OnnxModelPaths)
    disable_graph_optimizations: bool | None = None
    graph_optimization_level: str = "all"
    execution_mode: str = "sequential"
    log_severity_level: str = "error"
    intra_op_num_threads: int | None = 8
    inter_op_num_threads: int | None = 1
    enable_mem_pattern: bool | None = True
    enable_cpu_mem_arena: bool | None = True
    enable_mem_reuse: bool | None = True
    enable_profiling: bool = False
    profile_file_prefix: Path | None = None
    prefer_ort_format: bool = True
    prefer_optimized_onnx: bool = False
    _sessions: dict[str, ort.InferenceSession] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.paths = self.paths.expanded()

    def validate_paths(self) -> dict[str, Path]:
        # External-data ONNX exports store large weights next to the .onnx file.
        # Validate both files before creating sessions so startup errors are
        # actionable and do not depend on ONNX Runtime's lower-level messages.
        missing: list[str] = []
        resolved: dict[str, Path] = {}
        for name, path in self.paths.items():
            artifact_path, _ = self._preferred_artifact(path)
            resolved[name] = artifact_path
            if not artifact_path.is_file():
                missing.append(f"{name}: {artifact_path}")
                continue
            data_path = self._external_data_path(artifact_path)
            if data_path is not None and not data_path.is_file():
                missing.append(f"{name}_external_data: {data_path}")
        if missing:
            raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))
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
    def decode_chunk(self) -> ort.InferenceSession:
        return self._get("decode_chunk", self.paths.decode_chunk)

    def _session_options(
        self, session_name: str | None = None, *, artifact_kind: ArtifactKind = "onnx"
    ) -> ort.SessionOptions:
        options = ort.SessionOptions()
        graph_optimization_level = self._graph_optimization_level_for_artifact(artifact_kind)
        try:
            options.graph_optimization_level = GRAPH_OPTIMIZATION_LEVELS[graph_optimization_level]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported ORT graph optimization level {graph_optimization_level!r}; "
                f"expected one of {GRAPH_OPTIMIZATION_CHOICES}"
            ) from exc
        try:
            options.execution_mode = EXECUTION_MODES[self.execution_mode]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported ORT execution mode {self.execution_mode!r}; expected one of {EXECUTION_MODE_CHOICES}"
            ) from exc
        try:
            options.log_severity_level = LOG_SEVERITY_LEVELS[self.log_severity_level]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported ORT log severity {self.log_severity_level!r}; expected one of {LOG_SEVERITY_CHOICES}"
            ) from exc
        self._set_thread_option(options, "intra_op_num_threads", self.intra_op_num_threads)
        self._set_thread_option(options, "inter_op_num_threads", self.inter_op_num_threads)
        self._set_optional_bool_option(options, "enable_mem_pattern", self.enable_mem_pattern)
        self._set_optional_bool_option(options, "enable_cpu_mem_arena", self.enable_cpu_mem_arena)
        self._set_optional_bool_option(options, "enable_mem_reuse", self.enable_mem_reuse)
        if self.enable_profiling:
            options.enable_profiling = True
            if self.profile_file_prefix is not None:
                profile_prefix = self._profile_prefix_for_session(session_name or "session")
                profile_prefix.parent.mkdir(parents=True, exist_ok=True)
                options.profile_file_prefix = str(profile_prefix)
        return options

    def options_summary(self) -> dict[str, bool | int | str | None]:
        return {
            "provider": CPU_PROVIDER,
            "graph_optimization_level": self._resolved_graph_optimization_level(),
            "execution_mode": self.execution_mode,
            "log_severity_level": self.log_severity_level,
            "intra_op_num_threads": self.intra_op_num_threads,
            "inter_op_num_threads": self.inter_op_num_threads,
            "enable_mem_pattern": self.enable_mem_pattern,
            "enable_cpu_mem_arena": self.enable_cpu_mem_arena,
            "enable_mem_reuse": self.enable_mem_reuse,
            "enable_profiling": self.enable_profiling,
            "profile_file_prefix": str(self.profile_file_prefix) if self.profile_file_prefix else None,
            "prefer_ort_format": self.prefer_ort_format,
            "prefer_optimized_onnx": self.prefer_optimized_onnx,
        }

    def end_profiling(self) -> dict[str, Path]:
        profile_paths: dict[str, Path] = {}
        for name, session in self._sessions.items():
            profile_path = session.end_profiling()
            if profile_path:
                profile_paths[name] = Path(profile_path).expanduser().resolve()
        return profile_paths

    def _resolved_graph_optimization_level(self) -> str:
        # Backward-compatible shim for older callers that only passed the
        # previous boolean flag. New code should use graph_optimization_level.
        if self.disable_graph_optimizations is True:
            return "disable"
        if self.disable_graph_optimizations is False:
            return "all"
        return self.graph_optimization_level

    def _graph_optimization_level_for_artifact(self, artifact_kind: ArtifactKind) -> str:
        resolved = self._resolved_graph_optimization_level()
        if artifact_kind in {"ort", "optimized_onnx"} and resolved != "disable":
            # ORT and optimized ONNX artifacts already bake graph rewrites
            # offline. Re-running them at load time only adds session creation
            # cost and may duplicate work on every process start.
            return "disable"
        return resolved

    @staticmethod
    def _set_thread_option(options: ort.SessionOptions, name: str, value: int | None) -> None:
        if value is None:
            return
        if value < 0:
            raise ValueError(f"{name} must be >= 0; use 0 or omit it for ORT default scheduling")
        setattr(options, name, value)

    @staticmethod
    def _set_optional_bool_option(options: ort.SessionOptions, name: str, value: bool | None) -> None:
        if value is None:
            return
        if not hasattr(options, name):
            raise RuntimeError(f"Installed ONNX Runtime does not expose SessionOptions.{name}")
        setattr(options, name, bool(value))

    def _profile_prefix_for_session(self, session_name: str) -> Path:
        assert self.profile_file_prefix is not None
        expanded = self.profile_file_prefix.expanduser()
        if expanded.suffix:
            return expanded.with_name(f"{expanded.stem}_{session_name}{expanded.suffix}")
        return expanded / f"ort_profile_{session_name}"

    def _get(self, name: str, path: Path) -> ort.InferenceSession:
        if name not in self._sessions:
            session = self._load_session(name, path)
            self._assert_cpu_only(session, name)
            self._sessions[name] = session
        return self._sessions[name]

    @staticmethod
    def _external_data_path(path: Path) -> Path | None:
        if path.suffix != ".onnx":
            return None
        return path.with_suffix(path.suffix + ".data")

    def _optimized_onnx_path(self, path: Path) -> Path:
        if path.suffix != ".onnx":
            return path
        return path.with_name(f"{path.stem}.optimized{path.suffix}")

    def _artifact_candidates(self, path: Path) -> list[tuple[Path, ArtifactKind]]:
        candidates: list[tuple[Path, ArtifactKind]] = []
        if path.suffix == ".onnx":
            if self.prefer_ort_format:
                ort_path = path.with_suffix(".ort")
                if self._is_usable_ort_file(ort_path):
                    candidates.append((ort_path, "ort"))
            if self.prefer_optimized_onnx:
                optimized_path = self._optimized_onnx_path(path)
                if self._is_usable_file(optimized_path):
                    candidates.append((optimized_path, "optimized_onnx"))
        candidates.append((path, "onnx"))
        return candidates

    def _preferred_artifact(self, path: Path) -> tuple[Path, ArtifactKind]:
        return self._artifact_candidates(path)[0]

    @staticmethod
    def _is_usable_ort_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    @staticmethod
    def _is_usable_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    def _load_session(self, name: str, original_path: Path) -> ort.InferenceSession:
        candidates = self._artifact_candidates(original_path)
        last_error: Exception | None = None
        for index, (candidate_path, artifact_kind) in enumerate(candidates):
            self._assert_path(name, candidate_path)
            try:
                return self._create_session(candidate_path, name, artifact_kind=artifact_kind)
            except Exception as exc:  # noqa: BLE001 - ORT raises multiple load error types.
                last_error = exc
                if index == len(candidates) - 1:
                    raise
                next_path, _ = candidates[index + 1]
                warnings.warn(
                    f"{name}: failed to load preferred artifact {candidate_path.name}; "
                    f"falling back to {next_path.name}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        assert last_error is not None
        raise last_error

    def _create_session(self, path: Path, session_name: str, *, artifact_kind: ArtifactKind) -> ort.InferenceSession:
        # Providers are passed explicitly to avoid implicit accelerator
        # fallback on machines that happen to have GPU/CoreML packages.
        return ort.InferenceSession(
            str(path),
            sess_options=self._session_options(session_name, artifact_kind=artifact_kind),
            providers=[CPU_PROVIDER],
        )

    def _assert_path(self, name: str, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{name} model file not found: {path}")
        data_path = self._external_data_path(path)
        if data_path is not None and not data_path.is_file():
            raise FileNotFoundError(f"{name} external data file not found: {data_path}")

    @staticmethod
    def _assert_cpu_only(session: ort.InferenceSession, name: str) -> None:
        providers = session.get_providers()
        if providers != [CPU_PROVIDER]:
            raise RuntimeError(f"{name} must use only {CPU_PROVIDER}, got {providers}")
        forbidden = FORBIDDEN_PROVIDERS.intersection(providers)
        if forbidden:
            raise RuntimeError(f"{name} loaded forbidden execution providers: {sorted(forbidden)}")
