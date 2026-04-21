#!/usr/bin/env python3
"""Trace the official VoxCPM2 generate path without logging tensor contents."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections.abc import Generator as GeneratorABC
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SRC = REPO_ROOT / "third_party" / "VoxCPM" / "src"


def _install_upstream_import_path() -> None:
    if not UPSTREAM_SRC.exists():
        raise FileNotFoundError(f"missing upstream VoxCPM source: {UPSTREAM_SRC}")
    sys.path.insert(0, str(UPSTREAM_SRC))


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _shape(value: Any) -> list[int | str]:
    try:
        return [int(dim) for dim in value.shape]
    except Exception:
        return [str(value.shape)]


def _dtype(value: Any) -> str:
    try:
        return str(value.dtype)
    except Exception:
        return type(value).__name__


def _device(value: Any) -> str | None:
    device = getattr(value, "device", None)
    return str(device) if device is not None else None


def _summarize(value: Any, *, depth: int = 0) -> Any:
    if value is None:
        return None
    if _is_tensor_like(value):
        summary = {
            "type": type(value).__name__,
            "shape": _shape(value),
            "dtype": _dtype(value),
        }
        device = _device(value)
        if device is not None:
            summary["device"] = device
        return summary
    if isinstance(value, (str, Path)):
        return {
            "type": type(value).__name__,
            "present": bool(str(value)),
            "length": len(str(value)),
        }
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        if depth >= 2:
            return {"type": "dict", "keys": sorted(map(str, value.keys()))}
        return {
            "type": "dict",
            "keys": sorted(map(str, value.keys())),
            "items": {str(k): _summarize(v, depth=depth + 1) for k, v in value.items()},
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "len": len(value),
            "items": [_summarize(v, depth=depth + 1) for v in value[:8]],
        }
    if isinstance(value, list):
        items: list[Any] = []
        if value:
            items.append({"index": 0, "value": _summarize(value[0], depth=depth + 1)})
        if len(value) > 1:
            items.append({"index": len(value) - 1, "value": _summarize(value[-1], depth=depth + 1)})
        return {"type": "list", "len": len(value), "sample": items}
    return {"type": type(value).__name__}


def _summarize_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "args": [_summarize(arg) for arg in args],
        "kwargs": {key: _summarize(value) for key, value in kwargs.items()},
    }


def _pathways(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cli_mode": args.mode,
        "has_reference": bool(args.reference_wav_path),
        "has_prompt": bool(args.prompt_wav_path or args.prompt_text),
        "has_prompt_audio": bool(args.prompt_wav_path),
        "has_prompt_text": bool(args.prompt_text),
    }


class TraceLogger:
    def __init__(self, output_path: Path, pathways: dict[str, Any]) -> None:
        self.output_path = output_path
        self.pathways = pathways
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = output_path.open("w", encoding="utf-8")
        self._seq = 0

    def close(self) -> None:
        self._fh.close()

    def event(
        self,
        stage: str,
        function: Callable[..., Any],
        *,
        event: str,
        inputs: Any = None,
        outputs: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "seq": self._seq,
            "event": event,
            "stage": stage,
            "python_module": getattr(function, "__module__", None),
            "python_function": getattr(function, "__qualname__", getattr(function, "__name__", None)),
            "pathway": self.pathways,
            "inputs": inputs,
            "outputs": outputs,
            "extra": extra or {},
        }
        self._seq += 1
        self._fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()


def _wrap_callable(logger: TraceLogger, owner: Any, attr: str, stage: str) -> Callable[[], None]:
    original = getattr(owner, attr)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        logger.event(
            stage,
            original,
            event="call",
            inputs=_summarize_call(args[1:] if args and args[0].__class__ is owner else args, kwargs),
        )
        result = original(*args, **kwargs)
        if inspect.isgenerator(result) or isinstance(result, GeneratorABC):
            return _trace_generator(logger, stage, original, result, started)
        logger.event(
            stage,
            original,
            event="return",
            outputs=_summarize(result),
            extra={"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
        return result

    wrapper.__name__ = getattr(original, "__name__", attr)
    wrapper.__qualname__ = getattr(original, "__qualname__", attr)
    wrapper.__doc__ = getattr(original, "__doc__", None)
    setattr(owner, attr, wrapper)

    def restore() -> None:
        setattr(owner, attr, original)

    return restore


def _trace_generator(
    logger: TraceLogger,
    stage: str,
    original: Callable[..., Any],
    result: Any,
    started: float,
) -> Any:
    try:
        for index, item in enumerate(result):
            logger.event(
                stage,
                original,
                event="yield",
                outputs=_summarize(item),
                extra={"yield_index": index},
            )
            yield item
    finally:
        logger.event(
            stage,
            original,
            event="close",
            extra={"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
        )


def _wrap_bound_method(logger: TraceLogger, instance: Any, attr: str, stage: str) -> Callable[[], None]:
    original = getattr(instance, attr)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        logger.event(stage, original, event="call", inputs=_summarize_call(args, kwargs))
        result = original(*args, **kwargs)
        if inspect.isgenerator(result) or isinstance(result, GeneratorABC):
            return _trace_generator(logger, stage, original, result, started)
        logger.event(
            stage,
            original,
            event="return",
            outputs=_summarize(result),
            extra={"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
        return result

    setattr(instance, attr, wrapper)

    def restore() -> None:
        setattr(instance, attr, original)

    return restore


@contextmanager
def _trace_official_classes(logger: TraceLogger):
    from voxcpm.core import VoxCPM
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    restores = [
        _wrap_callable(logger, VoxCPM, "generate", "host.generate"),
        _wrap_callable(logger, VoxCPM, "_generate", "host._generate"),
        _wrap_callable(logger, VoxCPM2Model, "build_prompt_cache", "prompt_cache.build"),
        _wrap_callable(logger, VoxCPM2Model, "_generate_with_prompt_cache", "model.generate_with_prompt_cache"),
        _wrap_callable(logger, VoxCPM2Model, "_inference", "model.inference_loop"),
        _wrap_callable(logger, VoxCPM2Model, "_encode_wav", "audio.encode_wav"),
        _wrap_callable(logger, VoxCPM2Model, "_make_ref_prefix", "prompt.reference_prefix"),
    ]
    try:
        yield
    finally:
        for restore in reversed(restores):
            restore()


def _trace_loaded_modules(logger: TraceLogger, model: Any) -> list[Callable[[], None]]:
    tts = model.tts_model
    restores: list[Callable[[], None]] = []
    targets = [
        (tts.audio_vae, "encode", "neural.audio_vae.encode"),
        (tts.audio_vae, "decode", "neural.audio_vae.decode"),
        (tts.feat_encoder, "forward", "neural.locenc.forward"),
        (tts.base_lm, "forward", "neural.base_lm.prefill"),
        (tts.base_lm, "forward_step", "neural.base_lm.step"),
        (tts.residual_lm, "forward", "neural.residual_lm.prefill"),
        (tts.residual_lm, "forward_step", "neural.residual_lm.step"),
        (tts.fsq_layer, "forward", "neural.scalar_quantization"),
        (tts.feat_decoder, "forward", "neural.locdit_cfm.sample"),
        (tts.feat_decoder.estimator, "forward", "neural.locdit_estimator.forward"),
        (tts.enc_to_lm_proj, "forward", "neural.proj.enc_to_lm"),
        (tts.lm_to_dit_proj, "forward", "neural.proj.lm_to_dit"),
        (tts.res_to_dit_proj, "forward", "neural.proj.res_to_dit"),
        (tts.fusion_concat_proj, "forward", "neural.proj.fusion_concat"),
        (tts.stop_proj, "forward", "neural.stop.proj"),
        (tts.stop_head, "forward", "neural.stop.head"),
    ]
    for instance, attr, stage in targets:
        if hasattr(instance, attr):
            restores.append(_wrap_bound_method(logger, instance, attr, stage))
    return restores


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode in {"plain_tts", "voice_design"}:
        if args.reference_wav_path or args.prompt_wav_path or args.prompt_text:
            raise ValueError(f"{args.mode} must not use reference or prompt inputs")
    elif args.mode == "controllable_clone":
        if not args.reference_wav_path:
            raise ValueError("controllable_clone requires --reference-wav-path")
        if args.prompt_wav_path or args.prompt_text:
            raise ValueError("controllable_clone must not use prompt inputs")
    elif args.mode == "ultimate_clone":
        if not args.prompt_wav_path or not args.prompt_text:
            raise ValueError("ultimate_clone requires --prompt-wav-path and --prompt-text")


def _run(args: argparse.Namespace) -> None:
    _validate_args(args)
    _install_upstream_import_path()

    from voxcpm import VoxCPM

    logger = TraceLogger(args.trace_output, _pathways(args))
    try:
        with _trace_official_classes(logger):
            model = VoxCPM.from_pretrained(
                args.model_path,
                load_denoiser=False,
                local_files_only=args.local_files_only,
                optimize=False,
                device=args.device,
            )
            restores = _trace_loaded_modules(logger, model)
            try:
                wav = model.generate(
                    text=args.text,
                    prompt_wav_path=args.prompt_wav_path,
                    prompt_text=args.prompt_text,
                    reference_wav_path=args.reference_wav_path,
                    cfg_value=args.cfg_value,
                    inference_timesteps=args.inference_timesteps,
                    min_len=args.min_len,
                    max_len=args.max_len,
                    normalize=args.normalize,
                    denoise=False,
                    retry_badcase=args.retry_badcase,
                    retry_badcase_max_times=args.retry_badcase_max_times,
                    retry_badcase_ratio_threshold=args.retry_badcase_ratio_threshold,
                )
                logger.event(
                    "trace.result",
                    _run,
                    event="return",
                    outputs=_summarize(wav),
                    extra={"sample_rate": getattr(model.tts_model, "sample_rate", None)},
                )
                if args.wav_output is not None:
                    import soundfile as sf

                    sf.write(str(args.wav_output), wav, model.tts_model.sample_rate)
            finally:
                for restore in reversed(restores):
                    restore()
    finally:
        logger.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace the official VoxCPM2 generate() path as compact JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python -B src/parity/trace_generate.py --model-path openbmb/VoxCPM2 "
            "--mode plain_tts --text 'Hello.' --trace-output traces/plain_tts.jsonl"
        ),
    )
    parser.add_argument("--model-path", required=True, help="Local VoxCPM2 model directory or HF id.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["plain_tts", "voice_design", "controllable_clone", "ultimate_clone"],
        help="Official generate pathway to validate and trace.",
    )
    parser.add_argument("--text", required=True, help="Target text passed to official VoxCPM.generate().")
    parser.add_argument("--reference-wav-path", help="Reference WAV path for controllable_clone or optional ultimate_clone reference prefix.")
    parser.add_argument("--prompt-wav-path", help="Prompt WAV path required by ultimate_clone.")
    parser.add_argument("--prompt-text", help="Prompt text required by ultimate_clone.")
    parser.add_argument("--trace-output", type=Path, default=Path("trace_generate.jsonl"), help="JSONL trace output path.")
    parser.add_argument("--wav-output", type=Path, help="Optional synthesized WAV output path for manual listening.")
    parser.add_argument("--device", default="cpu", help="Device used by the official PyTorch model while tracing.")
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Require local Hugging Face cache/model files.")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only", help="Allow from_pretrained to fetch missing files.")
    parser.add_argument("--normalize", action="store_true", help="Enable official text normalization inside VoxCPM.generate().")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance value for official generation.")
    parser.add_argument("--inference-timesteps", type=int, default=10, help="Official CFM/LocDiT solver step count.")
    parser.add_argument("--min-len", type=int, default=2, help="Minimum generated audio-feature steps before stopping.")
    parser.add_argument("--max-len", type=int, default=4096, help="Maximum generated audio-feature steps.")
    parser.add_argument("--retry-badcase", action="store_true", help="Enable official retry_badcase behavior.")
    parser.add_argument("--retry-badcase-max-times", type=int, default=3, help="Maximum official retry_badcase attempts.")
    parser.add_argument("--retry-badcase-ratio-threshold", type=float, default=6.0, help="Official retry_badcase ratio threshold.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
