#!/usr/bin/env python3
"""Export all VoxCPM2 ONNX modules for one production precision profile"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import add_precision_argument, get_precision_profile, output_path_under_root
    from .export_audio_vae_decoder import export_audio_vae_decoder
    from .export_audio_vae_encoder import export_audio_vae_encoder
    from .export_decode_chunk import export_decode_chunk
    from .export_prefill import export_prefill
except ImportError:
    from common import add_precision_argument, get_precision_profile, output_path_under_root  # type: ignore[no-redef]
    from export_audio_vae_decoder import export_audio_vae_decoder  # type: ignore[no-redef]
    from export_audio_vae_encoder import export_audio_vae_encoder  # type: ignore[no-redef]
    from export_decode_chunk import export_decode_chunk  # type: ignore[no-redef]
    from export_prefill import export_prefill  # type: ignore[no-redef]


def _module_output(output_root: Path, module_key: str, precision_name: str) -> Path:
    return output_path_under_root(output_root, module_key, get_precision_profile(precision_name))


def export_all(args: argparse.Namespace) -> None:
    """Run the four standalone exporters with one shared precision profile"""

    precision = get_precision_profile(args.precision)
    print(f"export_profile={precision.name}")
    print(f"output_root={args.output_root.expanduser()}")

    common = {
        "model_path": args.model_path,
        "local_files_only": args.local_files_only,
        "batch_size": args.batch_size,
        "opset": args.opset,
        "precision": precision.name,
    }

    export_audio_vae_encoder(
        argparse.Namespace(
            **common,
            output=_module_output(args.output_root, "audio_vae_encoder", precision.name),
            samples=args.samples,
        )
    )
    export_audio_vae_decoder(
        argparse.Namespace(
            **common,
            output=_module_output(args.output_root, "audio_vae_decoder", precision.name),
            latent_steps=args.latent_steps,
        )
    )
    export_prefill(
        argparse.Namespace(
            **common,
            output=_module_output(args.output_root, "prefill", precision.name),
            seq_len=args.seq_len,
            mode=args.mode,
            reference_steps=args.reference_steps,
            prompt_steps=args.prompt_steps,
            seed=args.seed,
        )
    )
    export_decode_chunk(
        argparse.Namespace(
            **common,
            output=_module_output(args.output_root, "decode_chunk", precision.name),
            chunk_size=args.chunk_size,
            current_length=args.current_length,
            max_cache_seq=args.max_cache_seq,
            inference_timesteps=args.inference_timesteps,
            cfg_value=args.cfg_value,
            seed=args.seed,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export all VoxCPM2 ONNX module boundaries for one precision profile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python -B src/export/export_all.py --precision fp32",
    )
    parser.add_argument(
        "--model-path", default="openbmb/VoxCPM2", help="Local VoxCPM2 model directory or Hugging Face id."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/onnx"),
        help="Root directory that receives <precision>/<module>/<file>.onnx outputs.",
    )
    add_precision_argument(parser)
    parser.add_argument("--batch-size", type=int, default=1, help="Example batch dimension used during export.")
    parser.add_argument("--samples", type=int, default=20480, help="AudioVAE encoder example padded samples.")
    parser.add_argument("--latent-steps", type=int, default=4, help="AudioVAE decoder example latent steps.")
    parser.add_argument("--seq-len", type=int, default=16, help="Prefill example full prompt sequence length.")
    parser.add_argument(
        "--mode",
        choices=["plain_tts", "voice_design", "controllable_clone", "ultimate_clone"],
        default="plain_tts",
        help="Synthetic prefill layout used during export.",
    )
    parser.add_argument("--reference-steps", type=int, default=3, help="Synthetic reference-audio steps.")
    parser.add_argument("--prompt-steps", type=int, default=3, help="Synthetic prompt-audio steps.")
    parser.add_argument("--chunk-size", type=int, default=4, help="Decode steps per production ONNX session.run.")
    parser.add_argument("--current-length", type=int, default=16, help="Decode-chunk example valid KV-cache length.")
    parser.add_argument(
        "--max-cache-seq",
        type=int,
        default=64,
        help="Decode-chunk example fixed KV-cache capacity; must be at least --current-length + --chunk-size.",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Fixed CFM/LocDiT solver steps embedded in each internal decode step.",
    )
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Example classifier-free guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="PyTorch RNG seed for synthetic export inputs.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version for torch.onnx.export.")
    parser.add_argument(
        "--local-files-only", action="store_true", default=True, help="Require local Hugging Face cache/model files."
    )
    parser.add_argument(
        "--allow-download",
        action="store_false",
        dest="local_files_only",
        help="Allow snapshot_download to fetch missing files.",
    )
    return parser


def main() -> int:
    export_all(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
