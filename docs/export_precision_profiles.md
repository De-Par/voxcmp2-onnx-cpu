# Export Precision Profiles

## Scope

This page defines the unified production export pipeline for VoxCPM2 CPU-only ONNX artifacts.

Supported precision profiles:

- `fp32`
- `bf16`

Both profiles use the same module boundaries:

- `AudioVAEEncoder`
- `AudioVAEDecoder`
- `VoxCPM2Prefill`
- `VoxCPM2DecodeStep`

The shared implementation entry point for precision policy and public contracts is `src/export/common.py`.

## Production Layout

```text
models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
models/onnx/fp32/prefill/voxcpm2_prefill.onnx
models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx

models/onnx/bf16/audio_vae_encoder/audio_vae_encoder.onnx
models/onnx/bf16/audio_vae_decoder/audio_vae_decoder.onnx
models/onnx/bf16/prefill/voxcpm2_prefill.onnx
models/onnx/bf16/decode_step/voxcpm2_decode_step.onnx
```

Large external-data files must stay next to their `.onnx` files.

## Shared Contract

FP32 and BF16 exports must keep the same public graph contract:

- same module names
- same input names
- same output names
- same ranks
- same dynamic axes
- same prefill/decode state tensor semantics
- same feature coverage for text-only, voice design, controllable clone, and ultimate clone

Both profiles are exported by the same wrapper code paths. Precision policy may change dtype attributes and intentional boundary casts, but it must not introduce a separate module architecture. Any unexpected topology divergence found in ONNX graph inspection is a blocker for production BF16 promotion.

Host-visible floating tensors remain `float32` for both profiles in the current production runtime contract. This keeps the runtime path identical while allowing the export wrapper to change internal compute dtype.

## Precision Profiles

### `fp32`

- model compute dtype: `float32`
- host-visible floating dtype: `float32`
- role: correctness anchor

### `bf16`

- model compute dtype: `bfloat16`
- host-visible floating dtype: `float32`
- role: production BF16 target

BF16 is not the old storage-only experiment. The production BF16 export path must minimize `Cast` and `CastLike` churn. Boundary casts caused by the shared FP32 host-visible contract are allowed, but they must remain explicit and measurable.

Mixed precision is allowed only for:

- correctness
- numerical stability
- ONNX Runtime CPU kernel coverage
- the documented host-visible boundary contract

Current intentional mixed-precision sites:

- graph boundaries cast host-visible FP32 floating tensors to the profile compute dtype and cast floating outputs back to FP32
- `VoxCPM2DecodeStep` rotary position embedding keeps the rotary multiply/add in FP32, then casts back to the active compute dtype

## Commands

Export all modules for one precision profile:

```bash
python -B src/export/export_all.py --precision fp32
python -B src/export/export_all.py --precision bf16
```

Equivalent console scripts after editable install:

```bash
voxcpm2-export --precision fp32
voxcpm2-export --precision bf16
```

Export one module:

```bash
python -B src/export/export_audio_vae_encoder.py --precision fp32
python -B src/export/export_audio_vae_decoder.py --precision fp32
python -B src/export/export_prefill.py --precision fp32 --mode plain_tts
python -B src/export/export_decode_step.py --precision fp32 --cache-seq 16
```

Use `--precision bf16` for the BF16 artifact family. If `--output` is omitted, each script writes to `models/onnx/<precision>/<module>/...`.

## Non-Goals

- Separate BF16 architecture.
- Separate BF16 runtime semantics.
- Monolithic ONNX export.
- Quantization.
- GPU, CoreML, CUDA, DirectML, or accelerator providers.

## Acceptance Criteria

- `src/export/common.py` is the single source for precision profiles, default artifact layout, and public module contracts.
- Every exporter accepts `--precision {fp32,bf16}`.
- `export_all.py` can export one complete artifact family with a single command.
- FP32 and BF16 public graph contracts match.
- Any BF16 blocker is documented before changing model math.
