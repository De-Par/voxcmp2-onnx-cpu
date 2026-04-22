# BF16 Compute Strategy

## Scope

This page defines the production BF16 compute path for VoxCPM2 ONNX CPU artifacts.

BF16 production means:

- weights and activations use BF16 where ONNX Runtime CPU has usable kernels
- FP32 islands are explicit and justified
- storage-only BF16 initializers followed by immediate `Cast` back to FLOAT are not a production success
- quality and performance are compared against FP32 ONNX and the official VoxCPM2 API

The shared implementation policy lives in `src/export/common.py`.

## Current Export Policy

The `bf16` precision profile is a real compute profile:

- `compute_dtype = bfloat16`
- `model_config_dtype = bfloat16`
- `storage_only = false`
- `production_compute = true`

Export wrappers keep the public host boundary as FP32 so the runtime path is shared by FP32 and BF16 artifacts. Inside the wrapper, floating inputs are cast once to the profile compute dtype. Floating outputs are cast back to FP32 at graph boundaries.

This boundary cast policy is different from the old storage-only experiment. The old experiment converted FLOAT initializers to BFLOAT16 and inserted `Cast(BFLOAT16 -> FLOAT)` before use. That pattern saves disk bytes but gives no BF16 compute benefit.

## Module Regions

### AudioVAEEncoder

BF16 compute candidates:

- encoder convolution/residual stack

FP32 islands:

- none currently forced by wrapper code

Boundary casts:

- `waveform`: FP32 host tensor -> BF16 compute
- `latent`: BF16 compute -> FP32 host tensor

### AudioVAEDecoder

BF16 compute candidates:

- decoder convolution/residual stack

FP32 islands:

- none currently forced by wrapper code

Boundary casts:

- `latent`: FP32 host tensor -> BF16 compute
- `waveform`: BF16 compute -> FP32 host tensor

### VoxCPM2Prefill

BF16 compute candidates:

- feature encoder
- text embedding
- base LM prefill
- FSQ/fusion projection
- residual LM prefill

FP32 islands:

- none currently forced by wrapper code

Boundary casts:

- text/audio masks and audio features: FP32 host tensors -> BF16 compute
- hidden states and cache tensors: BF16 compute -> FP32 host tensors

### VoxCPM2DecodeStep

BF16 compute candidates:

- DiT conditioning projections
- LocDiT/CFM solve
- feature encoder
- base LM decode step
- residual LM decode step
- stop head

FP32 islands:

- rotary position embedding multiply/add

The rotary island is kept in FP32 because the operation is small, numerically sensitive, and already isolated. It casts back to the active compute dtype immediately after the rotary multiply/add.

Boundary casts:

- hidden/cache/noise/cfg inputs: FP32 host tensors -> BF16 compute
- generated features, hidden states, cache updates: BF16 compute -> FP32 host tensors

## Forbidden Pattern

Production BF16 must not rely on:

```text
BF16 initializer -> Cast to FLOAT -> FLOAT compute
```

If an op must remain FP32, keep that op's tensors/weights as FP32 and document the island. Do not store weights as BF16 only to cast them back before every use.

The legacy feasibility tool now writes storage-only artifacts under `artifacts/experiments/bf16_storage_only` by default and refuses to write into `models/onnx/bf16`.

## Verification

Static policy tests:

```bash
python -B -m pytest tests/parity/test_bf16_compute_path.py
```

After exporting BF16 artifacts, inspect them for storage-only cast-back patterns:

```bash
VOXCPM2_BF16_ONNX_PATHS="models/onnx/bf16/audio_vae_encoder/audio_vae_encoder.onnx:models/onnx/bf16/audio_vae_decoder/audio_vae_decoder.onnx:models/onnx/bf16/prefill/voxcpm2_prefill.onnx:models/onnx/bf16/decode_step/voxcpm2_decode_step.onnx" \
  python -B -m pytest tests/parity/test_bf16_compute_path.py
```

Run module parity with BF16 wrappers:

```bash
python -B tests/parity/test_decode_step.py \
  --onnx-path models/onnx/bf16/decode_step/voxcpm2_decode_step.onnx \
  --precision bf16 \
  --cache-seq 16 \
  --max-cache-seq 64
```

Then run benchmark/profiling:

```bash
python -B tools/bench/run_benchmarks.py \
  --output-dir artifacts/perf_baseline_bf16_compute \
  --json-report artifacts/perf_baseline_bf16_compute/baseline.json \
  --markdown-report artifacts/perf_baseline_bf16_compute/baseline.md \
  --variants official onnx
```

## Acceptance Criteria

- BF16 artifacts are produced by `src/export/export_all.py --precision bf16`.
- BF16 model weights are exported from BF16 compute wrappers, not storage-only conversion.
- Graph inspection finds no BF16-initializer-to-FLOAT cast-back regions.
- FP32 islands are listed in this document and in export metadata.
- BF16 parity passes with explicit tolerances.
- BF16 benchmark/profiling shows Cast churn is not the dominant bottleneck.
