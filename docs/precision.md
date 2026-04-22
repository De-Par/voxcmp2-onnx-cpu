# Precision Strategy

This page defines the production FP32/BF16 policy for VoxCPM2 CPU-only ONNX artifacts.

## Production Targets

The project has two production artifact families:

- FP32 ONNX artifacts
- BF16 ONNX artifacts

FP32 is the correctness anchor. BF16 is a parallel production target, not a replacement for FP32 and not a storage-only final state.

Both families must use:

- the same module boundaries
- the same input/output names
- the same host-visible ranks and dynamic axes
- the same fixed-capacity cache/state contract
- the same runtime path
- the same feature coverage
- the same benchmark and quality case matrix

## Precision Profiles

The profile registry is in `src/export/common.py`.

| profile | compute dtype | host-visible float dtype | role |
|---|---|---|---|
| `fp32` | `float32` | `float32` | correctness anchor |
| `bf16` | `bfloat16` | `float32` | production BF16 compute target |

Host-visible floating tensors remain FP32 for both profiles. This keeps the runtime path identical and localizes precision decisions inside export wrappers and graph metadata.

## BF16 Compute Policy

Production BF16 must:

- use BF16 weights and activations where ONNX Runtime CPU has correct and performant kernels
- keep FP32 islands explicit and justified
- minimize BF16-to-FP32 and FP32-to-BF16 casts
- keep feature coverage identical to FP32
- compare quality/performance against FP32 ONNX and the official VoxCPM2 API

Allowed mixed-precision reasons:

- no usable ORT CPU BF16 kernel for an op on target platforms
- numerical instability exceeds accepted tolerance
- the shared FP32 host-visible boundary requires an explicit cast

Forbidden as production BF16:

```text
BF16 initializer -> Cast to FLOAT -> FLOAT compute
```

That pattern is storage-only and gives no BF16 compute benefit.

## Module BF16 Regions

| module | BF16 compute candidates | FP32 islands | boundary casts |
|---|---|---|---|
| `AudioVAEEncoder` | encoder convolution/residual stack | none forced | waveform FP32 -> BF16, latent BF16 -> FP32 |
| `AudioVAEDecoder` | decoder convolution/residual stack | none forced | latent FP32 -> BF16, waveform BF16 -> FP32 |
| `VoxCPM2Prefill` | feature encoder, text embedding, base LM, FSQ/fusion, residual LM | none forced | masks/audio FP32 -> BF16, hidden/cache BF16 -> FP32 |
| `VoxCPM2DecodeStep` | DiT projections, LocDiT/CFM, feature encoder, base/residual LM, stop head | rotary position embedding multiply/add | hidden/cache/noise/cfg FP32 -> BF16, outputs BF16 -> FP32 |

The rotary island is intentionally small and isolated. It casts q/k/cos/sin to FP32 for rotary multiply/add and casts q/k back to the active compute dtype.

## Dtype Cleanup Policy

No-op casts are not allowed in production exports. Export wrappers must skip dtype conversion when the tensor already has the target dtype.

Cleanup classification:

| class | policy |
|---|---|
| redundant cast chains | remove when they come from no-op wrapper casts or same-dtype `Cast -> Cast` edges |
| FP32/BF16 ping-pong | production BF16 blocker unless it is a documented FP32 island |
| unavoidable boundaries | allowed only at shared FP32 host-visible graph boundaries |
| exporter artifacts | track `CastLike` and generated casts separately before changing model math |

The current wrappers use `src/export/common.py::cast_tensor_if_needed()` so FP32 exports do not emit no-op graph-edge casts while BF16 keeps intentional boundary casts.

Graph-level Cast summary:

```bash
python -B tools/profile/summarize_dtype_casts.py \
  --after-root models/onnx \
  --profile-json artifacts/profile/parsed_hotspots.json \
  --json-report artifacts/reports/dtype_cleanup_casts.json \
  --markdown-report artifacts/reports/dtype_cleanup_casts.md
```

If pre-cleanup artifacts were archived:

```bash
python -B tools/profile/summarize_dtype_casts.py \
  --before-root artifacts/pre_dtype_cleanup/onnx \
  --after-root models/onnx \
  --profile-json artifacts/profile/parsed_hotspots.json \
  --json-report artifacts/reports/dtype_cleanup_casts.json \
  --markdown-report artifacts/reports/dtype_cleanup_casts.md
```

## Legacy BF16 Storage Experiment

`src/experiments/bf16_feasibility.py` is retained only for historical storage feasibility analysis.

It can:

- inspect initializer dtypes and logical bytes
- find Cast nodes and direct Cast chains
- compare model size before/after copied BF16 initializers
- write experimental storage-only copies under `artifacts/experiments/bf16_storage_only`

It must not write production files under `models/onnx/bf16`.

Analyze existing FP32 artifacts:

```bash
python -B src/experiments/bf16_feasibility.py --mode analyze
```

Create separate storage-only experimental copies:

```bash
python -B src/experiments/bf16_feasibility.py \
  --mode convert \
  --models audio_vae_encoder audio_vae_decoder \
  --output-dir artifacts/experiments/bf16_storage_only \
  --report-json artifacts/reports/bf16_feasibility/audio_vae_partial_bf16_report.json \
  --check-ort
```

Historic finding: storage-only conversion roughly halves large initializer bytes but adds many BF16-to-FP32 Cast nodes. That is useful for disk-size analysis, not production compute.

## Verification

Static precision policy tests:

```bash
python -B -m pytest tests/export/test_export_contract_consistency.py tests/export/test_dtype_cleanup.py tests/parity/test_bf16_compute_path.py
```

After BF16 export, check for forbidden storage-only and ping-pong patterns:

```bash
VOXCPM2_BF16_ONNX_PATHS="models/onnx/bf16/audio_vae_encoder/audio_vae_encoder.onnx:models/onnx/bf16/audio_vae_decoder/audio_vae_decoder.onnx:models/onnx/bf16/prefill/voxcpm2_prefill.onnx:models/onnx/bf16/decode_step/voxcpm2_decode_step.onnx" \
python -B -m pytest tests/parity/test_bf16_compute_path.py
```

Windows PowerShell uses `;` instead of `:` in `VOXCPM2_BF16_ONNX_PATHS`.

## Final Acceptance Criteria

- Optimized FP32 artifacts exist for all four modules.
- Optimized BF16 artifacts exist for all four modules.
- Both families load with ONNX Runtime CPU only.
- Both families use the same runtime path and feature matrix.
- FP32 and BF16 have module-level parity reports.
- FP32 and BF16 have end-to-end quality reports against official VoxCPM2 API on CPU.
- FP32 and BF16 have CPU performance baseline reports against official VoxCPM2 API.
- BF16 reports show minimized Cast/CastLike overhead and document every required mixed-precision island.
