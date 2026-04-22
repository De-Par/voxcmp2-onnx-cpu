# Precision Strategy

## Scope

The production target is a dual-precision CPU-only ONNX Runtime port of VoxCPM2:

- production FP32 ONNX artifacts
- production BF16 ONNX artifacts
- one production runtime path that can load either artifact family
- quality and CPU performance comparison against the official VoxCPM2 API

This document is policy only. It does not authorize changing runtime code, export code, model math, or graph structure without a separate implementation plan and parity evidence.

## Production Targets

FP32 and BF16 are both production targets.

FP32 remains the correctness anchor and the first artifact family that must pass parity, smoke, profiling, and benchmark checks. BF16 is not a replacement for FP32; it is a second production artifact family with the same feature coverage and the same host runtime contract.

The current BF16 initializer-copy experiment is not the production BF16 strategy. It stores selected weights as BF16 and casts them back to FP32 before use. That was useful for storage and ORT loader feasibility, but production BF16 is now defined as a real compute path in `docs/bf16_compute_strategy.md`.

## Artifact Families

The production layout is:

```text
models/onnx/fp32/<module>/<module>.onnx
models/onnx/bf16/<module>/<module>.onnx
```

Both families must contain the same module set:

- `AudioVAEEncoder`
- `AudioVAEDecoder`
- `VoxCPM2Prefill`
- `VoxCPM2DecodeStep`

Both families must preserve the same module boundaries. Do not merge modules into one monolithic ONNX graph to solve precision or performance issues.

The concrete export profile registry lives in `src/export/common.py`. `src/export/export_all.py` applies one profile across all four module exporters so FP32 and BF16 artifacts are produced by the same pipeline.

## Common Graph Policy

FP32 and BF16 artifacts must use a common graph structure by contract:

- same module boundaries
- same graph input names
- same graph output names
- same host-visible ranks and dynamic axes
- same cache/state tensor contract between prefill and decode_step
- same stop-logit semantics
- same mode coverage

Internal tensor dtypes may differ by precision family. BF16 graphs may contain FP32 islands only when required for correctness, numerical stability, or ONNX Runtime CPU kernel support. Every intentional mixed-precision island must be documented in the export report or manifest.

## BF16 Policy

Production BF16 must:

- keep neural math in BF16 where ONNX Runtime CPU has correct and performant kernels
- minimize BF16-to-FP32 and FP32-to-BF16 casts
- avoid blanket weight-storage conversion followed by immediate FP32 casts as the final strategy
- use mixed precision only where needed for correctness or CPU kernel support
- keep external API and runtime orchestration identical to FP32

Allowed BF16 mixed-precision reasons:

- an op has no usable ORT CPU BF16 kernel on a target platform
- an op is numerically unstable in BF16 beyond accepted tolerance
- a graph boundary requires FP32 host-visible tensors for compatibility and the cast is documented

Not allowed as final production BF16:

- storage-only BF16 conversion with broad immediate casts back to FP32
- undocumented Cast/CastLike chains
- feature coverage lower than FP32
- separate host runtime logic for BF16 behavior

## Runtime Policy

There is one production runtime path. Runtime selection between FP32 and BF16 must be an artifact-selection concern, not a model-logic fork.

The runtime must keep these behaviors identical for FP32 and BF16:

- text normalization boundary
- tokenization boundary
- reference/prompt WAV loading and resampling
- multilingual path
- decode loop
- stop policy
- cache/state orchestration
- WAV writing

The runtime may validate precision-family manifests and reject incompatible artifacts, but it must not rewrite graph math or silently insert precision conversions.

## Quality Policy

FP32 and BF16 must use the same quality criteria:

- same fixed benchmark/test cases
- same required modes: text-only, voice design, controllable clone, ultimate clone
- same decode-loop semantics
- same output validation fields: sample rate, duration, peak/RMS sanity, decode steps, stop reason
- same comparison framework against the official VoxCPM2 API on CPU

BF16 quality may use a wider numerical tolerance than FP32 at module level, but the tolerance must be explicit and justified per module. End-to-end BF16 audio must be compared against FP32 ONNX and official API outputs before being called production-ready.

## Performance Policy

Performance comparisons must include:

- official VoxCPM2 API on CPU
- ONNX FP32 production artifacts
- ONNX BF16 production artifacts

The comparison must report:

- model load latency
- prefill latency
- decode_step latency
- total synthesis latency
- decode steps
- output duration
- p50/p90 wall time
- ORT profiling hotspots for both FP32 and BF16

BF16 promotion requires evidence that Cast/dtype churn is not the dominant bottleneck.

No-op casts are not allowed as part of the production policy. Export wrappers must skip dtype conversion when the tensor already has the target dtype; required graph-boundary casts and documented FP32 islands must remain explicit and measurable. See `docs/dtype_cleanup_report.md` for the cleanup classification and cast-count workflow.

## Final Goal Acceptance Criteria

The project reaches the production precision goal when:

- optimized FP32 ONNX artifacts exist for all four modules
- optimized BF16 ONNX artifacts exist for all four modules
- both artifact families load with ONNX Runtime CPU only on target platform classes
- both artifact families use the same production runtime path
- both artifact families support the same feature matrix
- FP32 and BF16 have module-level parity reports
- FP32 and BF16 have end-to-end quality reports against official VoxCPM2 API on CPU
- FP32 and BF16 have CPU performance baseline reports against official VoxCPM2 API
- BF16 reports show minimized Cast/CastLike overhead and document every required mixed-precision island

## Non-Goals

- Quantization.
- GPU, CoreML, CUDA, DirectML, MPS, or accelerator providers.
- A separate BF16-only runtime.
- A monolithic ONNX graph.
- Silent precision conversion in host runtime.
