# VoxCPM2 CPU ONNX Feature Matrix

## Definitions

Target: VoxCPM2 exported to separate ONNX neural modules and executed with ONNX Runtime CPU only.

"Fully ONNX" means all neural modules for the supported path are ONNX. Host code still performs text normalization, tokenization, WAV I/O, resampling, and orchestration.

This page is the feature scope for v1. It intentionally separates required behavior from deferred work so publication does not imply unsupported capabilities.

## Matrix

| Feature | V1 Status | Requirement |
| --- | --- | --- |
| Text-to-speech | Must have | Neural modules in ONNX; host code handles text preprocessing and orchestration. |
| Voice design | Must have | Preserve the VoxCPM2 path required by the source model; do not invent replacement behavior. |
| Controllable cloning | Must have | Preserve reference-audio path and controls exposed by the source model. |
| Ultimate cloning | Must have | Preserve reference-audio path required by the source model. |
| Multilingual operation | Must have | Preserve multilingual path; language must not be hardcoded. |
| Streaming | Defer v2 | No v1 implementation or export requirement. |
| FP32 execution | Must have | Production correctness anchor and optimized artifact family. |
| BF16 execution | Must have | Parallel production artifact family with the same runtime path, quality criteria, and feature coverage as FP32. |
| Quantization | Non-goal | No v1 requirement. |
| GPU or accelerator execution | Non-goal | CPU execution provider only. |
| Single merged ONNX model | Non-goal | Module boundaries must remain separate. |

## Acceptance Criteria

- Every must-have feature maps to an export requirement and a runtime orchestration requirement.
- Deferred and non-goal items are not implemented accidentally in v1.
- No feature depends on hardcoded language, removed reference audio, or undocumented model simplification.
- Unknown VoxCPM2 internals are recorded as blockers instead of replaced with assumptions.
- FP32 and BF16 artifacts are compared against the official VoxCPM2 API on CPU for quality and performance.

## Verification

```bash
python -B tests/smoke/test_cpu_only_runtime.py
```

The smoke test validates the runtime path for text-only synthesis and validates prefill tensor assembly for voice design, controllable clone, and ultimate clone.
