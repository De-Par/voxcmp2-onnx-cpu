# VoxCPM2 ONNX Runtime Contract

## Scope

Runtime target: ONNX Runtime CPU only for VoxCPM2 on:

- macOS arm64
- Linux x86_64 / arm64
- Windows x86_64 / arm64

The runtime consumes the export manifest and separate ONNX neural-module files. It does not assume one combined model.
Platform implementation status is tracked in `docs/platform_support.md`.

## Host Code Responsibilities

Host code owns:

- text normalization
- tokenization
- WAV I/O
- resampling
- loading configs and export manifests
- session orchestration across ONNX modules
- language selection and propagation without hardcoding one language
- reference-audio loading and preprocessing for cloning modes

ONNX Runtime owns only neural-module execution.

## Required V1 Modes

The runtime contract must support orchestration for:

- text-to-speech
- voice design
- controllable cloning
- ultimate cloning

Streaming is deferred to v2.

## Runtime Rules

- Use ONNX Runtime CPU execution provider only.
- Treat ONNX file boundaries as stable module boundaries from the export manifest.
- Validate input names, output names, dtypes, ranks, and required dynamic axes against the manifest before execution.
- Keep multilingual and reference-audio paths active in runtime orchestration.
- Fail fast with actionable errors when a required ONNX module or manifest entry is missing.
- Do not change model math in runtime code.

## Acceptance Criteria

- Runtime design can load and validate the manifest without executing synthesis.
- Each v1 mode has a documented module orchestration path.
- Host/ONNX boundaries match the export contract exactly.
- CPU-only execution is explicit and testable.
- Missing modules, incompatible opset/runtime versions, and shape/dtype mismatches have defined error cases.

## Non-Goals

- Writing the actual inference implementation in this phase.
- Streaming runtime.
- GPU, accelerator, or vendor-specific execution providers.
- Runtime-side model simplification, quantization, or BF16 conversion.
- Replacing host tokenization, normalization, WAV I/O, or resampling with ONNX graphs.
