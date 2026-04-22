# VoxCPM2 ONNX Runtime Contract

## Scope

Runtime target: ONNX Runtime CPU only for VoxCPM2 on:

- macOS arm64
- Linux x86_64 / arm64
- Windows x86_64 / arm64

The runtime consumes separate ONNX neural-module files. It does not assume one combined model.
Platform implementation status is tracked in `docs/platform_support.md`.

Current code uses explicit module paths from `src/runtime/session_factory.py`. A machine-readable export manifest remains a release-candidate requirement, but the runtime contract below is already enforced by the smoke test.

## Host Code Responsibilities

Host code owns:

- text normalization
- tokenization
- WAV I/O
- resampling
- loading configs and exported module paths
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
- Treat ONNX file boundaries as stable module boundaries.
- Validate input names, output names, dtypes, ranks, and required dynamic axes before release.
- Keep multilingual and reference-audio paths active in runtime orchestration.
- Fail fast with actionable errors when a required ONNX module or external data file is missing.
- Do not change model math in runtime code.

## Acceptance Criteria

- Runtime can validate required ONNX/external-data files without executing synthesis.
- Each v1 mode has a documented module orchestration path.
- Host/ONNX boundaries match the export contract exactly.
- CPU-only execution is explicit and testable.
- Missing modules, incompatible opset/runtime versions, and shape/dtype mismatches have defined error cases.

## Reproduce

```bash
python -B tests/smoke/test_cpu_only_runtime.py
python -B src/cli/synthesize.py --text "Hello from VoxCPM2." --output artifacts/runtime_sample.wav --mode text_only
```

The default `--max-steps 0` means the host decode-loop runs until ONNX `stop_logits` end the stream. The runtime still has an internal safety cap to avoid infinite loops if a decode-step export is invalid. `--max-steps 1 --min-steps 0` is valid only for graph-load smoke tests and produces intentionally truncated audio.

## Non-Goals

- Streaming runtime.
- GPU, accelerator, or vendor-specific execution providers.
- Runtime-side model simplification, quantization, or BF16 conversion.
- Replacing host tokenization, normalization, WAV I/O, or resampling with ONNX graphs.
