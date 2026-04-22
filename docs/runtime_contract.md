# VoxCPM2 ONNX Runtime Contract

## Scope

Runtime target: ONNX Runtime CPU only for VoxCPM2 on:

- macOS arm64
- Linux x86_64 / arm64
- Windows x86_64 / arm64

The runtime consumes separate ONNX neural-module files. It does not assume one combined model.
Platform implementation status is tracked in `docs/platform_support.md`.

Current code uses explicit module paths from `src/runtime/session_factory.py`. A machine-readable export manifest remains a release-candidate requirement, but the runtime contract below is already enforced by the smoke test.

Precision target: FP32 and BF16 are both production artifact families. The runtime must use the same host orchestration path for both. Precision selection is an artifact-selection concern, not a separate runtime semantics path.

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
- Support production FP32 and production BF16 artifact families with the same module boundaries.
- Keep the same runtime path, mode handling, decode loop, stop policy, and cache/state orchestration for FP32 and BF16.
- Validate input names, output names, dtypes, ranks, and required dynamic axes before release.
- Keep multilingual and reference-audio paths active in runtime orchestration.
- Fail fast with actionable errors when a required ONNX module or external data file is missing.
- Do not change model math in runtime code.
- Do not insert silent runtime-side precision conversions to hide incompatible artifacts.

## Precision Policy

The runtime contract follows `docs/precision_strategy.md`.

Runtime-visible FP32 and BF16 artifacts must provide:

- the same graph input and output names
- the same host-visible ranks and dynamic axes
- the same feature coverage
- the same cache/state tensor contract
- the same quality and performance benchmark matrix

BF16 graphs may contain mixed-precision islands only when required for correctness or ONNX Runtime CPU kernel support. Those islands must be documented by export reports or manifests. The current BF16 storage-only initializer-copy experiment is not the final production BF16 target.

## Acceptance Criteria

- Runtime can validate required ONNX/external-data files without executing synthesis.
- Each v1 mode has a documented module orchestration path.
- Host/ONNX boundaries match the export contract exactly.
- CPU-only execution is explicit and testable.
- Missing modules, incompatible opset/runtime versions, and shape/dtype mismatches have defined error cases.
- FP32 and BF16 artifacts can be selected without changing runtime semantics.
- Final release reports compare official API, FP32 ONNX, and BF16 ONNX on CPU for quality and performance.

## Reproduce

```bash
python -B tests/smoke/test_cpu_only_runtime.py
python -B src/cli/synthesize.py --text "Hello from VoxCPM2." --output artifacts/samples/runtime_sample.wav --mode text_only
```

The default `--max-steps 0` means the host decode-loop runs until ONNX `stop_logits` end the stream. The runtime still has an internal safety cap to avoid infinite loops if a decode-step export is invalid. `--max-steps 1 --min-steps 0` is valid only for graph-load smoke tests and produces intentionally truncated audio.

## Non-Goals

- Streaming runtime.
- GPU, accelerator, or vendor-specific execution providers.
- Runtime-side model simplification, quantization, or precision conversion.
- Replacing host tokenization, normalization, WAV I/O, or resampling with ONNX graphs.
