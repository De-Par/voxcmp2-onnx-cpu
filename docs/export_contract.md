# VoxCPM2 ONNX Export Contract

## Scope

Target model: VoxCPM2.

Target runtime: ONNX Runtime CPU only.

Target platforms:

- macOS arm64
- Linux x86_64 / arm64
- Windows x86_64 / arm64

"Fully ONNX" means every neural module required by the supported VoxCPM2 paths is exported to ONNX. Text normalization, tokenization, WAV I/O, resampling, configuration loading, and orchestration remain in host code.

This page is the publication contract for export work. It defines what an exported artifact set must provide before it can be used by the CPU-only runtime.

Precision target: production FP32 and production BF16 artifact families. FP32 is the correctness anchor. BF16 is a parallel production target, not a storage-only final state.

## Current Implementation Status

The repository currently provides export scripts for:

- `AudioVAEEncoder`
- `AudioVAEDecoder`
- `VoxCPM2Prefill`
- `VoxCPM2DecodeStep`

Each script logs input names, output names, dtypes, and dynamic/static dimensions. A formal machine-readable export manifest is still a release-candidate requirement; until then, `src/runtime/session_factory.py` uses explicit default paths and the docs define the contract.

## Required V1 Modes

The export must preserve the neural paths needed for:

- text-to-speech
- voice design
- controllable cloning
- ultimate cloning

Streaming is deferred to v2 and must not shape the v1 export contract.

## Export Rules

- Export neural modules as separate ONNX graphs; do not merge the whole system into one ONNX model.
- Preserve the multilingual path. Do not hardcode language.
- Preserve the reference-audio path used by cloning modes.
- Export both FP32 and BF16 production artifact families.
- Keep FP32 and BF16 graph structures aligned: same module boundaries, input names, output names, ranks, dynamic axes, and cache/state semantics.
- Treat FP32 as the correctness anchor and BF16 as a production target with explicit quality and performance gates.
- Production BF16 must minimize unnecessary `Cast`/`CastLike` nodes and dtype churn.
- Use mixed precision only where required for correctness, numerical stability, or ONNX Runtime CPU kernel support.
- Do not simplify model math unless the blocker and behavior change are documented before the change.
- Keep export changes minimal and reversible.

## Required Artifacts

A stable export step must produce:

- one FP32 ONNX file per exported neural module
- one BF16 ONNX file per exported neural module
- a machine-readable manifest listing every ONNX file, precision family, module role, opset, expected inputs, expected outputs, dtypes, dynamic axes, and intentional mixed-precision islands
- export logs with source checkpoint/config identifiers
- parity fixtures sufficient to compare exported module outputs against the source implementation and FP32/BF16 artifact families against each other

## Precision Requirements

FP32 and BF16 artifacts must share a common public graph contract:

- `AudioVAEEncoder` FP32 and BF16 expose the same host-visible input/output contract.
- `AudioVAEDecoder` FP32 and BF16 expose the same host-visible input/output contract.
- `VoxCPM2Prefill` FP32 and BF16 expose the same hidden-state/cache output contract.
- `VoxCPM2DecodeStep` FP32 and BF16 expose the same state tensor input/output contract.

The current BF16 initializer-storage conversion, where weights are stored as BF16 and broadly cast back to FP32, is a feasibility artifact only. It must not be described as the final production BF16 export target.

Production BF16 export reports must include:

- initializer dtype summary
- graph input/output dtype summary
- Cast/CastLike count and top latency hotspots from ORT profiling
- documented FP32 islands and the reason each island exists
- parity tolerance and measured error per module
- CPU performance comparison against FP32 ONNX and official VoxCPM2 API

## Acceptance Criteria

- All neural modules needed by the four v1 modes are represented in ONNX or listed as a documented blocker.
- The manifest has no host-code-only items listed as ONNX modules.
- ONNX graphs load with ONNX Runtime CPU execution provider on all target platform classes.
- FP32 module-level parity checks exist for each exported graph.
- BF16 module-level parity checks exist for each exported graph with explicit tolerances.
- FP32 and BF16 support identical feature coverage.
- Production BF16 reports show minimized Cast/dtype churn or document unavoidable mixed-precision islands.
- Any unsupported operator, dynamic-shape issue, or required math change is documented before implementation work continues.
- Final release includes optimized FP32 models, optimized BF16 models, and official-API quality/performance comparisons on CPU.

## Reproduce

```bash
python -B src/export/export_audio_vae_encoder.py --output models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
python -B src/export/export_audio_vae_decoder.py --output models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
python -B src/export/export_prefill.py --output models/onnx/fp32/prefill/voxcpm2_prefill.onnx --mode plain_tts
python -B src/export/export_decode_step.py --output models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx --cache-seq 16
```

## Non-Goals

- Implementing inference runtime.
- Implementing streaming.
- Exporting text normalization, tokenization, WAV I/O, resampling, or orchestration into ONNX.
- Combining all modules into a single ONNX graph.
- Quantization, GPU, CoreML, DirectML, CUDA, or mobile runtimes.
- Storage-only BF16 conversion as the final production BF16 strategy.
