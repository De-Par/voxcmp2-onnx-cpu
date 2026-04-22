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
- FP32 correctness is the v1 baseline. BF16 is out of scope for v1.
- Do not simplify model math unless the blocker and behavior change are documented before the change.
- Keep export changes minimal and reversible.

## Required Artifacts

A stable export step must produce:

- one ONNX file per exported neural module
- a machine-readable manifest listing every ONNX file, module role, opset, expected inputs, expected outputs, dtypes, and dynamic axes
- export logs with source checkpoint/config identifiers
- parity fixtures sufficient to compare exported module outputs against the source implementation

## Acceptance Criteria

- All neural modules needed by the four v1 modes are represented in ONNX or listed as a documented blocker.
- The manifest has no host-code-only items listed as ONNX modules.
- ONNX graphs load with ONNX Runtime CPU execution provider on all target platform classes.
- FP32 module-level parity checks exist for each exported graph.
- Any unsupported operator, dynamic-shape issue, or required math change is documented before implementation work continues.

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
- BF16, quantization, GPU, CoreML, DirectML, CUDA, or mobile runtimes.
