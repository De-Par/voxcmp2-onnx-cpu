# BF16 Feasibility

## Scope

This page evaluates BF16 for the existing CPU-only ONNX Runtime pipeline without replacing the FP32 baseline.

FP32 remains the production contract. BF16 work is allowed only as an experiment that writes separate artifacts and can be discarded without touching:

- `src/runtime/session_factory.py`
- `src/runtime/pipeline.py`
- default `models/onnx/fp32/*/*.onnx` paths
- FP32 export scripts

## Experiment Utility

Use `src/experiments/bf16_feasibility.py`.

It can:

- inspect ONNX initializer dtypes and logical initializer bytes
- find `Cast` nodes and direct Cast-to-Cast chains
- compare `.onnx` plus external-data file sizes before and after conversion
- write experimental BF16-initializer model copies under `models/onnx/bf16`
- optionally try loading converted copies with ONNX Runtime CPU

The conversion mode is intentionally conservative: selected FLOAT initializers are stored as BFLOAT16, then explicit `Cast` nodes convert them back to FLOAT before graph use. That tests storage reduction and ORT CPU loader coverage. It does not claim BF16 compute speedup.

Terminal output is intentionally compact. Detailed data is written to the JSON report path printed by the tool.

## Commands

Analyze existing FP32 artifacts:

```bash
python -B src/experiments/bf16_feasibility.py \
  --mode analyze
```

Create a small, separate partial-BF16 patch set for the AudioVAE modules:

```bash
python -B src/experiments/bf16_feasibility.py \
  --mode convert \
  --models audio_vae_encoder audio_vae_decoder \
  --output-dir models/onnx/bf16 \
  --report-json artifacts/reports/bf16_feasibility/audio_vae_partial_bf16_report.json \
  --check-ort
```

Create full copied artifacts, including the multi-GiB prefill and decode-step graphs:

```bash
python -B src/experiments/bf16_feasibility.py \
  --mode convert \
  --models audio_vae_encoder audio_vae_decoder prefill decode_step \
  --include-large-models \
  --output-dir models/onnx/bf16 \
  --report-json artifacts/reports/bf16_feasibility/full_bf16_report.json \
  --check-ort
```

Optional branch workflow:

```bash
git switch -c experiment/partial-bf16
python -B src/experiments/bf16_feasibility.py --mode convert --models audio_vae_encoder audio_vae_decoder --check-ort
```

Do not point production runtime defaults at `models/onnx/bf16`.

## Current Findings

The existing FP32 artifacts are dominated by FLOAT initializers.

| Module | FLOAT initializers | FLOAT logical bytes | Artifact bytes | Existing Cast nodes | Cast-chain edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| `AudioVAEEncoder` | 118 | 192,166,912 | 194,583,771 | 0 | 0 |
| `AudioVAEDecoder` | 190 | 183,115,528 | 186,535,963 | 23 | 0 |
| `VoxCPM2Prefill` | 451 | 8,345,376,768 | 8,359,338,988 | 26 | 0 |
| `VoxCPM2DecodeStep` | 583 | 8,658,995,500 | 8,701,079,672 | 131 | 0 |

The large prefill/decode-step graphs are the biggest storage opportunity, but they are also the highest-risk candidates because converting them duplicates multi-GiB artifacts and introduces thousands of extra Cast nodes if every FLOAT initializer is protected by BF16-to-FP32 casts.

## BF16 Patch Sets

Experimental artifacts are copied under:

```text
models/onnx/bf16/audio_vae_encoder/audio_vae_encoder.onnx
models/onnx/bf16/audio_vae_encoder/audio_vae_encoder.onnx.data
models/onnx/bf16/audio_vae_decoder/audio_vae_decoder.onnx
models/onnx/bf16/audio_vae_decoder/audio_vae_decoder.onnx.data
models/onnx/bf16/prefill/voxcpm2_prefill.onnx
models/onnx/bf16/prefill/voxcpm2_prefill.onnx.data
models/onnx/bf16/decode_step/voxcpm2_decode_step.onnx
models/onnx/bf16/decode_step/voxcpm2_decode_step.onnx.data
```

These files are ignored artifacts. They are not production defaults.

Measured full-conversion result with `--min-tensor-bytes 4096`:

| Module | Converted FLOAT initializers | New BF16 initializers | Added BF16-to-FP32 Cast nodes | Before bytes | After bytes | Saved bytes | Size ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AudioVAEEncoder` | 49 | 49 | 49 | 194,583,771 | 98,505,917 | 96,077,854 | 0.506 |
| `AudioVAEDecoder` | 67 | 67 | 67 | 186,535,963 | 95,052,071 | 91,483,892 | 0.510 |
| `VoxCPM2Prefill` | 450 | 450 | 450 | 8,359,338,988 | 4,186,707,901 | 4,172,631,087 | 0.501 |
| `VoxCPM2DecodeStep` | 580 | 580 | 580 | 8,701,079,672 | 4,371,716,659 | 4,329,363,013 | 0.502 |

ONNX checker and ONNX Runtime CPU session load passed for all four converted modules. Synthetic ORT runs also passed:

- `AudioVAEEncoder`: output `latent` was `float32[1, 64, 32]`.
- `AudioVAEDecoder`: output `waveform` was `float32[1, 1, 7680]`.
- `VoxCPM2Prefill`: output caches and hidden states were `float32`, cache lengths were `int64`.
- `VoxCPM2DecodeStep`: output feature, hidden, and cache tensors were `float32`, next cache lengths were `int64`.

An end-to-end text-only sample using the copied BF16-initializer artifacts also reached waveform:

```bash
python -B src/cli/synthesize.py \
  --text "Hello from VoxCPM2." \
  --output artifacts/samples/bf16_runtime_sample.wav \
  --mode text_only \
  --audio-encoder-onnx models/onnx/bf16/audio_vae_encoder/audio_vae_encoder.onnx \
  --audio-decoder-onnx models/onnx/bf16/audio_vae_decoder/audio_vae_decoder.onnx \
  --prefill-onnx models/onnx/bf16/prefill/voxcpm2_prefill.onnx \
  --decode-step-onnx models/onnx/bf16/decode_step/voxcpm2_decode_step.onnx
```

## Safety Assessment

Safe for experimentation:

- storing large FLOAT initializers as BFLOAT16 in separate copied ONNX files
- inserting BF16-to-FP32 Cast nodes at graph load/runtime boundaries
- testing ORT CPU session load on converted copies
- running module-level parity on converted copies before any runtime integration

Not safe for v1 runtime yet:

- replacing default FP32 artifacts
- changing `OnnxModelPaths` to BF16 artifacts
- removing FP32 parity requirements
- assuming BF16 compute kernels exist for MiniCPM, LocDiT/CFM, AudioVAE, or attention ops on every CPU target

## ORT CPU Kernel Coverage Risk

The conservative conversion keeps most graph computation in FLOAT because each converted weight is cast back to FLOAT before use. This avoids relying on broad BF16 compute kernel coverage.

True BF16 compute would require changing operator input dtypes beyond initializers. That must be evaluated per graph and per target CPU. If ONNX Runtime CPU rejects a BF16 graph, or if parity exceeds tolerance, the experiment must be discarded and the FP32 baseline kept.

For this pass, ORT CPU coverage is sufficient for conservative BF16-initializer copies of all four modules. It is not proven for true BF16 compute.

## Acceptance Criteria For Any Future BF16 Runtime

- FP32 runtime smoke remains unchanged and passing.
- BF16 artifacts live under a separate directory and are never default paths.
- `onnx.checker.check_model(str(path))` passes for every converted graph.
- ONNX Runtime CPU can create sessions for every converted graph.
- Module parity is measured against FP32/PyTorch baselines.
- End-to-end WAV smoke is measured separately from FP32 and can be disabled by deleting the BF16 artifact directory.

## Decision

Current recommendation: keep FP32 production runtime. Treat BF16 as an artifact-size experiment first. Promote only if per-module parity and ORT CPU coverage are proven for all target platforms.
