# VoxCPM2 ONNX CPU Runtime

This repository is a CPU-only ONNX Runtime porting workspace for VoxCPM2. It keeps VoxCPM2 neural modules in separate ONNX graphs and keeps host responsibilities, such as tokenization, WAV I/O, resampling, and decode orchestration, in Python.

The goal is FP32 correctness first on:

- macOS arm64
- Linux x86_64 / arm64
- Windows x86_64 / arm64

GPU, CoreML, DirectML, CUDA, quantization, BF16, streaming, and a single monolithic ONNX graph are out of scope for v1.

## Repository Layout

```text
src/export/      PyTorch -> ONNX export wrappers
src/runtime/     CPU-only ONNX Runtime sessions and pipeline
src/cli/         User-facing synthesis CLI
src/parity/      Official VoxCPM2 generate-path tracing tool
src/contracts/   Typed module-boundary schemas
tests/parity/    PyTorch vs ONNX Runtime module parity checks
tests/smoke/     End-to-end CPU-only runtime smoke test
docs/            Contracts, boundary specs, blockers, platform notes
REPORT.md        Trace-based call map and ONNX boundary rationale
```

Ignored local state is split by purpose:

- `models/onnx/fp32/`: production FP32 ONNX exports and external data.
- `models/onnx/bf16/`: experimental BF16-initializer ONNX copies.
- `models/hf/`: optional local Hugging Face snapshots if you do not use the default cache.
- `artifacts/`: reports, logs, benchmark WAV/JSON, and smoke output samples.
- `traces/`: generate-path JSONL traces.
- `.venv/` and `third_party/`: local environment and upstream checkout.

ONNX files, external data, downloaded weights, traces, and generated reports are not source files and are intentionally ignored by git.

## Architecture

The v1 runtime is split into four ONNX modules:

1. `AudioVAEEncoder`: reference/prompt WAV tensor -> audio latent features.
2. `VoxCPM2Prefill`: text/audio aligned sequence -> initial hidden states and explicit K/V caches.
3. `VoxCPM2DecodeStep`: one autoregressive audio-feature step -> predicted feature, stop logits, next state.
4. `AudioVAEDecoder`: generated latent feature sequence -> waveform.

Host code owns:

- text normalization
- tokenizer loading and token ID assembly
- audio loading, mono conversion, resampling, and padding
- reference/prompt path assembly
- decode loop and stop policy
- random diffusion noise
- WAV writing

This split is documented in [docs/module_boundaries.md](docs/module_boundaries.md) and [docs/decode_state_contract.md](docs/decode_state_contract.md).

## Setup

Clone with the upstream VoxCPM submodule:

```bash
git clone --recursive <repo-url> voxcpm2-onnx-cpu
cd voxcpm2-onnx-cpu
```

If the repository is already cloned:

```bash
git submodule update --init --recursive
```

Create and activate the environment by sourcing the setup script with an explicit mode:

```bash
source setup.sh base
```

Setup modes:

- `base`: core runtime dependencies plus export/parity dependencies.
- `dev`: `base` plus developer tools such as `pytest` and `ruff`.

```bash
source setup.sh dev
```

Both modes install this project in editable mode. If `third_party/VoxCPM` exists, setup also installs it in editable mode with `--no-deps`; this exposes the official `voxcpm` package while dependency versions remain controlled by this repository's `pyproject.toml`.

On Windows without a Unix-compatible shell, use the equivalent manual commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip>=24,<26" "setuptools>=70,<81" "wheel>=0.43,<1"
python -m pip install -e ".[export]"
python -m pip install -e "third_party/VoxCPM" --no-deps

# Developer mode:
python -m pip install -e ".[dev,export]"
python -m pip install -e "third_party/VoxCPM" --no-deps
```

## Download Weights

The runtime defaults to local files only. Download VoxCPM2 weights once:

```bash
python -c "from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2', load_denoiser=False)"
```

To force scripts to download missing files, pass `--allow-download` where available.

## Export ONNX Modules

Run exports in this order:

```bash
python -B src/export/export_audio_vae_encoder.py --output models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
python -B src/export/export_audio_vae_decoder.py --output models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
python -B src/export/export_prefill.py --output models/onnx/fp32/prefill/voxcpm2_prefill.onnx --mode plain_tts
python -B src/export/export_decode_step.py --output models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx --cache-seq 16
```

All exports use:

- `torch.onnx.export(..., dynamo=True)`
- `external_data=True`
- FP32 model weights and synthetic inputs
- no graph optimization
- no quantization

Large ONNX external data files must remain next to their `.onnx` files.

## Check Exported Graphs

Run path-based ONNX checker and one CPU ORT invocation per module:

```bash
python -B src/runtime/run_audio_vae_encoder_ort.py --onnx-path models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
python -B src/runtime/run_audio_vae_decoder_ort.py --onnx-path models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
python -B src/runtime/run_prefill_ort.py --onnx-path models/onnx/fp32/prefill/voxcpm2_prefill.onnx --mode plain_tts
python -B src/runtime/run_decode_step_ort.py --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx
```

Each checker prints input names, output names, dtype, dynamic/static dimensions, CPU provider list, and compact output statistics.

## Parity Checks

Parity scripts compare the PyTorch export wrapper against ONNX Runtime CPU:

```bash
python -B tests/parity/test_audio_vae_encoder.py --onnx-path models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
python -B tests/parity/test_audio_vae_decoder.py --onnx-path models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
python -B tests/parity/test_prefill.py --onnx-path models/onnx/fp32/prefill/voxcpm2_prefill.onnx
python -B tests/parity/test_decode_step.py --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx
```

PyTest wrappers are also available through environment variables:

```bash
VOXCPM2_AUDIO_VAE_ENCODER_ONNX=models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx pytest tests/parity/test_audio_vae_encoder.py
VOXCPM2_AUDIO_VAE_DECODER_ONNX=models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx pytest tests/parity/test_audio_vae_decoder.py
VOXCPM2_PREFILL_ONNX=models/onnx/fp32/prefill/voxcpm2_prefill.onnx pytest tests/parity/test_prefill.py
VOXCPM2_DECODE_STEP_ONNX=models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx pytest tests/parity/test_decode_step.py
```

## CPU-Only Runtime Smoke

Run one short end-to-end synthesis through already exported ONNX modules:

```bash
python -B tests/smoke/test_cpu_only_runtime.py
```

Expected output:

```text
cpu_only_runtime_smoke=ok
```

The smoke test verifies:

- all required ONNX artifacts and external data files exist
- sessions are created lazily
- every ONNX Runtime session uses only `CPUExecutionProvider`
- runtime orchestration does not import PyTorch, Transformers, `soundfile`, or `librosa`
- one short text-only inference reaches waveform
- prefill input assembly supports text-only, voice design, controllable clone, and ultimate clone pathways

## Synthesize WAV

```bash
python -B src/cli/synthesize.py \
  --text "Hello from VoxCPM2." \
  --output artifacts/samples/runtime_sample.wav \
  --mode text_only
```

For voice design:

```bash
python -B src/cli/synthesize.py \
  --mode voice_design \
  --voice-design "calm voice" \
  --text "Hello from VoxCPM2." \
  --output artifacts/samples/voice_design.wav
```

For clone modes, provide `--reference-wav`, and for ultimate clone also provide `--prompt-wav` and `--prompt-text`.

`--max-steps` is optional. The default `--max-steps 0` means "run until the model emits stop logits" with an internal safety cap. `--max-steps 1 --min-steps 0` is only for very fast graph-load smoke checks and writes a deliberately truncated WAV.

## Benchmark Variants

Compare the official VoxCPM2 API, ONNX FP32, and experimental ONNX BF16 artifacts:

```bash
python -B src/bench/compare_pipelines.py \
  --text "Hello from VoxCPM2." \
  --output-dir artifacts/bench \
  --variants orig onnx_fp32 onnx_bf16
```

The benchmark prints human-readable progress and saves JSON to `artifacts/bench/report.json` unless `--report-json` overrides the path. Each result contains:

- `output_wav`
- `load_seconds`
- `synth_seconds`
- `total_seconds`
- `sample_rate`
- `samples`
- `duration_seconds`
- `peak`
- `rms`
- `decode_steps` / `stop_reason` for ONNX variants

BF16 paths default to `models/onnx/bf16/*`. Production FP32 defaults are not changed.

## Trace Official Generate Path

Use the trace tool before changing module boundaries:

```bash
python -B src/parity/trace_generate.py \
  --model-path openbmb/VoxCPM2 \
  --mode plain_tts \
  --text "Hello." \
  --trace-output traces/plain_tts.jsonl
```

Trace records are compact JSONL events with stage name, input/output shapes, dtypes, reference/prompt pathway flags, and Python module/function names. Full tensor values are not logged.

## Documentation Map

- [REPORT.md](REPORT.md): traced call map and selected ONNX boundary rationale.
- [docs/export_contract.md](docs/export_contract.md): export scope, rules, acceptance criteria, and non-goals.
- [docs/runtime_contract.md](docs/runtime_contract.md): runtime responsibilities and CPU-only constraints.
- [docs/feature_matrix.md](docs/feature_matrix.md): v1 feature status and deferred work.
- [docs/module_boundaries.md](docs/module_boundaries.md): module split and tensor boundaries.
- [docs/decode_state_contract.md](docs/decode_state_contract.md): explicit decode-step state and cache contract.
- [docs/prefill_blockers.md](docs/prefill_blockers.md): prefill export boundary, mode inputs, and blockers.
- [docs/audio_vae_encoder_onnx_report.md](docs/audio_vae_encoder_onnx_report.md): encoder blocker isolation and parity notes.
- [docs/bf16_feasibility.md](docs/bf16_feasibility.md): experimental BF16 initializer-size analysis and rollback rules.
- [docs/torch_dependency_audit.md](docs/torch_dependency_audit.md): runtime PyTorch dependency audit.
- [docs/platform_support.md](docs/platform_support.md): platform matrix and verification commands.

## Release Checklist

Before publishing a release candidate:

```bash
python -B -m py_compile $(find src tests -name '*.py' -print)
python -B tests/smoke/test_cpu_only_runtime.py
python -B src/cli/synthesize.py --text "Hello from VoxCPM2." --output artifacts/samples/runtime_sample.wav --mode text_only
rg -n "import torch|from torch|soundfile|librosa|transformers" src/runtime src/cli tests/smoke
git diff --check
```

Run the same smoke commands on every target platform class before claiming platform release support.
