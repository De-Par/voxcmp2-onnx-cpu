# VoxCPM2 ONNX CPU

CPU-only ONNX Runtime export and runtime workspace for VoxCPM2.

This repository keeps VoxCPM2 neural work in separate ONNX graphs and keeps host responsibilities in Python: text normalization boundary, tokenizer use, WAV I/O, resampling, reference/prompt orchestration, decode loop, stop policy, random diffusion noise, and WAV writing.

## Status

| Area | Status |
|---|---|
| Target model | VoxCPM2 |
| Runtime target | ONNX Runtime CPU only |
| Platforms | macOS arm64, Linux x86_64 / arm64, Windows x86_64 / arm64 |
| V1 modes | text-to-speech, voice design, controllable clone, ultimate clone |
| Deferred | streaming |
| Precision targets | production FP32 and production BF16 |
| Non-goals | GPU/CoreML/CUDA/DirectML/MPS, quantization, monolithic ONNX graph |

## Architecture

The runtime is split into four ONNX modules:

```text
AudioVAEEncoder   reference/prompt waveform -> latent audio features
VoxCPM2Prefill    text/audio prompt tensors -> hidden states + initial KV caches
VoxCPM2DecodeStep one autoregressive audio-feature step + explicit state updates
AudioVAEDecoder   generated latent features -> waveform
```

Host code owns everything that is not neural module execution:

- text normalization boundary and tokenizer-driven sequence assembly
- WAV reading/writing and resampling
- reference and prompt-audio path construction
- decode loop and stop policy
- fixed-capacity cache mutation between decode steps
- NumPy random diffusion noise

The full contract is in [docs/architecture.md](docs/architecture.md).

## Repository Layout

```text
src/export/       PyTorch -> ONNX export wrappers and precision profiles
src/runtime/      CPU-only ONNX Runtime session factory and pipeline
src/cli/          synthesis CLI
src/bench/        quick official/API vs ONNX benchmark
src/parity/       official generate-path tracing
src/contracts/    Typed module-boundary schemas
tools/bench/      production baseline benchmark runner
tools/profile/    ORT profiling and Cast-summary tools
tests/export/     export contract and dtype cleanup tests
tests/parity/     PyTorch-wrapper vs ONNX Runtime parity checks
tests/smoke/      CPU-only runtime smoke checks
docs/             self-contained project documentation
third_party/      local VoxCPM submodule checkout
models/           local ONNX exports and optional HF snapshots
artifacts/        local reports, logs, WAVs, traces, and benchmark outputs
```

`third_party/`, `models/`, `artifacts/`, `traces/`, and `.venv/` are local/generated state and are ignored by git.

## From Fresh Clone To Exported Models

The complete path from a clean clone to ready ONNX models is:

1. Clone repository and submodule.
2. Create Python environment and install dependencies.
3. Download official VoxCPM2 weights.
4. Export FP32 and BF16 ONNX artifacts into `models/onnx`.
5. Run graph checks, parity checks, smoke synthesis, and benchmarks.

### 1. Clone

```bash
git clone --recursive <repo-url> voxcpm2-onnx-cpu
cd voxcpm2-onnx-cpu
```

If the repository is already cloned or `third_party/VoxCPM` was deleted:

```bash
git submodule update --init --recursive
```

`source setup.sh <mode>` also tries to restore the submodule if it is missing.

### 2. Install Environment

Use Python 3.11, 3.12, or 3.13. Python 3.12 is the locally used baseline.

Base mode installs runtime plus export/parity dependencies:

```bash
source setup.sh base
```

Development mode adds pytest and ruff:

```bash
source setup.sh dev
```

The script:

- creates `.venv`
- activates it in the current shell
- installs this project in editable mode
- initializes `third_party/VoxCPM` if missing
- installs `third_party/VoxCPM` in editable mode with `--no-deps`

Manual Windows PowerShell equivalent:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip>=24,<26" "setuptools>=70,<81" "wheel>=0.43,<1"
git submodule update --init --recursive
python -m pip install -e ".[export]"
python -m pip install -e "third_party/VoxCPM" --no-deps

# Developer tools:
python -m pip install -e ".[export,dev]"
```

### 3. Download VoxCPM2 Weights

Download weights into the Hugging Face cache:

```bash
python -c "from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2', load_denoiser=False)"
```

Export scripts default to local files only. If a script should fetch missing files directly, pass `--allow-download`.

### 4. Export ONNX Models

Export production FP32:

```bash
python -B src/export/export_all.py --precision fp32
```

Export production BF16:

```bash
python -B src/export/export_all.py --precision bf16
```

Expected output layout:

```text
models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
models/onnx/fp32/prefill/voxcpm2_prefill.onnx
models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx

models/onnx/bf16/audio_vae_encoder/audio_vae_encoder.onnx
models/onnx/bf16/audio_vae_decoder/audio_vae_decoder.onnx
models/onnx/bf16/prefill/voxcpm2_prefill.onnx
models/onnx/bf16/decode_step/voxcpm2_decode_step.onnx
```

Large `.onnx.data` files must stay next to their `.onnx` files.

Module-level exports:

```bash
python -B src/export/export_audio_vae_encoder.py --precision fp32
python -B src/export/export_audio_vae_decoder.py --precision fp32
python -B src/export/export_prefill.py --precision fp32 --mode plain_tts
python -B src/export/export_decode_step.py --precision fp32 --current-length 16 --max-cache-seq 64
```

### 5. Validate Exported Graphs

Path-based ONNX checker plus one ORT CPU run per module:

```bash
python -B src/runtime/run_audio_vae_encoder_ort.py --onnx-path models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
python -B src/runtime/run_audio_vae_decoder_ort.py --onnx-path models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
python -B src/runtime/run_prefill_ort.py --onnx-path models/onnx/fp32/prefill/voxcpm2_prefill.onnx --mode plain_tts
python -B src/runtime/run_decode_step_ort.py --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx --cache-seq 16 --max-cache-seq 64
```

Parity against PyTorch export wrappers:

```bash
python -B tests/parity/test_audio_vae_encoder.py --onnx-path models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
python -B tests/parity/test_audio_vae_decoder.py --onnx-path models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
python -B tests/parity/test_prefill.py --onnx-path models/onnx/fp32/prefill/voxcpm2_prefill.onnx
python -B tests/parity/test_decode_step.py --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx --precision fp32 --cache-seq 16 --max-cache-seq 64
```

CPU-only runtime smoke:

```bash
python -B tests/smoke/test_cpu_only_runtime.py
```

Expected after models exist:

```text
cpu_only_runtime_smoke=ok
```

## Synthesize WAV

Text-only:

```bash
python -B src/cli/synthesize.py \
  --text "Hello from VoxCPM2." \
  --output artifacts/samples/text_only.wav \
  --mode text_only
```

Voice design:

```bash
python -B src/cli/synthesize.py \
  --text "Hello from VoxCPM2." \
  --output artifacts/samples/voice_design.wav \
  --mode voice_design \
  --voice-design "calm voice"
```

Clone modes require `--reference-wav`. Ultimate clone also requires `--prompt-wav` and `--prompt-text`.

`--max-steps 0` is the default and means "run until `stop_logits` ends the stream" with an internal safety cap. `--max-steps 1 --min-steps 0` is only for graph-load smoke checks and writes intentionally truncated audio.

## Benchmark And Profile

Quick comparison:

```bash
python -B src/bench/compare_pipelines.py \
  --text "Hello from VoxCPM2." \
  --output-dir artifacts/bench \
  --report-json artifacts/bench/report.json \
  --variants orig onnx_fp32 onnx_bf16
```

Production baseline matrix:

```bash
python -B tools/bench/run_benchmarks.py \
  --output-dir artifacts/perf_baseline \
  --json-report artifacts/perf_baseline/baseline.json \
  --markdown-report artifacts/perf_baseline/baseline.md \
  --variants official onnx \
  --repeats 3
```

ORT node profiling:

```bash
python -B tools/profile/run_profiled_bench.py \
  --output-dir artifacts/profile \
  --cases controllable_clone_short \
  --top-n 20
```

Cast and dtype cleanup summary:

```bash
python -B tools/profile/summarize_dtype_casts.py \
  --after-root models/onnx \
  --profile-json artifacts/profile/parsed_hotspots.json \
  --json-report artifacts/reports/dtype_cleanup_casts.json \
  --markdown-report artifacts/reports/dtype_cleanup_casts.md
```

Benchmark details are in [docs/benchmarking.md](docs/benchmarking.md).

## Development Checks

These checks work on a clean checkout before model export:

```bash
ruff check .
python -B -m compileall -q src tests tools
python -B -m pytest
```

The full pytest suite skips model-dependent smoke/parity checks when ONNX artifacts are absent.

Check runtime stays free of PyTorch:

```bash
rg -n "\btorch\b|import torch|from torch|soundfile|librosa|transformers" src/runtime src/cli tests/smoke
```

## Documentation

- [docs/architecture.md](docs/architecture.md): feature matrix, traced generate path, module boundaries, runtime contract, fixed cache, platform and dependency rules.
- [docs/exporting.md](docs/exporting.md): export contract, artifact layout, module blockers, checker commands, parity commands.
- [docs/precision.md](docs/precision.md): FP32/BF16 policy, BF16 compute regions, dtype cleanup, legacy storage-only BF16 experiment.
- [docs/benchmarking.md](docs/benchmarking.md): benchmark matrix, ORT tuning, profiling, hotspot interpretation.

## Release Checklist

Before publishing a release candidate:

```bash
ruff check .
python -B -m compileall -q src tests tools
python -B -m pytest
python -B src/export/export_all.py --precision fp32
python -B src/export/export_all.py --precision bf16
python -B tests/smoke/test_cpu_only_runtime.py
python -B tools/bench/run_benchmarks.py --output-dir artifacts/perf_baseline --variants official onnx --repeats 3
git diff --check
```

Run the runtime smoke and synthesis commands on every target platform class before claiming release support.
