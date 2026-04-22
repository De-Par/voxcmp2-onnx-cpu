# Exporting And Validation

This page is the export contract and validation guide for VoxCPM2 CPU-only ONNX artifacts.

## Export Scope

The export target is four neural ONNX modules:

- `AudioVAEEncoder`
- `AudioVAEDecoder`
- `VoxCPM2Prefill`
- `VoxCPM2DecodeStep`

Host code remains responsible for text normalization, tokenization, WAV I/O, resampling, prompt/reference orchestration, random diffusion noise, decode loop, stop policy, and WAV writing.

## Required Rules

- Use `torch.onnx.export(..., dynamo=True)`.
- Save large weights with `external_data=True`.
- Do not apply graph optimization or quantization during export.
- Do not merge all modules into one ONNX graph.
- Preserve multilingual path.
- Preserve reference-audio path.
- Do not hardcode language.
- Do not simplify model math unless a blocker is documented first.
- Keep FP32 and BF16 public graph contracts identical.
- Use the fixed-capacity decode cache contract from `docs/architecture.md`.

## Artifact Layout

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

External data files must remain next to their `.onnx` files.

The shared source of truth for module names, paths, input/output contracts, precision profiles, and shape reports is `src/export/common.py`.

## One-Command Export

Export production FP32:

```bash
python -B src/export/export_all.py --precision fp32
```

Export production BF16:

```bash
python -B src/export/export_all.py --precision bf16
```

Equivalent console script after editable install:

```bash
voxcpm2-export --precision fp32
voxcpm2-export --precision bf16
```

Module-level exports:

```bash
python -B src/export/export_audio_vae_encoder.py --precision fp32
python -B src/export/export_audio_vae_decoder.py --precision fp32
python -B src/export/export_prefill.py --precision fp32 --mode plain_tts
python -B src/export/export_decode_step.py --precision fp32 --current-length 16 --max-cache-seq 64
```

Use `--precision bf16` for the BF16 family.

## Module Notes And Blockers

### AudioVAE Encoder

The ONNX wrapper exports `AudioVAE.encoder` only:

```text
AudioVAEEncoderWrapper.forward(waveform) -> latent
```

Blocked upstream helper logic is intentionally host-side:

- rank branch in `AudioVAE.encode()`
- Python padding math in `AudioVAE.preprocess()`
- dynamic right padding
- Python dict output from `CausalEncoder.forward()`

The wrapper requires padded rank-3 input and returns only `["mu"]`, matching the official encode output on preprocessed audio.

Known FP32 parity observation for `samples=20480`: max absolute diff around `8.6e-4`, mean absolute diff around `7.9e-5`. Default encoder parity tolerance is `1e-3`.

### AudioVAE Decoder

The decoder wrapper exposes:

- `latent`: `float32[batch, 64, latent_steps]`
- `sr_cond`: `int32[batch]`
- `waveform`: `float32[batch, 1, samples]`

No WAV writing, trimming, or resampling belongs in this graph.

### VoxCPM2 Prefill

The prefill wrapper exports only the non-iterative neural section from `VoxCPM2Model._inference()`:

1. feature encoder and `enc_to_lm_proj`
2. token embedding
3. text/audio embedding merge by masks
4. base MiniCPM full-prefix pass
5. FSQ replacement at audio positions
6. residual MiniCPM full-prefix pass
7. explicit K/V tensor cache outputs

Known risks:

- official `StaticKVCache` mutation is replaced by explicit tensor outputs
- prompt continuation Python list/split logic stays host-side
- MiniCPM returns cache as Python tuples, which the wrapper stacks immediately
- `scaled_dot_product_attention(enable_gqa=True)` is the main exporter/runtime risk
- LongRoPE dynamic indexing must stay documented if it fails export

### VoxCPM2 Decode Step

The decode-step wrapper exports exactly one neural step. It does not capture the host decode loop.

The graph takes fixed-capacity cache tensors, valid lengths, and host-supplied diffusion noise. It returns one-position K/V updates and next lengths.

`inference_timesteps` is fixed at export time for the internal CFM/LocDiT solve. It does not fix the number of outer autoregressive steps.

## Runtime Checkers

After export, run path-based ONNX checker and one ORT CPU invocation per module:

```bash
python -B src/runtime/run_audio_vae_encoder_ort.py --onnx-path models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
python -B src/runtime/run_audio_vae_decoder_ort.py --onnx-path models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
python -B src/runtime/run_prefill_ort.py --onnx-path models/onnx/fp32/prefill/voxcpm2_prefill.onnx --mode plain_tts
python -B src/runtime/run_decode_step_ort.py --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx --cache-seq 16 --max-cache-seq 64
```

Each checker logs input names, output names, dtype, dynamic/static dims, CPU providers, and compact output statistics. Large models must use path-based `onnx.checker.check_model(str(path))`.

## Parity Checks

Run PyTorch-wrapper vs ONNX Runtime CPU checks:

```bash
python -B tests/parity/test_audio_vae_encoder.py --onnx-path models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
python -B tests/parity/test_audio_vae_decoder.py --onnx-path models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
python -B tests/parity/test_prefill.py --onnx-path models/onnx/fp32/prefill/voxcpm2_prefill.onnx
python -B tests/parity/test_decode_step.py --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx --precision fp32 --cache-seq 16 --max-cache-seq 64
```

PyTest environment-variable form:

```bash
VOXCPM2_AUDIO_VAE_ENCODER_ONNX=models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx python -B -m pytest tests/parity/test_audio_vae_encoder.py
VOXCPM2_AUDIO_VAE_DECODER_ONNX=models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx python -B -m pytest tests/parity/test_audio_vae_decoder.py
VOXCPM2_PREFILL_ONNX=models/onnx/fp32/prefill/voxcpm2_prefill.onnx python -B -m pytest tests/parity/test_prefill.py
VOXCPM2_DECODE_STEP_ONNX=models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx python -B -m pytest tests/parity/test_decode_step.py
```

## Runtime Smoke

```bash
python -B tests/smoke/test_cpu_only_runtime.py
```

Expected after models are exported:

```text
cpu_only_runtime_smoke=ok
```

On a clean checkout without exported models, the pytest version skips this smoke instead of failing:

```bash
python -B -m pytest tests/smoke/test_cpu_only_runtime.py
```

## Acceptance Criteria

- All four modules export for FP32 and BF16.
- ONNX checker passes for every graph.
- ONNX Runtime CPU creates every session.
- Public contracts match `src/export/common.py`.
- FP32 parity exists for every module.
- BF16 parity exists with explicit tolerances before production signoff.
- Missing unsupported ops or required math changes are documented before implementation continues.
