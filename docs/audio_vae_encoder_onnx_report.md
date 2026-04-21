# AudioVAE Encoder ONNX Export Report

## Boundary

Export target: VoxCPM2 `AudioVAE.encoder` in FP32 for ONNX Runtime CPU.

The wrapper is `AudioVAEEncoderWrapper.forward(waveform) -> latent`, where:

- `waveform`: `float32[B, 1, samples]`
- `latent`: `float32[B, 64, latent_steps]`

Host code remains responsible for loading audio, resampling to the AudioVAE encode sample rate, converting mono audio to `[B, 1, samples]`, and padding `samples` to a multiple of `audio_vae.chunk_size`.

## ONNX Blockers By Operation

- `AudioVAE.encode()` rank branch: `if audio_data.ndim == 2: audio_data = audio_data.unsqueeze(1)`.
  - Status: isolated in wrapper by requiring rank-3 input.
- `AudioVAE.preprocess()` Python shape math: `math.ceil(length / pad_to) * pad_to - length`.
  - Status: isolated in wrapper by moving right-padding to host code.
- `AudioVAE.preprocess()` dynamic `F.pad(audio_data, (0, right_pad))` where `right_pad` is computed from runtime length.
  - Status: isolated in wrapper by requiring already padded input.
- `CausalEncoder.forward()` returns a Python dict with `hidden_state`, `mu`, and `logvar`.
  - Status: wrapper returns only `["mu"]`, which is the original `AudioVAE.encode()` output.
- `weight_norm` parametrized conv layers trigger PyTorch exporter warnings about assigned tensor attributes.
  - Status: not an active blocker; export, ONNX checker, ORT run, and parity pass.

The neural encoder itself is not rewritten. The wrapper only fixes the ONNX-facing signature and removes host preprocessing from the graph boundary.

## Validation

Exporter settings:

- `torch.onnx.export(..., dynamo=True)`
- `external_data=True`
- `optimize=False`
- `do_constant_folding=False`
- FP32 weights and inputs

Runtime settings:

- ONNX path-based checker: `onnx.checker.check_model(str(path))`
- ONNX Runtime CPU execution provider
- ORT graph optimizations disabled

Observed smoke parity for `samples=20480`:

- wrapper vs original `AudioVAE.encode()` on already padded input: exact match.
- PyTorch wrapper vs ONNX Runtime: `max_abs_diff` about `8.6e-4`, `mean_abs_diff` about `7.9e-5`.
- The parity script default tolerance is therefore `1e-3` for encoder FP32 CPU checks.
