# VoxCPM2Prefill Export Notes

## Boundary

`src/export/export_prefill.py` exports only the neural prefill section from `VoxCPM2Model._inference()`:

1. `feat_encoder(audio_features)` and `enc_to_lm_proj`.
2. token embedding and text/audio embedding merge by masks.
3. base MiniCPM full-prefix pass.
4. `fsq_layer` replacement of audio positions.
5. residual MiniCPM full-prefix pass.
6. tensorized K/V caches for the next decode step.

This page records what is inside the prefill ONNX graph, what stays in host code, and which upstream operations were intentionally isolated instead of rewritten.

The wrapper does not call `StaticKVCache.fill_caches()` and does not enter the autoregressive loop. It returns cache tensors explicitly:

- `base_k_cache`, `base_v_cache`: `float32[base_layers, batch, kv_heads, seq, head_dim]`
- `base_cache_length`: `int64[1]`
- `residual_k_cache`, `residual_v_cache`: `float32[residual_layers, batch, kv_heads, seq, head_dim]`
- `residual_cache_length`: `int64[1]`

## Inputs By Mode

Common inputs:

- `text_tokens`: `int64[batch, seq]`, produced by host tokenizer with VoxCPM2 special tokens.
- `text_mask`: `float32[batch, seq]`, `1` for text/control-token positions.
- `audio_features`: `float32[batch, seq, patch_size, feat_dim]`, reference/prompt audio features aligned to the token sequence.
- `audio_mask`: `float32[batch, seq]`, `1` for positions carrying audio features.

Mode-specific host assembly:

- `plain_tts`: target text tokens plus audio-start token; `audio_features` are zeros; `audio_mask` is all zero.
- `voice_design`: same tensor structure as `plain_tts`; style/control text is part of `text_tokens`. No language is hardcoded by the export path.
- `controllable_clone`: host prepends `[ref_audio_start, reference audio features, ref_audio_end]` before target text. Reference audio positions have `text_mask=0`, `audio_mask=1`; marker and target text positions have `text_mask=1`, `audio_mask=0`.
- `ultimate_clone`: host may combine the reference prefix with continuation prompt text/audio. Prompt audio is appended as audio-feature positions, preserving the reference path and multilingual token path.

Text normalization, tokenizer behavior, language handling, WAV loading/resampling, AudioVAEEncoder calls, prompt/reference sequence policy, decode orchestration, stopping, and WAV writing remain in host code.

## ONNX Blockers And Risks

- `StaticKVCache` mutation: official inference mutates Python cache objects through `fill_caches()`. The wrapper avoids this by returning stacked K/V tensors. This is a boundary change, not a math change.
- Prompt continuation context: official `_inference()` uses `.item()`, `nonzero()`, Python lists, and `split()` to seed streaming/continuation context. This stays host-side and is not part of prefill ONNX.
- MiniCPM cache return type: upstream returns a Python `list[tuple[key, value]]`. The wrapper immediately stacks it into named tensor outputs.
- `scaled_dot_product_attention(enable_gqa=True)`: this is the main exporter/runtime risk. The wrapper does not replace attention math; if export fails on a PyTorch/ONNX version, the minimal next step is an export-only attention wrapper that expands grouped KV heads before SDPA or lowers attention to matmul/softmax.
- LongRoPE cache indexing: RoPE uses dynamic `arange` and cached `cos/sin` indexing by sequence length. This should stay inside ONNX if exported cleanly; failures should be localized before changing math.
- FP32 target: upstream config is BF16, but the export loader casts the full prefill path to FP32 for CPU correctness first. BF16 and quantization are non-goals for this path.

## Acceptance Criteria

- `torch.onnx.export(..., dynamo=True, external_data=True)` is used.
- ONNX graph has only CPU-targeted FP32 neural inputs except `text_tokens` and cache lengths.
- Exported inputs/outputs are named and logged with dtype and dynamic/static dimensions.
- `onnx.checker.check_model(str(path))` is used by the runtime script for path-based checking.
- Parity compares PyTorch wrapper outputs against ONNX Runtime CPU outputs without logging full tensors.

## Non-Goals

- No monolithic VoxCPM2 export.
- No tokenization or text normalization in ONNX.
- No audio I/O, resampling, or WAV writing in ONNX.
- No hardcoded language.
- No removal of reference or multilingual paths.
- No graph optimization, quantization, or math rewrite in the first export path.

## Verification

```bash
python -B src/runtime/run_prefill_ort.py --onnx-path artifacts/prefill/voxcpm2_prefill.onnx --mode plain_tts
python -B tests/parity/test_prefill.py --onnx-path artifacts/prefill/voxcpm2_prefill.onnx
```
