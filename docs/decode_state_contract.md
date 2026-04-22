# VoxCPM2DecodeStep State Contract

## Scope

`VoxCPM2DecodeStep` is one autoregressive decode step, not the full decode loop.

This page defines the state tensors that cross the prefill/decode boundary. It should be read together with `docs/module_boundaries.md` before changing `src/export/export_decode_step.py` or `src/runtime/pipeline.py`.

Host code owns:

- loop counter and max/min length policy
- stop decision from `stop_logits`
- retry/badcase policy
- generated feature list append
- final AudioVAEDecoder call
- streaming orchestration

The ONNX graph owns one neural step:

1. project `lm_hidden` and `residual_hidden` into DiT conditioning
2. run one fixed-size CFM/LocDiT sample using host-supplied `diffusion_noise`
3. encode the predicted audio feature patch
4. run one base MiniCPM step with explicit K/V tensors
5. run one residual MiniCPM step with explicit K/V tensors
6. return updated hidden states, prefix feature condition, stop logits, and extended caches

## Inputs

- `lm_hidden`: `float32[batch, hidden]`
- `residual_hidden`: `float32[batch, hidden]`
- `prefix_feat_cond`: `float32[batch, patch_size, feat_dim]`
- `base_k_cache`: `float32[base_layers, batch, kv_heads, cache_seq, head_dim]`
- `base_v_cache`: `float32[base_layers, batch, kv_heads, cache_seq, head_dim]`
- `base_cache_length`: `int64[1]`
- `residual_k_cache`: `float32[residual_layers, batch, kv_heads, cache_seq, head_dim]`
- `residual_v_cache`: `float32[residual_layers, batch, kv_heads, cache_seq, head_dim]`
- `residual_cache_length`: `int64[1]`
- `diffusion_noise`: `float32[batch, feat_dim, patch_size]`
- `cfg_value`: `float32[1]`

`base_cache_length` and `residual_cache_length` must equal the number of valid positions in their cache tensors. The first decode step consumes the cache tensors returned by `VoxCPM2Prefill`.

Current exported artifacts are single-batch by default (`batch=1`) and keep `cache_seq` dynamic. Multi-batch decode-step export is deferred until proven by ONNX Runtime parity.

## Outputs

- `pred_audio_feature`: `float32[batch, 1, patch_size, feat_dim]`
- `decoder_latent`: `float32[batch, feat_dim, patch_size]`
- `stop_logits`: `float32[batch, 2]`
- `next_lm_hidden`: `float32[batch, hidden]`
- `next_residual_hidden`: `float32[batch, hidden]`
- `next_prefix_feat_cond`: `float32[batch, patch_size, feat_dim]`
- `next_base_k_cache`: `float32[base_layers, batch, kv_heads, cache_seq + 1, head_dim]`
- `next_base_v_cache`: `float32[base_layers, batch, kv_heads, cache_seq + 1, head_dim]`
- `next_base_cache_length`: `int64[1]`
- `next_residual_k_cache`: `float32[residual_layers, batch, kv_heads, cache_seq + 1, head_dim]`
- `next_residual_v_cache`: `float32[residual_layers, batch, kv_heads, cache_seq + 1, head_dim]`
- `next_residual_cache_length`: `int64[1]`

`next_*_cache_length = *_cache_length + 1`.

## Mutable Cache Replacement

The official PyTorch code calls `MiniCPMModel.forward_step()`, which mutates `StaticKVCache` in Python. ONNX Runtime sessions cannot mutate Python objects, so the export wrapper uses an explicit tensor contract:

- cache tensors enter the graph with only valid positions
- the current step creates one new K/V position per layer
- the graph returns `torch.cat([cache, new_position], dim=seq)` as the next cache

This is a state representation change only. It keeps the same projections, RoPE, attention, MLP, normalization, FSQ, LocEnc, LocDiT/CFM, and stop-head math.

## Diffusion Noise

The official `UnifiedCFM.forward()` samples `torch.randn` inside the graph. The ONNX boundary takes `diffusion_noise` as an explicit input so CPU inference is deterministic and parity-testable. Host code must supply fresh noise for each decode step unless deterministic replay is desired.

`inference_timesteps` is an export-time graph parameter. It does not fix the outer decode loop; it only fixes the internal CFM solver step count for one generated audio feature patch.

## Stop Semantics

`stop_logits` are computed from the incoming `lm_hidden`, matching the official loop order. The graph still returns updated state tensors. If host code decides to stop, it should ignore the returned next state and finish with the generated feature sequence according to the runtime contract.

## Weight Scope

This graph includes only weights needed by one decode step: projection layers, LocDiT/CFM, LocEnc, base MiniCPM step, residual MiniCPM step, FSQ, and stop head. It does not include AudioVAE encode/decode or prefill graph logic.

Separate ONNX files for `VoxCPM2Prefill` and `VoxCPM2DecodeStep` will each contain their required transformer weights. This is an artifact-level duplication caused by separate ORT sessions, not an in-graph duplicate.

## Non-Goals

- No full decode loop in ONNX.
- No tokenizer or text normalization in ONNX.
- No WAV I/O or resampling in ONNX.
- No graph optimization or quantization.
- No removal of multilingual or reference-audio paths.

## Verification

```bash
python -B src/runtime/run_decode_step_ort.py --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx
python -B tests/parity/test_decode_step.py --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx
```
