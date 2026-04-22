# VoxCPM2DecodeStep State Contract

## Scope

`VoxCPM2DecodeStep` is one autoregressive decode step, not the full decode loop.

This page defines the production state tensors that cross the host/decode-step ONNX boundary. The contract is shared by FP32 and BF16 exports.

Host code owns:

- loop counter and max/min length policy
- stop decision from `stop_logits`
- fixed-capacity cache allocation
- applying one-position K/V cache updates
- generated feature list append
- final AudioVAEDecoder call
- streaming orchestration

The ONNX graph owns one neural step:

1. project `lm_hidden` and `residual_hidden` into DiT conditioning
2. run one fixed-size CFM/LocDiT sample using host-supplied `diffusion_noise`
3. encode the predicted audio feature patch
4. run one base MiniCPM step over valid cache prefix plus the new position
5. run one residual MiniCPM step over valid cache prefix plus the new position
6. return updated hidden states, prefix feature condition, stop logits, one-position K/V updates, and new lengths

## Fixed-Capacity Policy

The old decode-step contract grew K/V tensors every step:

```text
cache_seq -> cache_seq + 1 -> cache_seq + 2 -> ...
```

The production contract uses fixed-capacity cache tensors:

```text
max_cache_seq = prefill_length + effective_max_decode_steps
```

`*_current_length` gives the number of valid positions. The ONNX graph must not return a larger cache tensor. It returns only the new K/V position for each transformer layer; host code writes that update at `current_length` and then stores `next_*_current_length`.

This is a state representation change only. It does not change projection, RoPE, attention, MLP, normalization, FSQ, LocEnc, LocDiT/CFM, or stop-head math.

## Inputs

- `lm_hidden`: `float32[batch, hidden]`
- `residual_hidden`: `float32[batch, hidden]`
- `prefix_feat_cond`: `float32[batch, patch_size, feat_dim]`
- `base_k_cache`: `float32[base_layers, batch, kv_heads, max_cache_seq, head_dim]`
- `base_v_cache`: `float32[base_layers, batch, kv_heads, max_cache_seq, head_dim]`
- `base_current_length`: `int64[1]`
- `residual_k_cache`: `float32[residual_layers, batch, kv_heads, max_cache_seq, head_dim]`
- `residual_v_cache`: `float32[residual_layers, batch, kv_heads, max_cache_seq, head_dim]`
- `residual_current_length`: `int64[1]`
- `diffusion_noise`: `float32[batch, feat_dim, patch_size]`
- `cfg_value`: `float32[1]`

`base_current_length` and `residual_current_length` must be less than `max_cache_seq`.

The first decode step consumes the cache tensors returned by `VoxCPM2Prefill` after host code copies them into fixed-capacity buffers. Prefill keeps its own output names (`base_cache_length`, `residual_cache_length`) because those tensors describe initial valid lengths, not decode-step inputs.

## Outputs

- `pred_audio_feature`: `float32[batch, 1, patch_size, feat_dim]`
- `decoder_latent`: `float32[batch, feat_dim, patch_size]`
- `stop_logits`: `float32[batch, 2]`
- `next_lm_hidden`: `float32[batch, hidden]`
- `next_residual_hidden`: `float32[batch, hidden]`
- `next_prefix_feat_cond`: `float32[batch, patch_size, feat_dim]`
- `base_k_update`: `float32[base_layers, batch, kv_heads, 1, head_dim]`
- `base_v_update`: `float32[base_layers, batch, kv_heads, 1, head_dim]`
- `next_base_current_length`: `int64[1]`
- `residual_k_update`: `float32[residual_layers, batch, kv_heads, 1, head_dim]`
- `residual_v_update`: `float32[residual_layers, batch, kv_heads, 1, head_dim]`
- `next_residual_current_length`: `int64[1]`

`next_*_current_length = *_current_length + 1`.

## Host Update Rule

For each decode step:

```text
base_k_cache[:, :, :, base_current_length:base_current_length + 1, :] = base_k_update
base_v_cache[:, :, :, base_current_length:base_current_length + 1, :] = base_v_update
residual_k_cache[:, :, :, residual_current_length:residual_current_length + 1, :] = residual_k_update
residual_v_cache[:, :, :, residual_current_length:residual_current_length + 1, :] = residual_v_update
```

Then host code stores `next_base_current_length` and `next_residual_current_length`.

If host code stops on `stop_logits`, it may ignore returned state updates and finish with the generated feature sequence.

## Diffusion Noise

The official `UnifiedCFM.forward()` samples `torch.randn` inside the graph. The ONNX boundary takes `diffusion_noise` as an explicit input so CPU inference is deterministic and parity-testable. Host code must supply fresh noise for each decode step unless deterministic replay is desired.

`inference_timesteps` is an export-time graph parameter. It does not fix the outer decode loop; it only fixes the internal CFM solver step count for one generated audio feature patch.

## Precision

FP32 and BF16 exports use the same input names, output names, ranks, dynamic axes, fixed-capacity cache semantics, and host update rule. BF16 may change internal compute dtype according to `docs/export_precision_profiles.md`; it must not change the state contract.

## Non-Goals

- No full decode loop in ONNX.
- No tokenizer or text normalization in ONNX.
- No WAV I/O or resampling in ONNX.
- No graph optimization or quantization.
- No removal of multilingual or reference-audio paths.
- No custom ops.

## Verification

```bash
python -B src/runtime/run_decode_step_ort.py \
  --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx \
  --cache-seq 16 \
  --max-cache-seq 64

python -B tests/parity/test_decode_step.py \
  --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx \
  --precision fp32 \
  --cache-seq 16 \
  --max-cache-seq 64
```
