# VoxCPM2 ONNX Module Boundaries

## Goal

Define CPU-only ONNX Runtime boundaries for VoxCPM2 without creating one monolithic graph.

The proposed v1 export keeps neural work in ONNX and keeps orchestration in host code. It follows the traced official path:

`VoxCPM.generate()` -> `VoxCPM._generate()` -> optional `build_prompt_cache()` -> `_generate_with_prompt_cache()` -> `_inference()` -> `AudioVAE.decode()`.

This page is the main architectural reference for engineers adding export or runtime work. Typed Python schemas for the same boundaries live in `src/contracts/module_schemas.py`.

## Host Code

These remain outside ONNX:

- text normalization
- tokenizer and special-token sequence assembly
- audio loading and resampling
- prompt/reference cache construction policy
- autoregressive orchestration loop
- stop decision from stop logits
- retry/badcase policy
- random seed/noise creation for diffusion sampling
- WAV writing

Host code prepares tensors, calls ONNX sessions, appends generated audio features, and decides when to stop.

## Proposed ONNX Modules

### AudioVAEEncoder

Purpose: encode already-loaded and resampled reference/prompt audio into continuous audio latents.

Input:

- `waveform`: `float32[B, 1, samples]`, mono audio already resampled to the model encode sample rate.

Output:

- `latent`: `float32[B, latent_dim=64, latent_steps]`.

Host code reshapes `latent` into VoxCPM2 patch features `[audio_steps, patch_size=4, latent_dim=64]` before prompt-cache assembly.

### VoxCPM2Prefill

Purpose: run the non-iterative prefix pass over text tokens, masks, and prompt/reference audio features.

Input:

- `text_tokens`: `int64[B, seq]`
- `text_mask`: `float32[B, seq]`
- `audio_features`: `float32[B, seq, patch_size=4, latent_dim=64]`
- `audio_mask`: `float32[B, seq]`

Output:

- `lm_hidden`: `float32[B, hidden=2048]`
- `residual_hidden`: `float32[B, hidden=2048]`
- `prefix_feat_cond`: `float32[B, patch_size=4, latent_dim=64]`
- base LM KV cache tensors
- residual LM KV cache tensors

Cache/state tensors between prefill and decode step:

- `base_k_cache`: `float32[base_layers=28, B, kv_heads, cache_seq, head_dim]`
- `base_v_cache`: `float32[base_layers=28, B, kv_heads, cache_seq, head_dim]`
- `base_cache_length`: `int64[1]`
- `residual_k_cache`: `float32[residual_layers=8, B, kv_heads, cache_seq, head_dim]`
- `residual_v_cache`: `float32[residual_layers=8, B, kv_heads, cache_seq, head_dim]`
- `residual_cache_length`: `int64[1]`

The trace showed `kv_heads=2`, `head_dim=128` for the current weights. Export code must read these from config, not hardcode them.

The PyTorch implementation stores cache internally as `StaticKVCache` with a combined leading K/V dimension. The ONNX boundary should expose K and V as separate tensors because ORT sessions cannot mutate Python objects and separate tensors make append/update behavior explicit.

### VoxCPM2DecodeStep

Purpose: perform one autoregressive audio-feature step.

Input:

- `lm_hidden`: `float32[B, 2048]`
- `residual_hidden`: `float32[B, 2048]`
- `prefix_feat_cond`: `float32[B, 4, 64]`
- base LM KV cache tensors and `base_cache_length`
- residual LM KV cache tensors and `residual_cache_length`
- `diffusion_noise`: `float32[B, 64, 4]`
- `cfg_value`: `float32[1]`

Output:

- `pred_audio_feature`: `float32[B, 1, 4, 64]`, appended by host to the generated feature list.
- `decoder_latent`: `float32[B, 64, 4]`, direct input chunk for AudioVAEDecoder when decoding step output.
- `stop_logits`: `float32[B, 2]`, host applies argmax and `min_len` policy.
- updated `lm_hidden`
- updated `residual_hidden`
- updated `prefix_feat_cond`
- updated base LM KV cache tensors and `base_cache_length + 1`
- updated residual LM KV cache tensors and `residual_cache_length + 1`

`diffusion_noise` is an explicit input so ONNX inference is deterministic and comparable. This replaces the in-graph `torch.randn` call with host-supplied noise while preserving the diffusion math.

`inference_timesteps` is fixed at export time for a given decode-step ONNX artifact. This does not put the outer autoregressive loop into ONNX; host code still calls the one-step graph repeatedly.

### AudioVAEDecoder

Purpose: decode generated latent features to waveform.

Input:

- `latent`: `float32[B, 64, latent_steps]`
- `sr_cond`: `int32[B]`, explicit output sample-rate condition when the checkpoint uses sample-rate conditioning.

Output:

- `waveform`: `float32[B, 1, samples]`

For non-streaming v1, host concatenates generated feature patches into `[B, 64, total_patch_steps]`, calls the decoder once, and trims continuation context audio outside ONNX.

## Why This Cut

This split follows the traced execution without changing model behavior:

- AudioVAE encode/decode are natural neural boundaries and are needed only when reference/prompt audio or final waveform conversion is used.
- Prefill and decode-step mirror the transformer cache lifecycle: prefill creates caches from the full prompt, decode-step consumes and returns updated cache/state tensors.
- The host loop remains visible and testable, which is necessary for stop policy, retry policy, mode differences, and future streaming.
- Small projections, scalar quantization, stop head, LocEnc, MiniCPM LMs, and LocDiT sampling stay inside prefill/decode-step wrappers where they are used, avoiding a fragile graph-per-layer export.
- The design avoids a single monolithic ONNX graph while keeping the number of ORT sessions small enough for CPU-only execution.

## Verification

Use the per-module runtime checkers after export:

```bash
python -B src/runtime/run_audio_vae_encoder_ort.py --onnx-path models/onnx/fp32/audio_vae_encoder/audio_vae_encoder.onnx
python -B src/runtime/run_audio_vae_decoder_ort.py --onnx-path models/onnx/fp32/audio_vae_decoder/audio_vae_decoder.onnx
python -B src/runtime/run_prefill_ort.py --onnx-path models/onnx/fp32/prefill/voxcpm2_prefill.onnx --mode plain_tts
python -B src/runtime/run_decode_step_ort.py --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx
```

## Non-Goals

- No single merged VoxCPM2 ONNX model.
- No tokenizer, text normalization, audio file I/O, resampling, orchestration loop, or WAV writing in ONNX.
- No BF16 or quantized boundary contract for v1; schemas use FP32 neural tensors for CPU correctness first.
