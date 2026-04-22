# Architecture And Runtime Contract

This page is the runtime architecture contract for the VoxCPM2 CPU-only ONNX port.

## Scope

- Model: VoxCPM2.
- Runtime: ONNX Runtime `CPUExecutionProvider` only.
- Platforms: macOS arm64, Linux x86_64 / arm64, Windows x86_64 / arm64.
- V1 modes: text-to-speech, voice design, controllable clone, ultimate clone.
- Deferred: streaming.

"Fully ONNX" means every neural module needed by the supported modes is exported to ONNX. Text normalization, tokenization, WAV I/O, resampling, random noise creation, orchestration, stop policy, and WAV writing remain host code.

## Feature Matrix

| Feature | V1 status | Requirement |
|---|---|---|
| Text-to-speech | required | Neural work in ONNX; host code handles text and loop orchestration. |
| Voice design | required | Preserve the official VoxCPM2 path; style text remains part of host token assembly. |
| Controllable clone | required | Preserve reference-audio path. |
| Ultimate clone | required | Preserve reference and prompt-audio path. |
| Multilingual path | required | Do not hardcode language. |
| FP32 | required | Correctness anchor and production artifact family. |
| BF16 | required | Parallel production artifact family with the same runtime path and feature coverage. |
| Streaming | v2 | No v1 export/runtime requirement. |
| Quantization | non-goal | Not part of v1. |
| GPU/CoreML/CUDA/DirectML/MPS | non-goal | CPU provider only. |
| Single merged ONNX graph | non-goal | Keep module boundaries separate. |

## Official Generate Path

The traced official path is:

```text
VoxCPM.generate()
  -> VoxCPM._generate()
  -> optional VoxCPM2Model.build_prompt_cache()
  -> VoxCPM2Model._generate_with_prompt_cache()
  -> VoxCPM2Model._inference()
  -> AudioVAEV2.decode()
```

Mode mapping:

- `text_only`: no prompt cache, equivalent to the plain zero-shot path.
- `voice_design`: same tensor structure as text-only; design text is encoded in the text prompt.
- `controllable_clone`: `reference_wav_path` activates reference prefix construction.
- `ultimate_clone`: `reference_wav_path` plus `prompt_wav_path`/`prompt_text` activates continuation-style prompt assembly.

Use the trace tool before changing module boundaries:

```bash
python -B src/parity/trace_generate.py \
  --model-path openbmb/VoxCPM2 \
  --mode plain_tts \
  --text "Hello from VoxCPM2." \
  --trace-output traces/plain_tts.jsonl
```

Trace output is compact JSONL with stage names, Python module/function names, shapes, dtypes, and reference/prompt path flags. Full tensor values are never logged.

## Module Boundaries

The runtime uses four separate ONNX sessions.

### `AudioVAEEncoder`

Purpose: reference/prompt waveform to latent audio features.

- input `waveform`: `float32[batch, 1, samples]`
- output `latent`: `float32[batch, 64, latent_steps]`

Host code loads WAV, converts to mono, resamples to the model encode rate, pads samples to `audio_vae.chunk_size`, and reshapes the latent into `[audio_steps, patch_size=4, feat_dim=64]` for prompt assembly.

### `VoxCPM2Prefill`

Purpose: full prompt pass over text/audio-aligned sequence.

Inputs:

- `text_tokens`: `int64[batch, seq]`
- `text_mask`: `float32[batch, seq]`
- `audio_features`: `float32[batch, seq, patch_size=4, feat_dim=64]`
- `audio_mask`: `float32[batch, seq]`

Outputs:

- `lm_hidden`: `float32[batch, hidden]`
- `residual_hidden`: `float32[batch, hidden]`
- `prefix_feat_cond`: `float32[batch, patch_size, feat_dim]`
- `base_k_cache`, `base_v_cache`, `base_cache_length`
- `residual_k_cache`, `residual_v_cache`, `residual_cache_length`

Mode-specific assembly is host-side:

- `text_only`: target text plus audio-start control tokens; no audio mask positions.
- `voice_design`: same structure as text-only; design text goes through tokenizer.
- `controllable_clone`: host prepends reference audio marker/features.
- `ultimate_clone`: host combines reference prefix with prompt text/audio continuation.

### `VoxCPM2DecodeStep`

Purpose: exactly one autoregressive audio-feature step. The outer decode loop stays in host code.

Inputs:

- `lm_hidden`: `float32[batch, hidden]`
- `residual_hidden`: `float32[batch, hidden]`
- `prefix_feat_cond`: `float32[batch, patch_size, feat_dim]`
- fixed-capacity base/residual K/V caches
- `base_current_length`, `residual_current_length`: `int64[1]`
- `diffusion_noise`: `float32[batch, feat_dim, patch_size]`
- `cfg_value`: `float32[1]`

Outputs:

- `pred_audio_feature`: `float32[batch, 1, patch_size, feat_dim]`
- `decoder_latent`: `float32[batch, feat_dim, patch_size]`
- `stop_logits`: `float32[batch, 2]`
- next hidden states and prefix feature condition
- one-position K/V updates and next cache lengths

The graph owns one neural step: DiT conditioning, fixed-size CFM/LocDiT solve, feature encode, base LM step, residual LM step, stop head, and tensor state outputs.

### `AudioVAEDecoder`

Purpose: generated latent feature sequence to waveform.

- input `latent`: `float32[batch, 64, latent_steps]`
- input `sr_cond`: `int32[batch]`
- output `waveform`: `float32[batch, 1, samples]`

For non-streaming v1, host concatenates generated features and calls the decoder once.

## Fixed-Capacity Decode Cache

The old experimental decode-step contract grew cache tensors every step:

```text
cache_seq -> cache_seq + 1 -> cache_seq + 2
```

The production contract uses fixed-capacity tensors:

```text
max_cache_seq = prefill_length + effective_max_decode_steps
```

Input cache shapes:

```text
base_k_cache:     [base_layers, batch, kv_heads, max_cache_seq, head_dim]
base_v_cache:     [base_layers, batch, kv_heads, max_cache_seq, head_dim]
residual_k_cache: [residual_layers, batch, kv_heads, max_cache_seq, head_dim]
residual_v_cache: [residual_layers, batch, kv_heads, max_cache_seq, head_dim]
```

Output update shapes:

```text
base_k_update:     [base_layers, batch, kv_heads, 1, head_dim]
base_v_update:     [base_layers, batch, kv_heads, 1, head_dim]
residual_k_update: [residual_layers, batch, kv_heads, 1, head_dim]
residual_v_update: [residual_layers, batch, kv_heads, 1, head_dim]
```

Host update rule:

```text
base_k_cache[:, :, :, base_current_length:base_current_length + 1, :] = base_k_update
base_v_cache[:, :, :, base_current_length:base_current_length + 1, :] = base_v_update
residual_k_cache[:, :, :, residual_current_length:residual_current_length + 1, :] = residual_k_update
residual_v_cache[:, :, :, residual_current_length:residual_current_length + 1, :] = residual_v_update
```

The traffic goal is to remove output-cache growth. Old output payload per step was `2 * K * (S + 1)`. New output payload is `2 * K`, where `K = (base_layers + residual_layers) * batch * kv_heads * head_dim`.

## Runtime Rules

- Use only `CPUExecutionProvider`.
- Do not fall back to CUDA, CoreML, DirectML, MPS, or any accelerator provider.
- Load sessions lazily.
- Keep ONNX external-data files next to their `.onnx` files.
- Fail fast with actionable errors for missing modules or external data.
- Do not change model math in runtime code.
- Do not insert silent runtime precision conversions.
- FP32 and BF16 select different artifact paths, not different runtime semantics.

## Runtime Dependency Audit

Runtime path:

- `src/runtime/session_factory.py`
- `src/runtime/pipeline.py`
- `src/cli/synthesize.py`
- `tests/smoke/test_cpu_only_runtime.py`

Allowed runtime dependencies are Python, NumPy, SciPy, `tokenizers`, `huggingface_hub`, and ONNX Runtime. Runtime orchestration must not import PyTorch, Transformers, `soundfile`, or `librosa`.

PyTorch is allowed only in:

- `src/export/*`
- `tests/parity/*`
- upstream reference code in `third_party/VoxCPM`

Audit command:

```bash
rg -n "\btorch\b|import torch|from torch|soundfile|librosa|transformers" src/runtime src/cli tests/smoke
```

Expected result: no runtime dependency imports. Provider names may appear only in explicit forbidden-provider validation.

## Platform Signoff

For each target platform class, run:

```bash
python -B -m py_compile src/runtime/session_factory.py src/runtime/pipeline.py src/cli/synthesize.py tests/smoke/test_cpu_only_runtime.py
python -B tests/smoke/test_cpu_only_runtime.py
python -B src/cli/synthesize.py --text "Hello from VoxCPM2." --output artifacts/samples/runtime_sample.wav --mode text_only
rg -n "CUDAExecutionProvider|CoreMLExecutionProvider|MPSExecutionProvider|DirectMLExecutionProvider" src/runtime src/cli tests/smoke
```

Expected results:

- smoke prints `cpu_only_runtime_smoke=ok`
- CLI writes a WAV file
- every ORT session reports only `CPUExecutionProvider`

## Acceptance Criteria

- Every required mode maps to export and runtime orchestration requirements.
- Host/ONNX boundaries match `src/contracts/module_schemas.py`.
- Runtime validates ONNX paths before synthesis.
- Decode-step state uses fixed-capacity cache tensors.
- Multilingual and reference-audio paths remain active.
- FP32 and BF16 artifacts use the same runtime path.
- Missing model files, incompatible opset/runtime versions, and shape/dtype mismatches fail with actionable errors.
