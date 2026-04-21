# VoxCPM2 Generate Path Report

## Entry Point

Official API entry is `VoxCPM.generate()` in `third_party/VoxCPM/src/voxcpm/core.py:174`. It delegates to `VoxCPM._generate()` (`core.py:180`), which validates paths, optionally denoises and normalizes host-side inputs, builds a prompt cache when prompt/reference audio is present, then calls `VoxCPM2Model._generate_with_prompt_cache()` (`core.py:283`).

## Call Map

Text to waveform path:

1. `VoxCPM.generate()` -> `VoxCPM._generate()`.
2. Host preprocessing: text whitespace cleanup, optional text normalization, optional denoising.
3. Optional prompt/reference path: `VoxCPM2Model.build_prompt_cache()` -> `_encode_wav()` -> `AudioVAEV2.encode()`.
4. Sequence assembly: `_generate_with_prompt_cache()` tokenizes text, inserts audio/ref markers, builds `text_token`, `text_mask`, `audio_feat`, `audio_mask`.
5. Core neural loop: `_inference()` runs LocEnc, base MiniCPM LM prefill, residual MiniCPM LM prefill, iterative LocDiT/CFM sampling, stop head, and LM `forward_step()` updates.
6. Waveform decode: `_generate_with_prompt_cache()` calls `AudioVAEV2.decode()` and returns waveform to `core._generate()`, which converts it to NumPy.

Mode mapping:

- plain TTS: no prompt cache, `mode=zero_shot`.
- voice design: same code path as plain TTS; design prompt is encoded in text parentheses.
- controllable clone: `reference_wav_path` -> `mode=reference`, reference prefix is built by `_make_ref_prefix()`.
- ultimate clone: `prompt_wav_path + prompt_text` -> continuation path; with `reference_wav_path` it becomes `mode=ref_continuation`.

## Candidate ONNX Boundaries

- `audio_vae_encoder`: `AudioVAEV2.encode()` for reference/prompt WAV to latent features.
- `locenc`: `VoxCPMLocEnc.forward()` for audio latent patch embedding.
- `base_lm_prefill` and `base_lm_step`: `MiniCPMModel.forward()` / `forward_step()` for text-semantic LM.
- `residual_lm_prefill` and `residual_lm_step`: residual acoustic LM prefill and autoregressive step.
- `projection_and_fsq`: small neural projections plus `ScalarQuantizationLayer`; can stay separate or be grouped with adjacent LM modules after parity checks.
- `locdit_cfm`: `UnifiedCFM.forward()` plus `VoxCPMLocDiTV2.forward()` for diffusion patch sampling.
- `stop_head`: stop predictor layers.
- `audio_vae_decoder`: `AudioVAEV2.decode()` for latent features to waveform.

Host code remains responsible for text normalization, tokenization, WAV I/O, resampling through `librosa`, prompt-cache orchestration, retry policy, and mode selection.

## Trace Tool

`src/parity/trace_generate.py` runs the official local implementation and writes compact JSONL events:

```bash
python src/parity/trace_generate.py \
  --model-path /path/to/VoxCPM2 \
  --mode plain_tts \
  --text "Hello from VoxCPM2." \
  --trace-output traces/plain_tts.jsonl \
  --device cpu
```

The log records stage name, Python module/function, input/output shapes, dtype, device, and whether reference/prompt pathways are active. It does not log full tensor contents.

## Smoke Validation

Weights were found in the local Hugging Face cache and the tracer was run on CPU with `inference_timesteps=1`, `max_len=1`:

- `traces/plain_tts_smoke.jsonl`: 48 events, no reference/prompt pathway.
- `traces/voice_design_smoke.jsonl`: 48 events, no reference/prompt pathway; design text is part of `text`.
- `traces/controllable_clone_smoke.jsonl`: 56 events, reference pathway active.
- `traces/ultimate_clone_smoke.jsonl`: 60 events, reference and prompt pathways active.
