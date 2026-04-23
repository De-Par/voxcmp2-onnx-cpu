# Integration API

This page shows the small application-facing API for embedding the CPU-only
VoxCPM2 ONNX runtime in another Python project.

The API is intentionally thin. It does not fork runtime behavior, change model
math, or hide the production contracts. It wraps `src.runtime.pipeline` with a
portable class that selects FP32 or BF16 artifacts and exposes the four supported
VoxCPM2 modes.

## Scope

Supported:

- `text_only`
- `voice_design`
- `controllable_clone`
- `ultimate_clone`
- FP32 and BF16 production ONNX artifacts
- ONNX Runtime `CPUExecutionProvider` only

Still host code:

- tokenizer and sequence assembly
- WAV read/write and resampling
- reference/prompt orchestration
- decode loop and stop policy
- diffusion noise creation

## Basic Use

```python
from app import VoxCPM2Onnx, VoxCPM2OnnxConfig

tts = VoxCPM2Onnx(VoxCPM2OnnxConfig(precision="bf16"))
tts.validate()

result = tts.synthesize(
    "Hello from VoxCPM2.",
    mode="text_only",
    output_wav="artifacts/samples/app_text_only.wav",
)

print(result.metadata.decode_steps, result.metadata.stop_reason)
```

## Voice Design

```python
from app import VoxCPM2Onnx, VoxCPM2OnnxConfig

tts = VoxCPM2Onnx(VoxCPM2OnnxConfig(precision="bf16"))
result = tts.synthesize(
    "Hello from VoxCPM2.",
    mode="voice_design",
    voice_design="pretty girl with sugar voice, slow",
    output_wav="artifacts/samples/app_voice_design.wav",
)
```

## Controllable Clone

```python
from app import VoxCPM2Onnx, VoxCPM2OnnxConfig

tts = VoxCPM2Onnx(VoxCPM2OnnxConfig(precision="bf16"))
result = tts.synthesize(
    "Hello from VoxCPM2.",
    mode="controllable_clone",
    reference_wav_path="artifacts/reference.wav",
    output_wav="artifacts/samples/app_controllable_clone.wav",
)
```

## Ultimate Clone

```python
from app import VoxCPM2Onnx, VoxCPM2OnnxConfig

tts = VoxCPM2Onnx(VoxCPM2OnnxConfig(precision="bf16"))
result = tts.synthesize(
    "Continue this voice.",
    mode="ultimate_clone",
    reference_wav_path="artifacts/reference.wav",
    prompt_wav_path="artifacts/prompt.wav",
    prompt_text="This is the prompt text.",
    output_wav="artifacts/samples/app_ultimate_clone.wav",
)
```

## Demo CLI

```bash
python -B app/demo.py \
  --precision bf16 \
  --text "Hello from VoxCPM2." \
  --mode text_only \
  --output artifacts/samples/app_demo.wav
```

Voice design:

```bash
python -B app/demo.py \
  --precision bf16 \
  --text "Hello from VoxCPM2." \
  --mode voice_design \
  --voice-design "pretty girl with sugar voice, slow" \
  --output artifacts/samples/app_voice_design.wav
```

## Configuration

`VoxCPM2OnnxConfig` defaults to the production BF16 artifact family:

```python
VoxCPM2OnnxConfig(
    precision="bf16",
    model_path="openbmb/VoxCPM2",
    onnx_root="models/onnx",
    graph_optimization_level="all",
    execution_mode="sequential",
    intra_op_num_threads=8,
    inter_op_num_threads=1,
)
```

Use `precision="fp32"` when validating correctness against the FP32 anchor.
Use BF16 for the production performance path when the local ONNX Runtime CPU
build supports the exported graph efficiently.

## Return Value

`synthesize()` returns `src.runtime.pipeline.SynthesisResult`:

- `waveform`: mono `numpy.float32` waveform
- `metadata.decode_steps`: accepted decode steps
- `metadata.stop_reason`: `stop_logits`, `max_steps`, or `safety_max_steps`
- `metadata.requested_max_steps`
- `metadata.effective_max_steps`
- `metadata.min_steps`

## Integration Rules

- Create one `VoxCPM2Onnx` instance and reuse it for repeated requests.
- Keep `max_steps=0` for normal generation; it runs until stop logits with a safety cap.
- Pass explicit `reference_wav_path`, `prompt_wav_path`, and `prompt_text` only for modes that require them.
- Do not run several benchmark variants in one process when collecting performance numbers; benchmark one precision/API path at a time.
- Do not enable GPU/CoreML/DirectML providers. This project targets CPU-only ONNX Runtime.
