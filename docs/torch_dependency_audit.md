# PyTorch Dependency Audit

## Scope

Target runtime is CPU-only ONNX Runtime. PyTorch is allowed only in export and parity tooling, not in runtime orchestration.

Runtime path means:

- `src/runtime/session_factory.py`
- `src/runtime/pipeline.py`
- `src/cli/synthesize.py`
- `tests/smoke/test_cpu_only_runtime.py`

Tooling path means:

- `src/export/*`
- `tests/parity/*`
- upstream reference code in `third_party/VoxCPM`

## Findings

### Runtime Orchestration

No direct `torch` imports remain in `src/runtime`, `src/cli`, or `tests/smoke`.

One indirect dependency was found and removed:

- `src/runtime/pipeline.py` previously imported `transformers.LlamaTokenizerFast`.
- Importing `transformers` loaded PyTorch transitively.
- Runtime now uses `tokenizers.Tokenizer.from_file(tokenizer.json)` and preserves the VoxCPM2 multi-character Chinese token split in pure Python.

The smoke test now asserts that `torch` is not present in `sys.modules` before validation, after model path validation, after one-step synthesis, and after reference/prompt input assembly.

### Export Tooling

These files intentionally depend on PyTorch:

- `src/export/export_audio_vae_encoder.py`
- `src/export/export_audio_vae_decoder.py`
- `src/export/export_prefill.py`
- `src/export/export_decode_step.py`

Reason: they instantiate official PyTorch modules and call `torch.onnx.export(..., dynamo=True, external_data=True)`.

Classification: must stay outside ONNX Runtime deployment; allowed in export environment only.

### Parity Tooling

These files intentionally depend on PyTorch:

- `tests/parity/test_audio_vae_encoder.py`
- `tests/parity/test_audio_vae_decoder.py`
- `tests/parity/test_prefill.py`
- `tests/parity/test_decode_step.py`

Reason: they compare exported ONNX Runtime outputs against PyTorch reference wrappers.

Classification: must stay outside production runtime; allowed in validation environment only.

### Documentation Mentions

Docs mention `torch.onnx.export`, `torch.randn`, and `torch.cat` when describing source/export behavior. These are not runtime dependencies.

## Classification

### Must Remain Outside ONNX

- Text normalization and tokenization: host code responsibility. Runtime implementation is Python/tokenizers, not PyTorch.
- WAV I/O and resampling: host code responsibility via `soundfile`/`librosa`, not PyTorch.
- Orchestration loop: host code responsibility via NumPy arrays and ONNX Runtime sessions.
- Random diffusion noise: host code responsibility via NumPy RNG.

### Can Be Rewritten In NumPy/Python

Already done in runtime:

- token sequence assembly
- text/audio masks
- reference prefix construction
- prompt/reference audio feature alignment
- decode-loop state update
- diffusion noise creation
- latent feature reshape before AudioVAEDecoder

### Should Later Be Rewritten In C++

Candidates for a deployment runtime:

- tokenizer loading and VoxCPM2 Chinese-token split
- WAV loading/writing
- resampling
- decode-loop orchestration and stop policy
- ONNX Runtime session factory and tensor buffer reuse

These are not required for FP32 correctness; they are deployment hardening work.

## Acceptance Check

Use:

```bash
rg -n "\btorch\b|import torch|from torch" src/runtime src/cli tests/smoke
```

Expected result: no matches.

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tests/smoke/test_cpu_only_runtime.py
```

Expected result:

```text
cpu_only_runtime_smoke=ok
```
