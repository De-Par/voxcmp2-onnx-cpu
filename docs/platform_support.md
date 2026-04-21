# Platform Support

## Target Matrix

Runtime target: VoxCPM2 ONNX Runtime CPU only.

This page records code-level platform support and the checks required before claiming a release on each target platform class.

| Platform | Architecture | Status |
| --- | --- | --- |
| macOS | arm64 | Supported and locally smoke-tested. |
| Linux | x86_64 | Supported by contract; requires target smoke run. |
| Linux | arm64 | Supported by contract; requires target smoke run. |
| Windows | x86_64 | Supported by contract; requires target smoke run. |
| Windows | arm64 | Supported by contract; requires target smoke run. |

## Runtime Constraints

- ONNX Runtime sessions are created only with `CPUExecutionProvider`.
- GPU, CoreML, MPS, CUDA, DirectML, and provider fallback are not allowed.
- Runtime orchestration uses Python, NumPy, SciPy, `tokenizers`, `huggingface_hub`, and ONNX Runtime.
- Runtime path does not import PyTorch, Transformers, `soundfile`, or `librosa`.
- Paths use `pathlib.Path`; no POSIX-only path joining or shell invocation is required by runtime code.
- ONNX artifacts may use external data files and must remain next to their `.onnx` model files.

## Dependency Portability Notes

- ONNX Runtime CPU is the only inference package; install target is `onnxruntime`.
- Current ONNX Runtime PyPI wheels cover Linux x86_64/aarch64, macOS arm64, and Windows x86_64/arm64 for supported CPython versions.
- SciPy is used for WAV I/O and resampling to avoid `soundfile`/`librosa` runtime dependencies. Current SciPy wheels cover the target platform classes for supported CPython versions.
- `tokenizers` is used directly from `tokenizer.json`, avoiding the heavier `transformers` import path. Current `tokenizers` wheels include Linux x86_64/aarch64, macOS arm64, and Windows x86_64/arm64.

## Host-Code Boundaries

These remain outside ONNX on every platform:

- text normalization
- tokenizer loading and token ID assembly
- WAV loading/writing
- resampling
- orchestration loop
- stop policy
- random diffusion noise generation

## Acceptance Criteria

- `src/runtime/session_factory.py` rejects unavailable or forbidden execution providers.
- `src/runtime/pipeline.py` can run text-only synthesis without loading PyTorch or accelerator providers.
- Reference and prompt WAV paths use host WAV/resampling code and stay available for controllable clone and ultimate clone modes.
- The same exported module set is used on every platform; no platform-specific ONNX graph is required for v1.
- Release signoff requires the smoke test below on each target platform class.

## Verification Commands

```bash
python -B -m py_compile src/runtime/session_factory.py src/runtime/pipeline.py src/cli/synthesize.py tests/smoke/test_cpu_only_runtime.py
python -B tests/smoke/test_cpu_only_runtime.py
python -B src/cli/synthesize.py --text "Hello from VoxCPM2." --output artifacts/runtime_smoke.wav --max-steps 1 --mode text_only
rg -n "import torch|from torch|soundfile|librosa|transformers|CUDAExecutionProvider|CoreMLExecutionProvider|MPSExecutionProvider" src/runtime src/cli tests/smoke
```

Expected results:

- smoke prints `cpu_only_runtime_smoke=ok`
- CLI writes a WAV file
- grep has no runtime imports of PyTorch, Transformers, `soundfile`, or `librosa`; provider names may appear only in the explicit forbidden-provider list

## Non-Goals

- GPU or vendor accelerator support.
- Platform-specific graph optimization.
- Quantization or BF16.
- Replacing host tokenizer, WAV I/O, resampling, or orchestration with ONNX graphs.
