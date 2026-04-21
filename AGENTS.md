# AGENTS.md

- Target: full VoxCPM2 -> ONNX Runtime CPU-only
- Platforms: macOS arm64, Linux x86_64, Windows x86_64
- Do not simplify the model unless explicitly documented
- Do not remove multilingual path
- Do not remove reference-audio path
- Do not hardcode language
- Do not merge everything into one ONNX model
- FP32 correctness first, BF16 only as a later experiment
- Always document blockers before changing math
- Keep changes minimal and reversible