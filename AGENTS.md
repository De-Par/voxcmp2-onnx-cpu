# AGENTS.md

- Target: full VoxCPM2 -> ONNX Runtime CPU-only
- Platforms: macOS arm64, Linux x86_64 / arm64, Windows x86_64 / arm64
- Do not simplify the model unless explicitly documented
- Do not remove multilingual path
- Do not remove reference-audio path
- Do not hardcode language
- Do not merge everything into one ONNX model
- FP32 is the correctness anchor; BF16 is a production performance target, not a storage-only experiment
- Always document blockers before changing math
- Keep changes minimal and reversible
