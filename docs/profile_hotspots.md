# ONNX Runtime Profiling Hotspots

This page documents the diagnostic-only profiling workflow for the current VoxCPM2 CPU-only ONNX runtime.

Profiling must not change:

- model math
- export wrappers
- ONNX files
- runtime stop/cache semantics
- BF16/FP32 policy

## One-Command Profile

Run a profiled controllable-clone case. This exercises all four ONNX sessions:

- `audio_vae_encoder`
- `prefill`
- `decode_step`
- `audio_vae_decoder`

```bash
python -B tools/profile/run_profiled_bench.py \
  --output-dir artifacts/profile \
  --cases controllable_clone_short \
  --top-n 20 \
  --onnx-graph-optimization all \
  --onnx-execution-mode sequential \
  --onnx-intra-op-threads 8 \
  --onnx-inter-op-threads 1
```

Outputs:

- raw ORT Chrome-trace JSON files: `artifacts/profile/profiles/*.json`
- combined JSON report: `artifacts/profile/profiled_bench.json`
- Markdown hotspot report: `artifacts/profile/hotspots.md`

## Parse Existing Profiles

```bash
python -B tools/profile/parse_ort_profile.py \
  --profile-dirs artifacts/profile/profiles \
  --json-report artifacts/profile/parsed_hotspots.json \
  --markdown-report artifacts/profile/hotspots.md \
  --top-n 20
```

## Report Sections

The generated Markdown report contains:

- top 20 hottest nodes by total latency
- top op types by total latency
- Cast hotspots
- cache-related hotspots
- a 3-5 item shortlist of the most expensive slowdown causes
- code-site hints for:
  - `src/export/export_prefill.py`
  - `src/export/export_decode_step.py`
  - `src/runtime/pipeline.py`

## Code-Site Mapping

The parser maps profile nodes by session/module and op hints:

| module | likely source |
|---|---|
| `prefill` | `src/export/export_prefill.py::VoxCPM2PrefillWrapper.forward` |
| `prefill` cache outputs | `src/export/export_prefill.py::VoxCPM2PrefillWrapper._stack_cache` |
| `decode_step` | `src/export/export_decode_step.py::VoxCPM2DecodeStepWrapper.forward` |
| `decode_step` attention nodes | `src/export/export_decode_step.py::VoxCPM2DecodeStepWrapper._attention_step` |
| `decode_step` cache movement | `src/runtime/pipeline.py::VoxCPM2OnnxPipeline.synthesize_with_metadata` |
| `audio_encoder` | `src/runtime/pipeline.py::VoxCPM2OnnxPipeline._encode_wav` |
| `audio_decoder` | `src/runtime/pipeline.py::audio_decoder.run` |

This mapping is diagnostic. It does not prove that a code site should be changed; it only narrows the next inspection target.

## Expected Shortlist Categories

The parser derives the shortlist from actual profile data. Common categories to expect:

1. Dominant op type, usually large matrix/attention work in `decode_step`.
2. Explicit cache/state movement. Production decode_step now uses fixed-capacity caches and one-position update outputs; any remaining cache `Concat`/shape hotspots should be treated as diagnostic targets.
3. Cast or CastLike nodes introduced by dtype guards or exported graph conversions.
4. AudioVAE encode/decode convolution blocks when clone/reference paths are active.
5. Host-visible session split overhead, if total wall time is high but individual node time is not.

## Acceptance Criteria

- `hotspots.md` has a top-20 node table.
- `hotspots.md` has op-type totals.
- Cast hotspots are listed separately.
- Cache-related hotspots are listed separately.
- The generated shortlist has 3-5 concrete diagnostic reasons when node events are present.
- No optimization or refactor is made until profile output is reviewed.
