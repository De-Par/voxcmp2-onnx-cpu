# Production Performance Baseline

This page defines the current reproducible performance baseline for the official VoxCPM2 API and the CPU-only ONNX runtime in this repository.

The baseline is measurement-only. It must not change:

- model math
- export wrappers
- ONNX graph files
- runtime stop policy
- runtime cache/state semantics

## Command

Run the full fixed matrix:

```bash
python -B tools/bench/run_benchmarks.py \
  --output-dir artifacts/perf_baseline \
  --json-report artifacts/perf_baseline/baseline.json \
  --markdown-report artifacts/perf_baseline/baseline.md \
  --variants official onnx \
  --repeats 3
```

The command writes:

- JSON report: `artifacts/perf_baseline/baseline.json`
- Markdown report: `artifacts/perf_baseline/baseline.md`
- WAV outputs: `artifacts/perf_baseline/wavs/*.wav`
- deterministic reference WAV for the clone case: `artifacts/perf_baseline/reference_16k.wav`

## Fixed Cases

The full matrix always uses these cases unless `--cases` is passed for a smoke run:

| case | mode | reference path |
|---|---|---|
| `text_only_short` | `text_only` | no |
| `text_only_medium` | `text_only` | no |
| `voice_design_short` | `voice_design` | no |
| `controllable_clone_short` | `controllable_clone` | yes |

The controllable-clone reference is generated deterministically as a short 16 kHz WAV unless `--reference-wav` is supplied.

## Metrics

Every run records:

- `model_load_seconds`
- `wall_seconds`
- `total_synth_seconds`
- `decode_steps`
- `output_duration_seconds`
- `sample_rate`
- `samples`
- output WAV path
- run seed

ONNX runs additionally record explicit runtime-stage timings:

- `input_build_seconds`: host tokenization, mode assembly, WAV I/O/resampling, and AudioVAEEncoder when a reference path is used
- `prefill_seconds`: `VoxCPM2Prefill` ONNX session run
- `decode_step_seconds`: one wall-time value per `VoxCPM2DecodeStep` ONNX session run
- `decode_step_total_seconds`
- `decode_step_seconds_p50`
- `decode_step_seconds_p90`
- `audio_decode_seconds`: `AudioVAEDecoder` ONNX session run

Official API runs record load, total synthesis, decode steps, and audio stats. Official per-stage prefill/decode timing is intentionally `null` because the public API does not expose the same explicit module boundaries and this baseline must not rewrite the official model internals.

## Aggregates

The JSON and Markdown reports aggregate each `(case, variant)` pair with:

- min / max / mean
- p50
- p90

`wall_seconds` is measured around the whole run call and is aggregated separately from `total_synth_seconds`.

The Markdown report includes an ONNX-vs-official table. This makes the slow point visible:

- if ONNX `prefill_seconds` is small but `decode_step_total_seconds` dominates, the decode loop is the current bottleneck
- if `input_build_seconds` dominates only clone cases, reference audio loading/resampling/AudioVAEEncoder is the bottleneck
- if official total synthesis is lower but ONNX output duration and decode steps differ, performance cannot be interpreted as pure kernel speed parity

## Recommended Smoke

Use this for quick validation without waiting for the full matrix:

```bash
python -B tools/bench/run_benchmarks.py \
  --output-dir artifacts/perf_baseline_smoke \
  --variants onnx \
  --cases text_only_short \
  --repeats 1 \
  --max-steps 1 \
  --min-steps 0
```

This smoke intentionally writes truncated audio. It is only a tool check.

## Acceptance Criteria

- One command produces JSON and Markdown reports.
- The full default case matrix includes official API and current ONNX runtime.
- Reports include model load latency, total synthesis latency, decode steps, output duration, and p50/p90 wall-time aggregates.
- ONNX reports include explicit `prefill` and `decode_step` latency.
- The benchmark does not modify model math, export path, or runtime semantics.
