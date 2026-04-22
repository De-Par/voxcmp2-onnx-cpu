# ONNX Runtime Performance Tuning

This page documents CPU-only performance knobs for the current VoxCPM2 ONNX runtime and how to interpret benchmark differences against the official PyTorch API.

## Current Interpretation

The benchmark below shows that ONNX FP32 and ONNX BF16 produce nearly identical observable outputs:

- same audio duration: `2.560s`
- same decode count: `16` steps
- same stop reason: `stop_logits`
- very close peak/RMS values

That means the large mismatch is not a BF16-specific failure. The mismatch is between the official PyTorch inference path and the current ONNX host-loop path.

Known causes:

- Official VoxCPM2 samples diffusion noise with `torch.randn()` inside `UnifiedCFM.forward()`. The ONNX boundary takes `diffusion_noise` from host NumPy so the graph can stay deterministic and exportable.
- The benchmark now seeds both paths for repeatability, but the same integer seed does not make PyTorch and NumPy produce identical diffusion-noise tensors.
- The official API runs its model path in the model-configured dtype, commonly BF16 on CPU. The export wrappers currently force FP32 for correctness-first ONNX export.
- The official loop stops when `i > min_len and stop_flag == 1`. The ONNX host loop stops after the completed-step count reaches `min_steps` and `stop_logits` selects stop.
- The official API has `retry_badcase=True` by default at the public API layer. The benchmark keeps retries off unless `--orig-retry-badcase` is passed so a retry does not hide the first-pass behavior.
- The ONNX decode-step graph uses explicit tensor-in/tensor-out K/V caches. This is exportable and inspectable, but it is heavier than the official Python-side mutable cache.

The current BF16 artifacts are also not a full BF16 compute runtime. `src/experiments/bf16_feasibility.py` converts selected FLOAT initializers to BFLOAT16 storage and inserts Cast nodes back to FLOAT. This tests model size and ORT CPU loader coverage while keeping most compute in FP32.

## Runtime Knobs

All knobs keep `CPUExecutionProvider` only. No CUDA, CoreML, DirectML, or GPU fallback is allowed.

- `graph_optimization_level`: `disable`, `basic`, `extended`, `all`
- `execution_mode`: `sequential`, `parallel`
- `log_severity_level`: `verbose`, `info`, `warning`, `error`, `fatal`
- `intra_op_num_threads`: integer thread count; omit for ORT default
- `inter_op_num_threads`: integer thread count; omit for ORT default

The default runtime remains conservative:

```text
graph_optimization_level=disable
execution_mode=sequential
log_severity_level=warning
intra_op_num_threads=default
inter_op_num_threads=default
```

This default is intentional for parity/debug work because exported module outputs are easier to compare when ORT graph rewrites are disabled.

The benchmark CLI defaults ONNX log severity to `error` so performance summaries stay readable. With `--onnx-graph-optimization all`, ORT may print warnings such as `Could not find a CPU kernel and hence can't constant fold CastLike`. These warnings mean the optimizer could not precompute that node and left it in the graph; they are not a synthesis failure.

The runtime creates ONNX sessions lazily. The benchmark preloads the sessions needed by the selected mode during its load phase by default, so ONNX `load` includes ORT session creation and graph optimization while ONNX `synth` is closer to model execution time. Pass `--no-onnx-preload-sessions` only when measuring first-request latency.

## Benchmark Commands

Baseline:

```bash
python -B src/bench/compare_pipelines.py \
  --text "Hello from VoxCPM2." \
  --mode voice_design \
  --voice-design "pretty girl with sugar voice, slow" \
  --output-dir artifacts/bench \
  --report-json artifacts/bench/report.json \
  --variants orig onnx_fp32 onnx_bf16
```

Performance experiment with ORT graph rewrites and fixed thread counts:

```bash
python -B src/bench/compare_pipelines.py \
  --text "Hello from VoxCPM2." \
  --mode voice_design \
  --voice-design "pretty girl with sugar voice, slow" \
  --output-dir artifacts/bench_ort_tuned \
  --report-json artifacts/bench_ort_tuned/report.json \
  --variants onnx_fp32 onnx_bf16 \
  --onnx-graph-optimization all \
  --onnx-execution-mode sequential \
  --onnx-log-severity error \
  --onnx-intra-op-threads 8 \
  --onnx-inter-op-threads 1
```

Try `--onnx-intra-op-threads` values around the number of physical performance cores first. `parallel` execution mode may help some graphs, but for a one-step decode loop it can also add overhead, so it should be measured per platform.

## Acceptance Criteria

- ONNX sessions report only `CPUExecutionProvider`.
- Tuning flags do not change default FP32 behavior unless explicitly passed.
- Benchmark output prints the active ORT settings.
- JSON reports remain saved to disk; large raw reports are not printed to the terminal.

## Non-Goals

- No BF16 replacement for the FP32 runtime.
- No quantization.
- No accelerator provider fallback.
- No merging prefill, decode loop, and AudioVAE into one ONNX graph.
