# Fixed-Capacity Decode Cache Report

## Scope

This report records the production state-contract change for `VoxCPM2DecodeStep`.

The change applies to both FP32 and BF16 exports because the contract is defined in `src/export/common.py` and consumed by the shared runtime loop.

## Contract Change

Before:

```text
base_k_cache:      [base_layers, batch, kv_heads, cache_seq, head_dim]
next_base_k_cache: [base_layers, batch, kv_heads, cache_seq + 1, head_dim]
```

Every decode step returned a larger K/V cache tensor.

After:

```text
base_k_cache:  [base_layers, batch, kv_heads, max_cache_seq, head_dim]
base_k_update: [base_layers, batch, kv_heads, 1, head_dim]
```

The cache tensor capacity is fixed for the whole host loop. The graph masks attention to valid positions plus the current update, returns one-position K/V updates, and emits explicit `next_*_current_length` tensors. Host code writes updates into preallocated cache arrays.

## Traffic

Let:

- `K = (base_layers + residual_layers) * batch * kv_heads * head_dim`
- `S = current_length`
- `C = max_cache_seq`

Old cache output payload per step:

```text
2 * K * (S + 1)
```

New cache output payload per step:

```text
2 * K
```

So cache output payload is reduced by `S + 1`.

Input payload changes from `2 * K * S` to `2 * K * C`. This is intentional fixed-capacity state. It removes per-step shape growth and output-cache reallocation, but total host/ORT transfer depends on ORT CPU input handling and future I/O binding work. Attention compute is masked to the valid prefix plus current position, but the graph still carries fixed-capacity input tensors.

## Cache Shapes

Before:

```text
input cache  seq = S
output cache seq = S + 1
next input   seq = S + 1
```

After:

```text
input cache capacity = C
output update seq    = 1
next input capacity  = C
current_length       = S + 1
```

No output cache tensor grows across the decode loop.

## Latency

Baseline before this contract change is recorded in `artifacts/perf_baseline_ort_tuned/baseline.md`:

| case | ONNX decode-step p50 | ONNX decode-total p50 |
|---|---:|---:|
| text_only_short | 0.477s | 25.820s |
| text_only_medium | 0.438s | 85.631s |
| voice_design_short | 0.562s | 43.294s |
| controllable_clone_short | 0.538s | 34.673s |

After-latency numbers require regenerating FP32/BF16 decode-step artifacts from this contract and rerunning the production baseline. Do not invent after numbers from old artifacts.

Required command:

```bash
python -B src/export/export_all.py --precision fp32
python -B src/export/export_all.py --precision bf16
python -B tools/bench/run_benchmarks.py \
  --output-dir artifacts/perf_baseline_fixed_cache \
  --json-report artifacts/perf_baseline_fixed_cache/baseline.json \
  --markdown-report artifacts/perf_baseline_fixed_cache/baseline.md \
  --variants official onnx
```

## Parity

Before parity command:

```bash
python -B tests/parity/test_decode_step.py \
  --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx \
  --cache-seq 16
```

After parity command:

```bash
python -B tests/parity/test_decode_step.py \
  --onnx-path models/onnx/fp32/decode_step/voxcpm2_decode_step.onnx \
  --precision fp32 \
  --cache-seq 16 \
  --max-cache-seq 64

python -B tests/parity/test_decode_step.py \
  --onnx-path models/onnx/bf16/decode_step/voxcpm2_decode_step.onnx \
  --precision bf16 \
  --cache-seq 16 \
  --max-cache-seq 64
```

The repository-level contract tests pass for the new state names and shared FP32/BF16 public contract. Full PyTorch-vs-ORT parity requires re-exported ONNX artifacts.

## Acceptance Gate

The fixed-capacity contract is production-ready only when:

- FP32 and BF16 decode-step exports both use the new input/output names
- `tests/parity/test_decode_step.py` passes for regenerated artifacts
- benchmark after numbers are recorded in `artifacts/perf_baseline_fixed_cache`
- ORT profiling confirms cache output growth and cache-output `Concat` hotspots are gone
