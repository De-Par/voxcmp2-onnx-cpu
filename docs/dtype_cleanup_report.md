# Dtype Cleanup Report

This report fixes the production dtype policy for FP32 and BF16 ONNX exports after hotspot profiling identified `Cast`/`CastLike` nodes as a diagnostic class to track.

The cleanup is intentionally narrow:

- no model math changes
- no runtime semantic changes
- no split FP32/BF16 architecture
- no custom ops
- same public input/output contracts for both precision profiles

## Profiling Input

The profiling workflow in `docs/profile_hotspots.md` separates Cast hotspots from general node latency. The relevant slowdown classes for dtype cleanup are:

- generated `Cast`/`CastLike` nodes from wrapper dtype guards
- BF16/FP32 ping-pong when a BF16 region is immediately cast back to FLOAT
- storage-only BF16 initializer conversion followed by `Cast(BFLOAT16 -> FLOAT)`
- required graph-boundary casts from the shared FP32 host-visible runtime contract

The local workspace used for this pass does not contain regenerated ONNX artifacts, so graph-level before/after counts must be produced after re-export with `tools/profile/summarize_dtype_casts.py`.

## Classification

| class | policy |
|---|---|
| redundant cast chains | Remove when they come from no-op wrapper casts or direct same-dtype `Cast -> Cast` chains. |
| FP32/BF16 ping-pong | Treat as a production BF16 blocker unless it is a documented precision island. |
| unavoidable precision boundaries | Allow graph input/output casts required by the shared FP32 host contract. Keep them explicit and measurable. |
| exporter artifacts | Track `CastLike` and generated Cast nodes separately; optimize only when they map to wrapper code, not model math. |

## Code Changes

`src/export/common.py` now exposes `cast_tensor_if_needed(tensor, dtype)`.

All export wrappers use that helper instead of unconditional `.to(dtype=...)` calls:

- `AudioVAEEncoderWrapper`
- `AudioVAEDecoderWrapper`
- `VoxCPM2PrefillWrapper`
- `VoxCPM2DecodeStepWrapper`

This removes no-op FP32 casts at graph boundaries and keeps BF16 boundary casts only where the precision profile requires them.

`VoxCPM2DecodeStepWrapper` also uses conditional casts around the rotary FP32 island:

- FP32 export: rotary multiply/add stays FP32 without extra Cast nodes
- BF16 export: q/k/cos/sin enter the documented FP32 island and return to BF16 compute

## Source-Level Cast Site Summary

This is a source-level cleanup summary. It counts wrapper cast sites that were unconditional before this pass and are now conditional.

| module | previous unconditional cast sites | after cleanup | expected effect |
|---|---:|---:|---|
| AudioVAE encoder | 2 | 2 conditional | FP32 no-op input/output casts disappear; BF16 keeps boundary casts. |
| AudioVAE decoder | 2 | 2 conditional | FP32 no-op input/output casts disappear; BF16 keeps boundary casts. |
| Prefill | 11 | 11 conditional | FP32 text/mask/audio/output no-op casts disappear; BF16 keeps mask/audio/output boundary casts. |
| Decode step | 27 | 27 conditional | FP32 boundary, length, `t_span`, and rotary no-op casts disappear; BF16 keeps boundary casts plus documented rotary island. |

Expected FP32 no-op Cast reduction after re-export: all graph-edge casts introduced only by wrapper `.to(float32)` guards should disappear.

Expected BF16 improvement after re-export: BF16 compute regions remain intact, direct BF16->FP32->BF16 ping-pong chains should be absent, and storage-only initializer cast-back remains forbidden.

## Graph-Level Summary Command

After exporting both production profiles, generate the actual before/after Cast count report:

```bash
python -B tools/profile/summarize_dtype_casts.py \
  --before-root artifacts/pre_dtype_cleanup/onnx \
  --after-root models/onnx \
  --profile-json artifacts/profile/parsed_hotspots.json \
  --json-report artifacts/reports/dtype_cleanup_casts.json \
  --markdown-report artifacts/reports/dtype_cleanup_casts.md
```

If no pre-cleanup models were archived, omit `--before-root` and use the report as the post-cleanup baseline.

## Acceptance Criteria

- FP32 re-export has fewer Cast nodes than the previous unconditional-wrapper export.
- BF16 re-export has no storage-only `BFLOAT16 initializer -> Cast FLOAT` pattern.
- BF16 re-export has no direct FLOAT/BFLOAT16 ping-pong Cast chains unless a documented FP32 island requires it.
- Public graph input/output names, ranks, cache semantics, and host-visible dtype contract remain unchanged.
- Parity tests still pass for FP32, and BF16 parity remains within the project tolerance after artifacts are regenerated.
