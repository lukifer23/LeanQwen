# Phase 3 checkpoint before bounded reasoning runs

The independent pre-GPU addendum is incorporated. This is a technical and
stock-model checkpoint, not evidence of an improved reasoning model.

## Repository and history

At addendum arrival, local and remote main matched
`e5b5b49cdd89f7f982f66db90c88fee8c9b907b6`. The working tree held completed context
measurements and their pending docs. Historical-log repair was committed separately
as `466041fc9b946af8d89b14312df4637832a2bd3e`; GitHub run 38286571600 passed.
Original Phase 1 entries from `8f67d3b2`, including all experiment IDs, failure
narratives and scoring corrections, and all later Phase 2/3 entries are preserved
verbatim. Formal definitions stay in `docs/metrics.md`. The restoration source/hash,
chronology/entry inventory and link checks are in `phase3_research_log_restoration.json`.

Latest full native check: `uv run --frozen pytest -q` gives **98 passed**;
Ruff and whitespace checks pass. This checkpoint's code/docs/evidence must be
committed/pushed and its CI observed before the next long run. The agent reports
the resulting actual HEAD, clean status and remote match immediately before launch.

## GPU ownership and resource headroom

PonderScope PID 23131 exited naturally and its manifest says `EVIDENCE_COMPLETE`.
It was never signaled or modified. See `phase3_gpu_handoff.json`. LeanQwen's own
tail was interrupted only after verifying PID/create-time/cwd/command ownership,
to hold escalation for the newly requested gates. Three complete algebra jobs
(2K/4K/8K) remain durable; the next job has zero durable rows and an explicit
KeyboardInterrupt completion. No partial tokens are claimed for that interrupted
job. See `phase3_tail_preflight_hold.json`.

The attributed backend preflight observed **7.10 GiB available RAM before loading**.
Short-generation MLX peaks were **1.605–1.609 GB**. After request cleanup, active
allocation returned to approximately **1.505 GB**, within 22 bytes of the original
loaded-model observation. This is sampled process-local evidence, not total
machine memory. Recheck available RAM and competing workers before resuming.

## Actual preflight results

| Gate | Measured result | Evidence |
|---|---|---|
| Small Metal bf16 hash | Stable repeats; changed values change hash; GPU/CPU transfer succeeds; host chunks at most 6 bytes, covering all 128 input bytes | `phase3_metal_integrity_preflight.json` |
| Short stock generation | Correct `5` with EOS; one-token request stops exactly at cap | `phase3_backend_preflight_attributed.json` |
| Real cache layout | Six ordinary KV caches, eighteen ArraysCache linear states; nonnegative consistent byte counts/offsets; progress and timing populated | Same backend artifact |
| Request cleanup | Full request state released; no meaningful active-allocation growth across the short checks | Same backend artifact |
| All profiles | mac-128k/mac-100k/native-262k produce identical short IDs; every loaded tensor, model args and tokenizer/template metadata unchanged | Same backend artifact |
| Historical P2 | First 128 emitted IDs match exactly for one arithmetic and one algebra replicate-zero source, with original prompt IDs/seeds/sampling | Same backend artifact |
| Full original inventory | 320 loaded tensors, 752393024 elements; unchanged bitwise across profile checks | Backend artifact and `phase3_live_profile_restoration.json` |

The initial backend preflight is retained separately as `phase3_backend_preflight.json`.
After adding explicit seed/config/hash attribution to every backend generation,
the short compatibility checks were rerun into the attributed artifact. No
original evidence was overwritten. LoRA attachment now independently compares
canonical frozen names/shapes/dtypes/raw bytes against the complete original
loaded inventory; optimizer execution remains pending.

## Narrow stock context evidence

The full context suite happened before this addendum arrived. Preserve
`context_document_v1` exactly; no evaluated stimulus was changed afterward.
Future session commands support one representative question before expansion.

| Combined envelope | Actual input | Correct / 5 | Positional retrieval / 3 | 2048-chunk MLX peak | 512-chunk MLX peak |
|---:|---:|---:|---:|---:|---:|
| 2048 | 1792 | 3 | 3 | 2.742 GB | Not tested |
| 8192 | 7936 | 4 | 3 | 3.016 GB | 2.473 GB |
| 32768 | 32512 | 4 | 3 | 5.016 GB | 2.960 GB |

Every row reserves 256 output tokens inside its envelope. At 32K, five emitted
sequences match across chunk variants; at 8K, four match and the changed inventory
trace remains incorrect. Mean 32K prefill is 16.274/16.381 seconds. Cache capacity
is 24/96/384 MiB; linear state about 18.6 MiB. Use 512 for paired long-context
checks, keeping historical reasoning sampler/prefill contracts unchanged.
These fixed-fact/repeated-filler questions do not establish general context
capability. The inventory-reasoning question fails at every tested length.
Full details, raw output IDs/text and archive hashes: `phase3-context-stock.md`,
`phase3_context_stock.json`, `phase3_context_stock_records.jsonl`.

Native maximum remains 262144. Default runtime ceiling is 131072, fallback 102400,
with 262144 selectable without retraining or edits to original position/RoPE
settings. Full 100K/128K/262K inference and adapter retention are unverified.
Restoration commands and future varied-suite design: `docs/context-management.md`.

## Next authorized runs

Run sequentially; do not overlap another project's GPU or any LeanQwen model lease.

```bash
uv run --frozen --no-sync python scripts/termination_tail.py
uv run --frozen --no-sync python scripts/early_exit_probe.py
# Only after those model processes exit; install/record the semantic extra between workloads:
uv sync --frozen --extra semantic
uv run --frozen --no-sync python scripts/calibrate_metrics.py
uv run --frozen --no-sync python scripts/train_candidates.py
```

The tail command resumes completed jobs without duplicate durable generations.
Cap escalation remains seven predeclared cases; prefix divergence halts it.
Treat capped traces as right-censored. Early exit retains mandatory zero-prefix
controls and remains a forced-close diagnostic. Literal guard replay is already
a completed negative finding on the 144 prompt traces: zero triggers/savings
at three predeclared settings; keep it disabled. The full calibration adds the
new tail data, not a new threshold sweep.

Only after measured 16×4 TRAIN coverage and individual quality review: construct
the complete-token pool and run `configs/training/smoke.yaml`. Do not implement
the conditional solver path before poor natural coverage is observed. Any later
solver source requires its own typed verification/provenance/contamination gate.
Real SFT is conditional on the smoke and a sufficiently diverse reviewed corpus.
No new TRAIN generations, solver corpus, optimizer steps or adapter exist at this
checkpoint. Correctness-preserving reasoning efficiency remains the objective.
