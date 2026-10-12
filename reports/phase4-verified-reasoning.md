# Phase 4 measured checkpoint

The CPU data milestone is complete. Model work stopped at the stock baseline's
configured available-memory floor. No optimizer step or improvement SFT ran. The final v1.1 corpus supersedes
the initial v1 corpus for future training; the first manifest remains historical.
This is an incomplete intervention experiment, not a negative result for SFT.

Starting clean `main`: `160f7e0065249af41520c485a6e69ffc063070b8`.
Data/approval milestone: `135f5d858a39e4c86812eab85fcfc0b5ef9e9c20`, pushed.
Local validation: 109 tests passed, Ruff passed. Milestone GitHub CI
[38306454283](https://github.com/lukifer23/LeanQwen/actions/runs/38306454283) passed.
The historical-file SHA256 audit detected zero changes to previous artifacts.

## Independently verified data

`solver-reasoning-v1.1`: 512 distinct TRAIN problems, 512 normalized task and response structures,
64 each in arithmetic, modular state, Boolean logic, sequence transformations,
ordering, algebra, exact-cent word problems and bounded counting. Difficulty:
45 trivial, 120 easy, 206 medium, 141 harder. Complexity is structural, not measured
model success. All steps and final answers passed independent algorithmic checks.
The initial 256/512-problem v1 corpora are retained separately. A stricter final
audit found 12 duplicate number-free word structures and 14 duplicate ordering
response schedules in v1. Version v1.1 excludes both, regenerates all eight
families, and freezes a fresh cohort before training. See the
[explicit correction](phase4-structural-audit-correction.json). All old manifests
and the zero-token baseline failure remain intact; no adapter used either corpus.

| Length | Reasoning only, cumulative under | Complete supervised response, cumulative under | Full prompt + response, fit at or below |
|---|---:|---:|---:|
| 512 | 413 | 411 | 368 |
| 1024 | 496 | 491 | 480 |
| 2048 | 511 | 511 | 511 |
| 4096 | 512 | 512 | 512 |

Reasoning: min 9, median 210, mean 315.0977, max 2134 tokens. Full sequences:
min 54, median 305, mean 416.0586, max 2301. Complete distributions and every
individual length are in [the corpus manifest](phase4_solver_corpus_v1_1.json).
Length-bound fit is not a training-memory measurement.

No accepted-corpus verification failure, normalized TRAIN structure duplicate,
semantic/prompt duplicate, structural OOD overlap or historical exact-prompt
contamination was detected. ID intentionally shares TRAIN topology with fresh
constants/entities; OOD has reserved structural hash buckets. TEST remains a sealed
generator namespace, with no TEST target generation or model evaluation. Reusing
an algorithm with new structure does not prove conceptual independence.

Records preserve real computed states, independent certificates, complete rendered
and tokenized official thinking suffixes, final-channel/EOS validity, pinned model
and tokenizer/template metadata, Apache-2.0 permission, source hashes and reproducible
seed/recipe. Source approval independently rebuilds the full raw archive. No external
corpus or closed-model response is used. Natural-Qwen parent validation is retained.
The approved smoke pool has eight complete examples, one per family, with total
sequence lengths 263/259/263/250/236/254/262/273. Sampling is explicitly 100% solver.

Initial development mistakes were fixed before approval: insufficient divisibility
for exact cents, an invalid tokenizer metadata assignment, and a preflight launch
rejected by the active CPU-generation lease. Details and failed destinations are
preserved in [development failures](phase4-development-failures.json). The archive
round-trip test initially compared Python tuples with JSON lists; its fixture was
corrected to exercise serialized training records. Final tests pass.

## Model work and stopping evidence

Small Metal integrity and short Qwen backend/profile/prefix preflights passed.
The corrected matched DEV protocol froze 32 tasks across eight families, ID/OOD, simple
and harder dependencies, one matched seed per task, P2, official mode-specific
samplers, 2048 total-output tokens and guard off. One replicate limits estimates
of sampler variance. No task was selected using model outcomes.

The first stock non-thinking request on the initial v1 cohort stopped at prefill before emitting a token:
MLX active/peak 1,504,791,388 bytes, observed RSS 943,620,096 bytes, available system
RAM 3,323,740,160 bytes (3.096 GiB), below the configured 4 GiB floor. The raw
failure is retained, not scored as task-solving evidence. See
[memory stop and raw pointer](phase4-baseline-initial-memory-stop.json).
Qwen was released. No unrelated application was terminated; an initially observed
external model unloaded before Qwen preflight. No competing model is left active
by this phase. Existing swap use is system state, not measured training swap growth.

Consequently: stock cohort accuracy/EOS/caps are unmeasured; optimizer steps 0;
loss, gradients, throughput and capacity unmeasured; no saved adapter; frozen-base
optimizer integrity and save/reload equivalence unexecuted; no before/after accuracy,
token savings, semantic redundancy, useful longer-reasoning, non-thinking utility
or supported long-context retention comparison. Original base/context/tokenizer
configuration is unchanged. The two-layer rank-4/scale-2 smoke config is prepared,
not evidence of adaptation capacity or an improvement claim.

## Next gated action

Free approximately 2 GiB of background memory. Retry the unchanged frozen stock
protocol in a new destination, then the four-step smoke using the eight newly approved
complete targets. Only after finite optimizer/update, full frozen-base integrity,
exact fixed-prefix reload logits and fresh inference pass should complete examples
be probed at 1024/2048/4096 bounds. Their actual lengths must be recorded (the
current corpus max is 2301, so it cannot by itself establish a full 4096-token
training step). Select one finite SFT schedule from those measurements; do not
queue it automatically or relax a failed floor to manufacture a passing gate.

```bash
uv run --frozen python scripts/phase4_dev.py --cohort data/generated/phase4-v1.1-frozen-dev --output runs/phase4-stock-dev-memory-retry
uv run --frozen qwenlean train-sft --config configs/training/phase4-smoke.yaml --dataset data/processed/phase4-v1.1-smoke8/train.jsonl --quality-manifest data/processed/phase4-v1.1-smoke8/manifest.json --output adapters/phase4-four-step-smoke
```

The research conclusion remains **inconclusive**: this phase produced a diverse,
verified intervention corpus, but resource admission stopped the first matched
baseline before training. No evidence yet distinguishes reasoning-efficiency
learning from answer-format learning. The next highest-value experiment is the
already frozen baseline followed by actual optimizer validation, once headroom is
available. The failed natural pilot is preserved as historical evidence.
