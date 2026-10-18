# QwenLean: current state and completion plan

This is the active execution map as of 2026-10-18. Historical reports retain their
original conclusions and commands; AGENTS.md governs current authorization.
The latest measured audit is [project-readiness-20261018.json](../reports/project-readiness-20261018.json).

## Outcome we are trying to establish

Determine whether the pinned, original bf16 Qwen3.5-0.8B can answer correctly with
less redundant reasoning and better termination. Deliver a reproducible stock
versus removable-adapter comparison, with complete provenance, raw outputs,
correctness, termination, token use, latency and supported-context retention.
Correctness takes priority over brevity. A shorter wrong answer fails the goal.

Research completion and model improvement are distinct. A reproducible negative
result can close a bounded experiment; it cannot be advertised as an improved
LeanQwen model. There is currently no approved adapter or demonstrated improvement.
An entire optimization program cannot be promised a successful scientific result.

## Evidence and implementation map

| Stage | What exists | What remains |
|---|---|---|
| Model and environment | Pinned original weights, direct Python/MLX inference, explicit EOS/template/samplers, single-workload lease | Fresh healthy supervised execution on the shared Mac |
| Evaluation | Durable task/seed/config/scorer records, explicit stream failures, exact resume, frozen v5 and policy v6, paired task statistics | Complete the current matched cohort |
| Decoding diagnosis | 144 prompt-policy trajectories; seven-case staged tail; 34 early-exit probes; semantic calibration and guard replay | No default stopping intervention is validated |
| Context | Native 262144 positions/RoPE retained; reversible profiles; stock behavioral probes through about 32K | Adapter retention of those measured supported cases; no claim of 128K/262K capacity |
| Data | 768 original human GSM8K TRAIN solutions, eight-example smoke subset, 64 reserved source DEV questions | No new source is needed for the technical smoke; broader sources remain separately gated |
| Data quality | Source/license/revision, published human QA, exact annotations/final linkage, preserved prose/template/EOS, contamination screening | Prose semantics are not formally verified; no fresh individual review is claimed |
| Training | Actual masked full-vocabulary LoRA optimizer, chunked head loss, save/reload and base integrity interfaces | One successful supervised four-step smoke, persistence and fresh inference |
| Safety | CPU admission before weights, independent owned watchdog, durable stop journals, deadlines, RAM/swap/RSS checks | Successful full Metal supervision remains unmeasured |
| Improvement experiment | Conditional rank-4 final-two-layer proposal | Smoke, full baseline, measured capacity, justified deadline and separate authorization |
| Release | Repository, pinned configurations, reports and reproducible interfaces | Accepted unfused adapter, matched results, limitations and restoration instructions if improvement passes |

The main path through the code is:

1. `datasets/` produces prompts, distinct task identities and provenance.
2. `training/external_contract.py` and `quality.py` verify permission before
   tokenized training records, then reproduce source text, reviews and token IDs.
3. `inference/mlx_backend.py` loads the pinned model, applies the runtime context
   contract and generates explicit token/termination records.
4. `evaluation/runner.py` journals outputs; `scoring/` separates correctness and
   formatting; `metrics/` measures repetition and weak redundancy proxies.
5. `scripts/phase4_dev.py` owns the frozen stock/adapter cohort and its bounded
   monitored invocations. `single_job.py` reuses valid durable generations and
   preserves failed attempts. Aggregate results wait for the complete cohort.
6. `training/mlx_sft.py` performs gated LoRA updates, frozen-base fingerprinting,
   save/destroy/reload, exact logit checks and fresh inference. The watchdog spans
   model loading through final technical acceptance.
7. `checkpoint_audit.py` rejects quarantine and partial attempts before ordinary
   adapter loading. `evaluation/summary.py` and `prompt_comparison.py` implement
   the statistics interfaces for matched comparisons.

The [operating guide](guide.md), [architecture](architecture.md),
[metrics contracts](metrics.md), [data strategy](training-data-strategy.md) and
[context contracts](context-management.md) provide the details.

## What the measurements say

The initial 60-task DEV study observed 0 correct official thinking completions,
4 with repetition penalty 1.05 and 24 in the official non-thinking control, under
a 2048-token cap. These historical prompts/scorers differ from the current cohort.

The prompt study selected P2 provisionally: 7/36 correct versus 3/36 for legacy
`FINAL:` wording, with 29/36 P2 trajectories still capped. Larger-budget diagnosis
showed both eventual correct completion and repeated re-derivation, instruction
doubt and arithmetic drift. The literal guard replay triggered on none of 161
saved trajectories. Early closure recovered only 3/34 probes. None establishes a
capability-preserving stopping policy. See [diagnosis](../reports/phase3-reasoning-diagnosis.md).

The natural TRAIN pilot yielded four correct EOS responses from a single distinct
task and 60 capped/unclosed responses. It remains a preserved negative result,
excluded from current training. Solver-authored corpora are also historical.

The current external human pool has 768 distinct approved source problems with
complete sequences of 97–499 tokens. Approval is scoped to published human QA plus
exact screening, at most 1,000 original references. OpenR1 and other synthetic
sources remain disabled pending source-specific permission/contamination/quality
work. No coding-agent prose or solutions become targets.

Historical failed SFT initializations recorded zero, one and three updates. The
unsafe frozen-prefix path's finite step-two checkpoint is quarantined; its fourth
step was interrupted. These separate failures do not add up to a passing smoke.
Frozen-prefix materialization stays disabled.

The frozen DEV protocol contains 24 human-source questions and 32 procedural
retention questions in both modes: 112 jobs. It has one matched seed per task/mode,
so future uncertainty estimates must retain task clustering and disclose that
sampler variance remains unestimated. Completion is still 0/112.

## Ordered remaining milestones

| Milestone | Completion evidence | Execution boundary |
|---|---|---|
| 1. Resource qualification | Fresh admission: at least 8 GiB available RAM and at most 2 GiB occupied swap | One check after a resource change; refusal stops model work |
| 2. Technical smoke | Four finite nonzero updates; unchanged frozen base; saved/reloaded adapter fingerprints and exact logits; fresh inference; clean watchdog exit | At most one qualified attempt; failure stops model work with no automatic GPU retry |
| 3. Stock reference | All 112 frozen jobs complete; raw records and per-mode correctness/EOS/caps/tokens/latency | Serial monitored batches of one or two jobs, unchanged tasks/seeds/samplers/budgets |
| 4. Capacity and experiment contract | Complete 499-token example feasibility; measured total cost; defensible longer deadline; frozen periodic DEV IDs and acceptance rules | Remains gated; the four-step smoke does not authorize new capacity workloads or longer SFT |
| 5. Bounded improvement SFT | Approved conditional configuration; complete update/resource/checkpoint journals and persistence acceptance | Proposal only; the existing 128-update design is not authorized |
| 6. Matched evaluation | Same 56 tasks/both modes, paired correctness and efficiency, hard-task/non-thinking retention and supported context checks | Separately qualified monitored execution; no partial-cohort success claim |
| 7. Decision and handoff | Reproducible accepted adapter or explicit negative result, raw evidence inventory, limitations and restore-stock instructions | Publish a capability claim only if its predeclared acceptance criteria pass |

Milestones 2 and 3 are already within the bounded Phase 4D authorization when
resource admission is healthy. A resource failure ends model work for the session.
No automatic loop may retry admission or a failed GPU workload. The total smoke
deadline is 120 seconds including loading, integrity, reload and inference; active
gradients have a 15-second deadline. Running limits retain 4 GiB RAM reserve,
128 MiB occupied-swap growth, 512 MiB swap-out growth and 3 GiB process RSS.
MLX limits are guidelines; sampled supervision cannot guarantee against instant spikes.

The current proposed improvement design is 128 updates, batch **one** and
accumulation **one**, at most 512 complete tokens, with checkpoints at 32/64/96/128.
Older accumulation-four text is superseded by the Phase 4D proposal. The deadline
remains unset. Neither that design nor a shorter speculative schedule can be run
to bypass the outstanding gates.

Before improvement training, freeze how to assess no accuracy decrease in either
mode, harder-task retention, paired efficiency on correct answers and supported
context retention. Publish uncertainty and censored lengths. A training-loss drop
does not establish reasoning improvement. TEST remains reserved; any final TEST
campaign needs a separately frozen protocol after DEV decisions, without TEST tuning.

Trajectory compression, preference optimization, adaptive budgets and a CUDA
backend are future alternatives. They are not prerequisites for completing this
bounded human-reference SFT experiment. A refused shared-host admission does not
by itself establish that the ordinary MLX path is infeasible or justify migration.

## Resume from evidence

The readiness inventory is CPU-only and can be rerun to a **new** report path:

```bash
uv run --frozen --no-sync python scripts/audit_project_readiness.py \
  --admission runs/phase4d-resource-admission-03/resource-admission.json \
  --output runs/project-readiness-next.json
```

It checks both approved pools with the pinned local tokenizer, compares local DEV
to the tracked protocol, inventories saved jobs and rejects unapproved adapters.
It starts no admission or model job. Missing local archives are errors, never
positive readiness. Its saved-admission measurements are historical observations,
not a reusable model-loading permit. Exit 0 means the inventory succeeded;
individual gates can still be incomplete. Exit 1 means invalid/missing evidence.

On 2026-10-18 one fresh CPU admission refused: 5.790 GiB available RAM versus 8 GiB
required; 0.943 GiB occupied swap versus the 2 GiB ceiling. Only RAM failed. No
model loaded or GPU retry followed. After a user-controlled resource change, the
next single admission destination is `runs/phase4d-resource-admission-04`.

```bash
uv run --frozen --no-sync python scripts/check_training_resources.py \
  --config configs/training/external-human-smoke-supervised.yaml \
  --output runs/phase4d-resource-admission-04
```

Only after it passes, the authorized bounded technical-smoke entrypoint is:

```bash
uv run --frozen --no-sync qwenlean train-sft \
  --config configs/training/external-human-smoke-supervised.yaml \
  --dataset data/processed/external-human-smoke8-v1/train.jsonl \
  --quality-manifest data/processed/external-human-smoke8-v1/manifest.json \
  --output adapters/phase4d-supervised-smoke-01
```

The trainer performs fresh admission itself and must retain the independent live
watchdog. Existing output destinations are protected. Do not run historical bare
inference/diagnostic commands during this safety-restricted stage. Do not stop
unrelated applications, clear another project's caches or relax resource limits.
