# Running QwenLean

Current work uses licensed external human reasoning, starting with 768 original
GSM8K TRAIN solutions. Qwen is the student/evaluation target only. No Qwen/LFM
teacher or rewrite/judge pipeline and no LM Studio commands. The completed natural
pilot and later solver corpus are preserved as historical evidence, excluded from
current SFT. See [source comparison](../reports/external-source-comparison.md),
[data strategy](training-data-strategy.md) and [corpus audit](../reports/external-human-pilot.json).
Phase 4C reconciled one ordinary chunked-loss update and three unsafe-prefix
updates. The partial unsafe checkpoint is quarantined and rejected before model
loading. No successful four-step smoke or approved adapter exists. Fresh admission
failed both starting RAM and occupied-swap gates, so no Phase 4C GPU job ran.
See [corrected evidence and operating decision](../reports/phase4c-review.md).

This guide describes implemented commands. It does not assume an adapter,
optimized decoder, successful training dataset, or benchmark improvement exists.
Use the status and reports in the README for measured findings.

## Install and validate

Use Apple Silicon macOS, Python 3.12 and uv. The default project dependencies
include MLX and are not a ready-to-install CUDA environment.

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check qwenlean scripts tests
uv run --frozen python scripts/validate_environment.py
uv run --frozen python scripts/smoke_inference.py
```

The optional `compat` dependencies install PyTorch, PEFT and TRL for compatibility
inspection; they are not an implemented training pipeline. Check probe artifacts
for exact support boundaries. `smoke_inference.py` is a diagnostic with no presence
penalty; it is intentionally not the official baseline sampler.

## Build and evaluate

The tracked small split pools are reproducible. Rebuilding overwrites these files
with the same generator seeds; do not change generator versions in the middle of
an experiment. Keep a run's dataset hash and IDs when comparing variants.

```bash
uv run --frozen qwenlean build-dataset
uv run --frozen qwenlean eval --config configs/baseline.yaml
uv run --frozen qwenlean eval --config configs/nonthinking.yaml
```

Both commands above use DEV by default. No TEST evaluation is needed for this
milestone. The CLI can evaluate a supplied JSONL via `--dataset`; reserving TEST
is an experimental policy, while the training provenance gate rejects non-TRAIN
records. A model process lock rejects concurrent model commands. Run them in
sequence; completed commands release the lock. Do not delete the lock file while
a workload is active.

Each `eval` prints its new directory. A run is complete only when its
`completion.json` says `complete`; an interrupted JSONL is partial evidence and
must not be reported as a complete benchmark. Weights remain in the user's
Hugging Face cache, separate from the repository.

## Inspect and compare

Substitute actual directories for the shell placeholders below:

```bash
uv run --frozen qwenlean analyze runs/<A-run>
uv run --frozen python scripts/sampling_ablation.py --baseline-run runs/<A-run>
uv run --frozen python scripts/create_reports.py --baseline-run runs/<A-run> --nonthinking-run runs/<A0-run> --output-dir reports/<new-derived-report-directory>
```

The sampler script uses twelve DEV examples (two per family) for a three-candidate
single-variable sweep. It can decline to promote any candidate. If it selects
one, it runs the full DEV suite and measures repeatability of pilot token IDs.
`reports/sampling_measurements.json` records the decision. A pilot cannot be
paired against a full 60-example run: compare exactly matching sample sets.
`--pilot-runs <cooler-run> <presence-zero-run> <repetition-run>` can reuse completed
pilots or resume an ordered partial pilot with an identical configuration and
dataset hash. A resumed run appends only pending generations. This recovery option
belongs to the sweep script; general `eval --resume` is now implemented for runs carrying the new scoring contract. Historical completed/aborted runs are protected.

```bash
uv run --frozen qwenlean compare runs/<A-run> runs/<B-run>
uv run --frozen qwenlean generate --config configs/baseline.yaml --prompt 'Compute 17 * 23.' --output runs/one-generation.json
```

`analyze` aggregates saved metrics and does not generate text or change scoring.
Metric reanalysis must write a new derived directory:

```bash
uv run --frozen python scripts/recompute_metrics.py runs/<source> --output runs/<new-derived-analysis> --scoring-version terminal_numeric_policies_v6
```

It preserves original configurations, raw output and historical measurements.
Derived metrics carry their source run, source scoring versions and new version.
Do not replace first-pass v5 reports with a new scoring contract.

Read `docs/metrics.md` before interpreting any efficiency statistic. In particular,
a 2048-total-output-token cap bounds observed percentiles; zero detected exact
cycles does not exclude semantic reconsideration; and candidate correct-number
appearances do not prove the model had reached a reliable solution.

## Not implemented

The Phase 2 candidate and SFT interfaces described below are implemented but have
not yet completed real model validation. Trajectory compression, preference
optimization, adaptive budgets and a CUDA backend remain unimplemented. No
trained checkpoint or training improvement is currently claimed. The optional exact-cycle guard
exists but remains experimental; enabling `loop_guard.enabled` changes termination
and needs real false-positive inspection before use as a default.


## Phase 2 task/policy and multi-seed evaluation

```bash
uv run --frozen qwenlean eval --config configs/baseline.yaml --limit 12 --prompt-policy P0 --replicates 3
uv run --frozen qwenlean eval --config configs/baseline.yaml --limit 12 --prompt-policy P0 --replicates 3 --resume runs/<incomplete-run>
uv run --frozen python scripts/prompt_policy_ablation.py
# Same single study after interruption:
uv run --frozen python scripts/prompt_policy_ablation.py --resume runs/<incomplete-study>
uv run --frozen qwenlean compare-prompts runs/<study> runs/<study> --left-policy P3 --right-policy P0 --output reports/<new-comparison>.json
```

The ordinary CLI limit selects the first tasks; the study script selects the
historical balanced two-per-family twelve-task subset. Policies are interleaved
within task/replicate to reduce block-order timing confounding. Three seeds per
policy are twelve task clusters, not 36 independent problems. P0 format adherence
is null because it imposes no format. New runs save explicit stream errors and
halt on invalid termination; they never relabel an exception EOS or cap completion.

Resume requires all supplied arguments to recreate the exact saved config/dataset.
Completed runs are protected. Optional torch/PEFT/TRL packages are not needed for
ordinary MLX evaluation. Environment validation writes a separate current artifact,
requires no compatibility extra and refuses existing paths. `analyze` prints by
default; use `--output` for a new summary without changing historical artifacts.

Read [Phase 2 methodology](phase2-methodology.md) for stream/resume/scoring contracts
and context-window preservation. The 262144-token architectural window is retained;
the default runtime combined input/output ceiling is 131072. Select `mac-100k` or
`native-262k` explicitly when needed. Output budgets and bounded smoke-training
sequence lengths do not change architecture or RoPE. See [context commands](context-management.md).

## Automated checks

GitHub Actions runs Ruff and device-free tests on Linux without downloading model
weights. MLX-only unit tests and the offline pinned-tokenizer checks are marked
`mlx` and `local_tokenizer`; run the complete suite on the Mac with its cached
checkpoint. The Linux command is:

```bash
uv sync --frozen
uv run --frozen pytest -q -m 'not local_tokenizer and not mlx'
```

The optional `semantic` dependency group prepares local embedding analysis; it
is not needed for ordinary Qwen evaluation. Install optional dependencies only
between model workloads so a running experiment keeps its recorded environment.


## Procedural-v2 and diagnostic controllers

```bash
uv run --frozen qwenlean build-dataset --version v2 --per-bin 2 --output data/splits/<new-v2-directory>
uv run --frozen python scripts/termination_tail.py
uv run --frozen python scripts/early_exit_probe.py
```

The v2 builder refuses nonempty destinations. The tracked first v2 pool contains
64 tasks per split, eight families and four structural difficulty bins. Numeric
ASTs, logic ASTs, state operation patterns, sequence operations, ordering sizes,
algebra forms and word-problem templates define structural signatures. Cross-split
signature and exact semantic-task collisions are rejected; this does not prove
conceptual independence. `trivial` is relative to the family, not model accuracy.
TEST has no generated model responses.

The tail controller requires the completed prompt study. It follows the committed
seven-case protocol, skips larger caps after EOS and verifies exact emitted-ID
prefixes. A prefix divergence stops escalation and writes its evidence. The
counterfactual controller forces closure on six saved DEV reasoning prefixes and
uses a separately labeled greedy 128-token final budget. It does not modify the
source traces or demonstrate that ordinary Qwen would stop at those positions.
Controllers retain individual generations and validate saved contracts on restart.

## External ingestion and optimizer interfaces

The first source is original human GSM8K TRAIN at the revision specified in
`qwenlean/datasets/external_reasoning.py`. Archive exact original TRAIN, TEST and
MIT license files under names `gsm8k-train.jsonl`, `gsm8k-test.jsonl` and
`gsm8k-LICENSE`. The importer validates registered hashes; TEST questions are
used for contamination metadata only. Every output destination must be new.

```bash
uv run --frozen python scripts/audit_external_sources.py --gsm-archive data/raw/<pinned-source> --output data/raw/<new-bounded-audit>
uv run --frozen python scripts/import_external_reasoning.py --archive data/raw/<pinned-source> --output data/generated/<new-external-pilot> --count 768 --dev-count 64
uv run --frozen python scripts/approve_external_dataset.py --corpus data/generated/<new-external-pilot> --published-human-qa data/raw/<pinned-human-qa-publication>.html --output data/processed/<new-approved-pool> --config configs/training/external-human-smoke-supervised.yaml
```

The QA publication is the exact archived `2110.14168v2` HTML registered in
`external_contract.py`. Published human QA is an explicit bounded reference-data
approval basis, not formal verification of unannotated prose or fresh individual
inspection. Arithmetic/completion/repetition flags fail this route. An alternative
`--reviews <jsonl>` accepts actual individual human or permitted independent
frontier judgments. The importer's review bundle starts with every checkbox unset;
no coding-agent prose or rewritten response may enter targets.

```bash
uv run --frozen --no-sync qwenlean train-sft --config configs/training/external-human-smoke-supervised.yaml --dataset data/processed/<actual-approved-pool>/train.jsonl --quality-manifest data/processed/<actual-approved-pool>/manifest.json --output adapters/<new-external-smoke>
```

Freeze DEV before model training; this CPU operation does not generate responses:

```bash
uv run --frozen python scripts/freeze_external_dev.py --corpus data/generated/<external-pilot> --retention data/generated/<existing-procedural-cohort> --output data/generated/<new-frozen-dev> --report reports/<new-protocol>.json
uv run --frozen python scripts/phase4_dev.py --cohort data/generated/<frozen-dev> --output runs/<new-stock-baseline>
```

The frozen external-human-v1 cohort has 24 source DEV questions (eight per
operation-count bin) plus the existing 32 procedural retention tasks. Both modes
use the same task identities and respective fixed samplers; no TEST evaluation.
Run stock and adapter separately with the identical cohort; `--adapter` selects
a saved unfused adapter. A baseline stop stays partial evidence, never accuracy.

Only run after source/quality and resource gates pass, one workload under the lease.
This is four real optimizer steps for technical validation, not an improvement
experiment. The maximum training sequence is 1024; actual human pilot sequences
are all under 512, complete and never truncated. Completion of this smoke will
not establish longer-sequence capacity. One optimizer update is measured in a
failed retry; the full technical smoke remains incomplete.

SFT checks permission before reading tokenized records or loading tensors, then
reproduces exact source text, official thinking prefill, first supervised token,
closing think, final answer and EOS. Prompt tokens are loss masked. Batch size one,
AdamW, accumulation and explicit mixed-attention LoRA modules are implemented.
All frozen live base tensors are fingerprinted before/after and after fresh reload,
including unadapted layers, embeddings and norms. Adapter logits must match after
save/reload and fresh inference must succeed. Existing destinations are refused;
optimizer resume is not implemented. Native context/RoPE/tokenizer remain unchanged.

No improvement pilot starts until the matched stock baseline is complete and
hardware feasibility is measured. Do not use training loss or earlier EOS as a
substitute for paired correctness/efficiency and capability-retention evaluation.
The earlier procedural DEV cohort is a retention/domain-shift check relative to
GSM8K; its historical solver ID/OOD labels do not describe the new teacher corpus.

## Historical natural/solver reproduction only

The 16×4 Qwen pilot is complete. No new teacher generation/resume is permitted.
Offline reanalysis requires an explicit completed archive and new destination:

```bash
uv run --frozen python scripts/train_candidates.py --publish-run runs/<completed-natural-pilot> --output data/generated/<new-historical-analysis>
```

Natural and solver approval/build scripts require `--historical-reproduction`
and emit reproduction-only manifests. Current SFT rejects them; original validators
retain their exact-parent/proof checks. Do not relabel historical targets external.
The [solver contract](phase4-verified-reasoning.md) records the superseded experiment.

Tail-controller recovery uses separate durable job directories for each task/cap.
A saved generation is reused even if report publication was interrupted; contract
changes, duplicate attempts and explicit generation failures are refused. Resume
never appends to a completed single-generation archive.

Once the completed prompt report exists, export measured figures with
`uv run --frozen python scripts/plot_phase2.py`. It reads completed report artifacts;
the optional tail figure distinguishes known survivors from earlier-censored
unknowns. Prompt lengths are observed capped lengths, not natural completion
lengths. Boundary task-bootstrap intervals can be degenerate on this small pilot.

## Publish compact data evidence

```bash
uv run --frozen python scripts/summarize_external_sources.py --archive data/raw/<bounded-audit> --output reports/<new-derived-source-measurements>.json
uv run --frozen python scripts/report_external_pilot.py --candidates data/generated/<pilot> --approved data/processed/<approved-pool> --smoke data/processed/<approved-smoke> --config configs/training/external-human-smoke-supervised.yaml --output reports/<new-corpus-audit>.json
```

The corpus report reproduces both approval/source contracts and tokenization;
its zero-step status is scoped to the corpus milestone. Future optimizer results
must be recorded in a separate measured report.

## Hardware admission after the pressure event

Use the supervised training profile and its independent CPU watchdog; do not run
`external-human-smoke-bounded-prefix.yaml`, which is retained as a rejected
historical configuration. Current validation rejects its unsafe option before
model loading. CPU equivalence does not authorize a GPU experiment. Model jobs
remain stopped after the pressure event until fresh resource admission passes.
Inference/context controllers do not yet carry the new independent monitor and
must not be used to bypass this stop. The corpus and license audit remain valid.
See [saved safety results](../reports/local-training-resource-safety.md).

Check current training admission without any model/tensor loading:

```bash
uv run --frozen python scripts/check_training_resources.py --config configs/training/external-human-smoke-supervised.yaml --output runs/<new-cpu-admission-check>
```

Exit2 means admission refused; its exact RAM/swap snapshot remains saved. Exit0
means only CPU resource admission/readiness passed, not GPU feasibility or a
successful optimizer experiment. The supervised SFT command repeats admission and
requires a live owned watchdog before loading weights. Native MLX memory settings
are guidelines; independent running-pressure checks must remain enabled.

## Phase 4C supervised baseline and retry

The cohort is immutable:24 source questions plus32 procedural retention tasks,
56 tasks × thinking/non-thinking =112 jobs. Keep its task/prompt/seed/scorer/model
contracts. The driver uses the supervised resource policy and a fresh watchdog
session for each bounded invocation; no backend import/loading before admission.

Inspect saved jobs on the CPU:

```bash
uv run --frozen --no-sync python scripts/phase4_dev.py --cohort data/generated/external-human-v1-frozen-dev --output runs/phase4c-stock-dev --inspect
```

After healthy admission, a stock-only batch can execute at most two new jobs:

```bash
uv run --frozen --no-sync python scripts/phase4_dev.py --cohort data/generated/external-human-v1-frozen-dev --output runs/phase4c-stock-dev --batch-jobs 2 --retry-failed
```

`--retry-failed` permits a new timestamp/UUID attempt for an existing failed job;
it never overwrites/resumes failed files or regenerates an accepted durable EOS/cap
row. Config/model/scoring changes or duplicate successful records fail closed.
A new failure ends that invocation; there is no automatic retry loop. Infrastructure
failures stay outside correctness denominators. Mode/overall aggregates appear
only when all112 jobs are accepted; until then CPU inventory reports pending jobs
and raw failure/resource events with accuracy null. Do not reuse historical32-task
measurements as the completed external-human56-task baseline.

Watchdog sessions use random identity tokens, exact PID/create-time checks, fresh
heartbeats and durable stop records before signaling only the owned trainer. A
separate liveness thread detects monitor death during blocking work. The total
120-second smoke deadline covers loading, training, integrity, save/reload and
short inference; the15-second gradient deadline applies only to the gradient phase.
CPU fixtures verify monitor-death and normal timeout interruption/cleanup. Metal
session behavior remains unvalidated, and sampled checks cannot guarantee protection
from arbitrarily fast spikes. Native MLX memory limits are guidelines, not hard caps.

Reproduce the historical checkpoint/accounting audit without a model:

```bash
uv run --frozen --no-sync python scripts/audit_phase4c_training.py --output reports/<new-derived-historical-audit>.json
```

Leave the original zero-step report and checkpoint bytes intact. No inference from
quarantined weights; only a complete source/integrity/reload-approved adapter can
be attached. The source corpus and license audit are unchanged. Phase4C does not
authorize a128-update improvement run; future SFT configuration selection requires
the completed technical gates and full stock baseline.
