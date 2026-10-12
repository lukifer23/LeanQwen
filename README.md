# QwenLean

**Cut the waste. Keep the reasoning.**

Research on whether `Qwen/Qwen3.5-0.8B` can retain or improve correctness while
reducing redundant reasoning and poor termination. Correctness dominates brevity:
a hard problem should still receive useful computation. No improved model is
claimed. The remote repository is [LeanQwen](https://github.com/lukifer23/LeanQwen);
all development stays on `main`.

## Current status

Phase 4 now uses independently sourced external reasoning. The first pilot contains
**768 distinct original human-written GSM8K TRAIN solutions**, with 64 reserved
DEV questions. The source is pinned and MIT licensed; original reasoning is
preserved. Exact annotated arithmetic, final-label linkage, completion, template
and contamination checks pass. Approval uses published human QA plus screening;
prose semantics are not formally verified and no fresh individual review is claimed.
See the [source comparison](reports/external-source-comparison.md),
[corpus audit](reports/external-human-pilot.json) and
[training-data strategy](docs/training-data-strategy.md).

Qwen is exclusively the student/evaluation target. Qwen-family, LFM and weak local
models cannot author, rewrite or judge targets. Historical natural and solver
corpora remain preserved and excluded from current SFT. Direct Python/MLX is the
model engine; no LM Studio. A separate source-aware gate validates licensing
before training data/tensor loading and reproduces exact source/template/token IDs.
The four-step technical smoke on eight human targets stopped after its first
backward pass at 2.583 GiB available RAM, below the unchanged 4 GiB floor.
A subsequent [memory-efficient retry](reports/external-human-smoke-memory-efficient-attempt1.json)
completed **one real optimizer update**, then stopped at the same floor before
update2. No saved adapter, successful four-step smoke or improvement result exists.
See [saved smoke stop](reports/external-human-smoke-memory-stop.md).
A new [56-task matched DEV protocol](reports/external-human-dev-protocol.json)
combines 24 human-source questions and the existing 32 procedural retention tasks.
The earlier stock baseline remains partial after memory stops. A complete matched baseline, actual smoke and capacity measurements remain
required before an improvement pilot.

Historical Phase 3: PonderScope finished naturally before LeanQwen model work.
Stock context probes scored 3/5 at 2K and 4/5 at 8K/32K envelopes, with all
positional retrieval questions correct. At 32K, 512-token prefill chunks reduced
measured MLX peak from 5.016 to 2.960 GB at unchanged accuracy. Live short-request
128K-to-262K profile switching preserved all loaded tensors and model arguments.
See [measured context results](reports/phase3-context-stock.md),
[current progress](reports/phase3-progress.md) and
[context management / restoration](docs/context-management.md).
The seven-case tail completed: two correct natural completions, five censored,
all 13 prefix checks exact. Early exit recovered 3/34 probes; 31 exhausted the
128-token final budget. Semantic calibration and offline guard replay completed.
See [reasoning diagnosis](reports/phase3-reasoning-diagnosis.md). No trained
adapter or reasoning improvement is claimed. The 16×4 natural TRAIN pilot completed:
4/64 correct EOS responses, all from `2 + (-3)`; 60/64 capped and unclosed.
One distinct eligible task is insufficient for meaningful SFT. See the
[generation review](reports/phase3-natural-pilot-review.md) and
[data options and safeguards](docs/training-data-strategy.md).
That completed negative result is preserved. Phase 4 continuation is authorized;
no optimizer has run yet.

Phase 2 completed the **144-trajectory prompt-policy milestone**
(12 DEV tasks × 3 matched seeds × 4 policies).
P2’s integer-only final-line instruction produced **7/36 correct** versus **3/36**
for legacy `FINAL:` and **0/36** for boxed-answer wording. P2 reduced observed
mean total output by **6.5%**, but median/p95 reasoning remained capped at **2048**;
29/36 P2 trajectories still hit the cap. These small DEV results do not establish
a capability-preserving Pareto improvement. See the [prompt study](reports/prompt_policy_ablation.md)
and [historical checkpoint / next steps](reports/phase2_checkpoint.md).
The audited harness now supports explicit stream failures, safe resume, independent
task/prompt identities, four output policies and task-clustered multi-seed analysis.
Procedural-v2 has eight families, four structural difficulty bins and 64 tasks per
split with zero detected cross-split structural-signature collisions. See its
[measured generator audit](reports/procedural_v2_quality.md) for limitations.

Tail diagnostics, counterfactual early-exit probes, local semantic measurement
and offline guard replay have run. Natural TRAIN generation/automatic selection
completed; no quality-approved training pool was created.
A guarded `train-sft` interface exists but has not executed real
optimizer steps or adapter reload validation. No substantial SFT, preference
training, compression pipeline, adaptive budget controller or CUDA backend has run.
The [guide](docs/guide.md) distinguishes implemented interfaces from validated results.

Original unquantized bf16 text inference works on an M3 Pro with 18 GiB unified
memory using MLX-LM. The native 262144-token context and RoPE remain unchanged;
inputs are never silently truncated. Runtime profiles select a combined input/output
ceiling: `mac-128k` (131072, default), `mac-100k` (102400, fallback) or `native-262k`
(262144). Checkpoint/RoPE/tokenizer file hashes are unchanged across profile selection.
Output caps and bounded training examples are separate budgets. Long-context
behavior is measured through 32512 actual input tokens with a 256-token output
reservation. 100K/128K/262K inference remains unverified; measured memory savings
come from smaller prefill chunks, not the operating ceiling.

## Historical first-pass evidence

These completed DEV results used a **2048 total-output-token cap** and historical
`FINAL:` prompts/scorer v5. Thinking lengths are right-censored, not natural
completion lengths. The prompt wording itself can provoke format deliberation;
The completed Phase 2 prompt pilot measures its contribution alongside seed
variance and ordinary mistakes; higher-cap diagnosis is now reported separately.

| Variant | Task-correct / 60 | Median / P95 thinking tokens | Cap failures |
|---|---:|---:|---:|
| A: official thinking | 0 | 2048 / 2048 | 60 |
| B: repetition penalty 1.05 | 4 | 2048 / 2048 | 56 |
| A0: official non-thinking control | 24 | 0 / 0 | 17 |

A generated 60.4 output tokens/s including prefill; MLX allocator peak was 1.684 GB
and sampled RSS peak 2.140 GB, overlapping memory measures. B reduced mean total
output only 2.0%, with unchanged median/p95 thinking length. A0 changes both mode
and its official sampler; zero explicit thinking tokens does not mean no calculations.
No material efficiency improvement has been established. Zero detected exact cycles
does not exclude repeated reconsideration.

The [first-pass report](reports/first-pass.md), [baseline](reports/baseline.md),
[sampling ablation](reports/sampling_ablation.md), [default-mode control](reports/nonthinking_control.md)
and [trace review](reports/trace_review.md) remain unchanged and discoverable.
These are small DEV measurements, not general capability or untouched TEST results.

## Reproduce

Apple Silicon macOS, Python 3.12 and [uv](https://docs.astral.sh/uv/). No Docker.

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check qwenlean scripts tests
uv run --frozen python scripts/validate_environment.py
uv run --frozen qwenlean eval --config configs/baseline.yaml
uv run --frozen python scripts/prompt_policy_ablation.py
```

Run model commands sequentially. A repository-wide OS lease rejects duplicate
workloads; do not remove its lock file while a process is active. Configurations,
model pins, environments, dataset hashes, seeds, full outputs, emitted IDs and
metrics are saved per run. Weights/caches/raw run directories stay out of Git.
Environment validation refuses existing output; use `--output` for a new manifest.

```bash
uv run --frozen qwenlean analyze runs/<run>
uv run --frozen qwenlean eval --config <same-config> --dataset <same-dataset> --resume runs/<incomplete-run>
uv run --frozen qwenlean compare-prompts runs/<study> runs/<study> --left-policy P3 --right-policy P2 --output reports/<new-comparison>.json
```

Read the [operating guide](docs/guide.md) for exact diagnostic, candidate and guarded
training commands. Historical reports are protected: reanalysis and report builders
require new derived destinations.

## Methodology

| Stage | Evidence / gate |
|---|---|
| DEV prompt study | 12 tasks × 3 matched seeds × 4 policies; official thinking, guard off |
| Targeted tail | Seven predeclared cases, staged 2K/4K/8K; at most two at 16K |
| Early exit | Explicit forced-close probes on saved own-prefix tokens; separate from natural stopping |
| Measurement calibration | Source-linked DEV judgments, local embeddings and exact guard replay |
| Historical natural best-of-N | Completed negative result; offline reproduction only, excluded from SFT |
| Optimizer smoke | Approved TRAIN targets, actual LoRA steps, save/reload/inference verification |
| External reference data | Pinned licensed human solutions, published QA, exact screening and source-preserving targets |
| Real SFT | One bounded pilot only after external quality, complete matched baseline, optimizer and capacity gates |

TEST stays sealed from model evaluation and training selection. Import contamination
checks hash official GSM8K TEST questions only; no TEST solutions are tokenized or
used as targets. TRAIN/DEV are distinct and source proportions are explicit.
Procedural code/stimuli and this repository are Apache-2.0; imported GSM8K human
solutions retain their separate MIT notice. No paid API generation or proprietary
closed-model targets are used. DEV reviewer annotations stay outside training.

## Layout and documentation

```text
configs/           # stock samplers and explicit diagnostic configurations
qwenlean/
  datasets/        # versioned tasks, prompt policies, provenance
  evaluation/      # durable runs, comparisons, task-clustered statistics, censoring
  inference/       # MLX backend, token partitioning, defensive streaming
  metrics/         # literal/lexical/semantic proxies, conclusions, guard replay
  generation/      # historical natural selection, reproduction only
  training/        # quality contracts and isolated MLX LoRA optimizer
  scoring/         # frozen v5 and prompt-policy v6 terminal scoring
  utils/           # hashes, environment manifests, single-model lease
scripts/, tests/, .github/workflows/
data/splits/       # historical v1 and structurally held-out v2 pools
reports/, docs/    # measured evidence, contracts, guides and research log
runs/              # ignored full local run artifacts
```

See [Phase 2 methodology](docs/phase2-methodology.md), [architecture](docs/architecture.md),
[metrics](docs/metrics.md), [research log](docs/research-log.md) and
[tracked file manifest](docs/repository-files.txt).
