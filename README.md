# QwenLean

**Cut the waste. Keep the reasoning.**

Research on whether `Qwen/Qwen3.5-0.8B` can retain or improve correctness while
reducing redundant reasoning and poor termination. Correctness dominates brevity:
a hard problem should still receive useful computation. No improved model is
claimed. The remote repository is [LeanQwen](https://github.com/lukifer23/LeanQwen);
all development stays on `main`.

## Current status

Phase 2 is stopped at the completed **144-trajectory prompt-policy milestone**
(12 DEV tasks × 3 matched seeds × 4 policies), at the user’s request.
P2’s integer-only final-line instruction produced **7/36 correct** versus **3/36**
for legacy `FINAL:` and **0/36** for boxed-answer wording. P2 reduced observed
mean total output by **6.5%**, but median/p95 reasoning remained capped at **2048**;
29/36 P2 trajectories still hit the cap. These small DEV results do not establish
a capability-preserving Pareto improvement. See the [prompt study](reports/prompt_policy_ablation.md)
and [checkpoint / next steps](reports/phase2_checkpoint.md). No model worker remains active.
The audited harness now supports explicit stream failures, safe resume, independent
task/prompt identities, four output policies and task-clustered multi-seed analysis.
Procedural-v2 has eight families, four structural difficulty bins and 64 tasks per
split with zero detected cross-split structural-signature collisions. See its
[measured generator audit](reports/procedural_v2_quality.md) for limitations.

Tail diagnostics, counterfactual early-exit probes, local semantic measurement,
guard replay and natural TRAIN selection code are implemented; their model studies
remain pending. A guarded `train-sft` interface exists but has not executed real
optimizer steps or adapter reload validation. No substantial SFT, preference
training, compression pipeline, adaptive budget controller or CUDA backend has run.
The [guide](docs/guide.md) distinguishes implemented interfaces from validated results.

Original unquantized bf16 text inference works on an M3 Pro with 18 GiB unified
memory using MLX-LM. The native 262144-token context and RoPE remain unchanged;
inputs are never silently truncated. Output caps and bounded training examples
are separate budgets. A documented 128K floor is allowed, but no reduction has
been applied. Long-context behavior has not been empirically validated.

## Historical first-pass evidence

These completed DEV results used a **2048 total-output-token cap** and historical
`FINAL:` prompts/scorer v5. Thinking lengths are right-censored, not natural
completion lengths. The prompt wording itself can provoke format deliberation;
The completed Phase 2 prompt pilot measures its contribution alongside seed
variance and ordinary mistakes; higher-cap diagnosis remains pending.

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
| TRAIN natural best-of-N | N=4; correctness, reasoning validity and clean closure precede length |
| Optimizer smoke | Approved TRAIN targets, actual LoRA steps, save/reload/inference verification |
| Real SFT | Later user review; no substantial run in this phase |

TEST stays sealed from model evaluation and training selection. Generator-only
contamination audits inspect split metadata. All response training targets must
come from Qwen itself; no proprietary closed-model teacher outputs. DEV reviewer
annotations are explicitly forbidden for training. Procedural code/stimuli and
this repository are Apache-2.0. No external benchmark dataset has been imported.

## Layout and documentation

```text
configs/           # stock samplers and explicit diagnostic configurations
qwenlean/
  datasets/        # versioned tasks, prompt policies, provenance
  evaluation/      # durable runs, comparisons, task-clustered statistics, censoring
  inference/       # MLX backend, token partitioning, defensive streaming
  metrics/         # literal/lexical/semantic proxies, conclusions, guard replay
  generation/      # correctness-first natural target selection
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
