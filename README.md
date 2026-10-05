# QwenLean

**Cut the waste. Keep the reasoning.**

Research on whether Qwen3.5-0.8B can move outward on the accuracy versus reasoning
compute Pareto frontier. The hypothesis is that repeated recalculation, circular
reconsideration and poor termination waste compute and can sometimes harm an answer.
These are hypotheses to measure, not claims of an improved model.

Correctness comes first. Token count alone is the wrong optimization target:
difficult problems may need long traces. We diagnose decoding before changing weights.

The GitHub repository is [LeanQwen](https://github.com/lukifer23/LeanQwen).
All development stays on `main`.

## Current status

Phase 2 is in progress: hardening resumption and stream handling, then separating
prompt-format effects, stochastic variance and capped termination. See the
[Phase 2 methodology](docs/phase2-methodology.md). New experiments are not yet complete.

The following measurements are the completed **historical first pass**.

Real inference works with the original, unquantized `Qwen/Qwen3.5-0.8B` checkpoint
on an M3 Pro with 18 GiB unified memory. MLX-LM is the selected text inference
backend; Transformers/MPS also passes an inference probe but uses slower hybrid
reference kernels. Adapter backward compatibility passes in MLX training mode;
no optimizer updates, SFT or preference training have run.

A tested evaluation harness, 60-problem DEV suite, stock thinking baseline,
official default-mode control, and initial decoding ablation are complete.
With a 2048 **total output** token cap:

| Variant | Task-correct / 60 | Median / P95 thinking tokens | Cap failures |
|---|---:|---:|---:|
| A: official thinking | 0 | 2048 / 2048 | 60 |
| B: repetition penalty 1.05 | 4 | 2048 / 2048 | 56 |
| A0: official non-thinking control | 24 | 0 / 0 | 17 |

A0's median total output is 381.5 tokens; a zero explicit thinking partition does
not imply zero reasoning-like calculations. Task grading accepts terminal numeric
answers/equations, independently of FINAL formatting; its calibrated contract is
[documented](docs/metrics.md). Earlier format-sensitive counts are retained in the
research log. A0 changes mode and its recommended sampler together.

A generated 60.4 output tokens/s including prefill, with a 1.684 GB MLX allocator
peak and 2.140 GB sampled process RSS peak; these memory measures overlap.
B reduced mean total output only 2.0%; median/p95 reasoning stayed at the cap.
There is no established material efficiency improvement or trained model.
Observed traces include correct calculations followed by repeated format
reconsideration, skipped operations, and short wrong answers. Zero detected exact
cycles does not mean zero pathological behavior.

Read the [first-pass decision report](reports/first-pass.md),
[baseline](reports/baseline.md), [sampling ablation](reports/sampling_ablation.md),
[default-mode control](reports/nonthinking_control.md), and
[manual trace review](reports/trace_review.md). These are small DEV results under a
cap, not general capability or untouched TEST results. Full raw runs stay local;
committed summaries and selected excerpts are inspectable but not a full output archive.

## Reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. No Docker.

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check qwenlean scripts tests
uv run --frozen python scripts/validate_environment.py
uv run --frozen python scripts/smoke_inference.py
uv run --frozen qwenlean build-dataset
uv run --frozen qwenlean eval --config configs/baseline.yaml
```

Run model commands sequentially. A repository-wide OS process lock rejects a
second model workload and recovers after process exit. Weights download into the
Hugging Face user cache; generated model files and run directories are ignored.

Use the completed baseline directory printed by `eval`:

```bash
uv run --frozen qwenlean analyze runs/<baseline-run>
uv run --frozen qwenlean eval --config configs/nonthinking.yaml
uv run --frozen python scripts/sampling_ablation.py --baseline-run runs/<baseline-run>
uv run --frozen python scripts/create_reports.py --baseline-run runs/<baseline-run>
# Only compare full runs with identical sample IDs and seeds:
uv run --frozen qwenlean compare runs/<baseline-run> runs/<candidate-run>
```

`generate --config ... --prompt ... --output ...` saves a real generation. `eval --resume` and task-paired `compare-prompts` are now available; see the guide.
There is currently no `train-sft`, preference trainer, trajectory compression,
embedding redundancy metric, or adaptive budget implementation. A CUDA backend
has not been implemented. These are possible later experiments, subject to the
data and measurement gates.

## Experimental design

Six deterministic families: integer arithmetic, multi-step discount word problems,
algebra, boolean logic, ordering constraints, and modular state transitions.
Algorithmic labels use exact rational scoring. Independently seeded TRAIN/DEV/TEST
pools have disjoint numeric ranges and prompt-hash contamination checks. Those
checks detect duplicate prompts, not all semantic overlap: the small boolean truth
domain can repeat across pools despite different instance labels. TEST is
reserved from tuning and training selection; first-pass experiments use DEV only.
The suite is intentionally small and does not establish general reasoning capability.

A uses official text-thinking sampling parameters with a practical **2048 total
output token cap**. The cap censors long traces. A0 controls for official default
non-thinking mode and its recommended sampler. The broad sampler stage changes
temperature, presence penalty or repetition penalty individually on the same 12
balanced DEV examples. A focused candidate is promoted only if its pilot contains
correct final answers and preserves pilot accuracy. No grid explosion or reward
coefficient tuning.

Every run saves unique IDs, prompts/labels/provenance, configuration and hashes,
per-sample seeds, rendered chat templates, emitted token IDs, complete raw output,
reasoning/final partitions, exact correctness, latency/throughput, memory,
termination and repetition diagnostics. Reports include percentiles, Wilson
accuracy intervals, paired bootstrap comparisons and actual trace excerpts.
Literal repeats, exact cycles and lexical redundancy are separate measurements.
Candidate correct-conclusion distance is a documented heuristic, not proof the
problem was already solved. The runtime exact-cycle guard is optional and disabled
in stock experiments; it never silently forces a final answer.

## Layout

```text
QwenLean/
├── AGENTS.md, README.md, LICENSE, pyproject.toml, uv.lock
├── configs/                 # baseline, default-mode control, sampler variations
├── qwenlean/
│   ├── cli.py
│   ├── datasets/            # procedural generators and fail-closed provenance
│   ├── evaluation/          # backend-independent runner and statistics
│   ├── inference/           # MLX backend and token-ID partitioning
│   ├── metrics/             # repetition, lexical proxy, conclusion distance
│   ├── scoring/             # exact final-answer scoring
│   └── utils/               # hashes, serialization, single-process lock
├── scripts/                 # actual compatibility probes, sweeps and reports
├── tests/
├── data/splits/             # reproducible small pools and manifest
├── runs/                    # ignored raw experiment artifacts
├── reports/                 # measured summaries, excerpts and figures
└── docs/                    # architecture, metric contract, research log
```

See the [operating guide](docs/guide.md), [exact tracked file manifest](docs/repository-files.txt),
[architecture](docs/architecture.md), [metrics](docs/metrics.md), and the
[research log](docs/research-log.md). This repository and its procedural stimuli
are Apache-2.0. Provenance gates reject DEV/TEST training, unknown licenses and
unverified closed-model origins. No external benchmark data has been imported.
