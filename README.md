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

Real inference works with the original, unquantized `Qwen/Qwen3.5-0.8B` checkpoint
on an M3 Pro with 18 GiB unified memory. MLX-LM is the selected text inference
backend; Transformers/MPS also passes an inference probe but uses slower hybrid
reference kernels. Adapter backward compatibility passes in MLX training mode;
no optimizer updates, SFT or preference training have run.

A tested evaluation harness and 60-problem DEV suite are implemented. The stock
thinking baseline and default-mode control are complete; staged decoding experiments
are in progress. At the 2048-token cap, the thinking baseline produced 0/60
correct final answers and all samples reached the cap (60.4 output tokens/s). The official default-mode control achieved 15/60 correct, median 381.5 total
output tokens, and 17/60 cap failures. An improved thinking sampler or trained
model has not been established. Small compatibility measurements are under `reports/`;
See [baseline measurements](reports/baseline.md) and the
[default-mode control](reports/nonthinking_control.md); these are small DEV results
under a cap, not general capability or untouched TEST results.

## Reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. No Docker.

```bash
uv sync --frozen --extra compat
uv run --frozen --extra compat pytest -q
uv run --frozen --extra compat ruff check qwenlean scripts tests
uv run --frozen --extra compat python scripts/validate_environment.py
uv run --frozen --extra compat python scripts/smoke_inference.py
uv run --frozen --extra compat qwenlean build-dataset
uv run --frozen --extra compat qwenlean eval --config configs/baseline.yaml
```

Run model commands sequentially. A repository-wide OS process lock rejects a
second model workload and recovers after process exit. Weights download into the
Hugging Face user cache; generated model files and run directories are ignored.

Use the completed baseline directory printed by `eval`:

```bash
uv run --frozen --extra compat qwenlean analyze runs/<baseline-run>
uv run --frozen --extra compat qwenlean eval --config configs/nonthinking.yaml
uv run --frozen --extra compat python scripts/sampling_ablation.py --baseline-run runs/<baseline-run>
uv run --frozen --extra compat python scripts/create_reports.py --baseline-run runs/<baseline-run>
# Only compare full runs with identical sample IDs and seeds:
uv run --frozen --extra compat qwenlean compare runs/<baseline-run> runs/<candidate-run>
```

`generate --config ... --prompt ... --output ...` saves a real generation. There is currently no `train-sft`, preference trainer, trajectory compression,
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

See [architecture](docs/architecture.md), [metrics](docs/metrics.md), and the
[research log](docs/research-log.md). This repository and its procedural stimuli
are Apache-2.0. Provenance gates reject DEV/TEST training, unknown licenses and
unverified closed-model origins. No external benchmark data has been imported.
