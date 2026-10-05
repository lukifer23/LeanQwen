# Running QwenLean

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
uv run --frozen python scripts/create_reports.py --baseline-run runs/<A-run> --nonthinking-run runs/<A0-run>
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
If a scoring bug is fixed, `scripts/recompute_metrics.py runs/<A-run>` reprocesses
actual emitted IDs, preserves `samples.original.jsonl`, and records the correction
in `reanalysis.json`. Rescore every run involved in a comparison under the same
version, then rerun `create_reports.py`; it refreshes saved sweep aggregates from
those records while preserving the original sampler-selection decision. This is
an offline metric correction, not a new experiment.

Read `docs/metrics.md` before interpreting any efficiency statistic. In particular,
a 2048-total-output-token cap bounds observed percentiles; zero detected exact
cycles does not exclude semantic reconsideration; and candidate correct-number
appearances do not prove the model had reached a reliable solution.

## Not implemented

Training-data candidate sampling, trajectory compression, dataset quality reports,
SFT training, preference optimization, adaptive budgets and a CUDA backend are
future work. No trained checkpoint is included. The optional exact-cycle guard
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
output budgets and bounded smoke-training sequence lengths do not change it.

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
