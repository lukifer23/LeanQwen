# Running the implemented first pass

This guide describes implemented commands. It does not assume an adapter,
optimized decoder, successful training dataset, or benchmark improvement exists.
Use the status and reports in the README for measured findings.

## Install and validate

Use Apple Silicon macOS, Python 3.12 and uv. The default project dependencies
include MLX and are not a ready-to-install CUDA environment.

```bash
uv sync --frozen --extra compat
uv run --frozen --extra compat pytest -q
uv run --frozen --extra compat ruff check qwenlean scripts tests
uv run --frozen --extra compat python scripts/validate_environment.py
uv run --frozen --extra compat python scripts/smoke_inference.py
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
uv run --frozen --extra compat qwenlean build-dataset
uv run --frozen --extra compat qwenlean eval --config configs/baseline.yaml
uv run --frozen --extra compat qwenlean eval --config configs/nonthinking.yaml
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
uv run --frozen --extra compat qwenlean analyze runs/<A-run>
uv run --frozen --extra compat python scripts/sampling_ablation.py --baseline-run runs/<A-run>
uv run --frozen --extra compat python scripts/create_reports.py --baseline-run runs/<A-run> --nonthinking-run runs/<A0-run>
```

The sampler script uses twelve DEV examples (two per family) for a three-candidate
single-variable sweep. It can decline to promote any candidate. If it selects
one, it runs the full DEV suite and measures repeatability of pilot token IDs.
`reports/sampling_measurements.json` records the decision. A pilot cannot be
paired against a full 60-example run: compare exactly matching sample sets.

```bash
uv run --frozen --extra compat qwenlean compare runs/<A-run> runs/<B-run>
uv run --frozen --extra compat qwenlean generate --config configs/baseline.yaml --prompt 'Compute 17 * 23.' --output runs/one-generation.json
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
