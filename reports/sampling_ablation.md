# Initial sampling ablation

Staged DEV-only design: 12 balanced problems, two per family. Baseline pilot reuses original A samples. Each candidate changes one parameter.

| Variant | Accuracy | Median Think Tokens | P95 Think Tokens | Exact Output Loop Rate | Max-Out Rate | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| A pilot | 0.0% | 2048 | 2048 | 0.0% | 100.0% | 33.61 s |
| cooler | 8.3% | 2048 | 2048 | 0.0% | 91.7% | 29.77 s |
| presence-zero | 0.0% | 2048 | 2048 | 0.0% | 100.0% | 32.62 s |
| repetition-105 | 16.7% | 2048 | 2048 | 0.0% | 83.3% | 30.40 s |

Selection rule: Require nonzero pilot accuracy at least as high as A; then accuracy descending, cap rate ascending, median reasoning ascending. Promote best eligible pilot to full DEV evaluation

The eligible candidate was evaluated on all 60 DEV problems. This is sampler selection, not untouched TEST evidence.

| Variant | Accuracy | Median Think Tokens | P95 Think Tokens | Exact Output Loop Rate | Max-Out Rate | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| A official | 0.0% | 2048 | 2048 | 0.0% | 100.0% | 33.90 s |
| B candidate | 6.7% | 2048 | 2048 | 0.0% | 93.3% | 32.34 s |

Paired accuracy change: +6.7%; 95% bootstrap interval 1.7%–13.3%. Gained 4, lost 0.
Exact paired McNemar two-sided p=0.125; four unopposed wins do not establish significance at 0.05. The paired bootstrap interval above is discrete and should be interpreted alongside this exact test.
Token-identical pilot repeats: 12/12.

## Interpretation

Inspect correctness, cap rate and traces together. Shortening wrong capped traces is not evidence of preserved capability. No weights changed. Pilots are too small for stable claims; full DEV results also need independent confirmation and multiple generation seeds.

## Reproduction

```bash
uv run --frozen --extra compat python scripts/sampling_ablation.py --baseline-run runs/20261005T170152-A-official-thinking-0d62e9fa
uv run --frozen --extra compat python scripts/create_reports.py --baseline-run runs/20261005T170152-A-official-thinking-0d62e9fa
```
