# Termination tail

Seven predeclared DEV cases; policy P2, original weights, official thinking, guard off, matched replicate-zero seeds. Higher caps are skipped after EOS.

| Horizon | EOS by horizon | Still generating | Unknown beyond earlier censoring | Survival bounds |
|---:|---:|---:|---:|---:|
| 2048 | 1 | 6 | 0 | 85.7%–85.7% |
| 4096 | 1 | 6 | 0 | 85.7%–85.7% |
| 8192 | 2 | 5 | 0 | 71.4%–71.4% |
| 16384 | 2 | 1 | 4 | 14.3%–71.4% |

Right-censored latest observations: 5/7. Exact prefix checks: 13, all passed. Observed natural reasoning lengths: [498, 6122].

This selected cohort cannot estimate population nontermination. Cap lengths are censored observations, not natural lengths. Unknown outcomes at 16K remain unknown for cases whose predeclared escalation ended at 8K. Survival horizons use total generated tokens; completed reasoning lengths exclude controls/final text.

Reproduce/resume: `uv run --frozen python scripts/termination_tail.py`. Raw records and prefix comparisons accompany this report.
