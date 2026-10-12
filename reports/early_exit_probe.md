# Counterfactual early-exit probes

Conditional forced-close recovery, not natural stopping, certainty, or population accuracy. Censored source savings are relative to observed cap only.

3/34 probes recovered the correct integer across six DEV sources. 2/6 sources had at least one successful tested prefix.

Final termination: 3 EOS, 31 capped at the predeclared 128-token final budget. Zero-prefix correct: 1/6. Extra explanation can exhaust the final budget even after a useful prefix; recovery failures must not be interpreted as proof of mathematical inability.

| Source task | Earliest successful tested prefix | Original correct | Original EOS |
|---|---:|---:|---:|
| v1-content-261d34b7df3c0d0a6eb6 | 512 | True | True |
| v1-content-13276a136d16a0f67643 | none | False | False |
| v1-content-89b0e50c1f45613d0cad | none | False | False |
| v1-content-ca26414b3c77cff24de9 | none | False | False |
| v1-content-6a64c022f1b4aa758836 | none | False | False |
| v1-content-19048e452785c34ac990 | 0 | True | True |

Uses actual saved chat-template prompt IDs plus generated reasoning IDs, then explicitly forces `</think>` and two newlines. Greedy final generation has 128 tokens and no repetition/presence penalties. Zero-prefix forced-close controls measure direct-answer recovery under the same final decoder. Original baseline records are untouched. Counts are nested within six tasks, not independent benchmark problems.

Reproduce/resume: `uv run --frozen python scripts/early_exit_probe.py`.
