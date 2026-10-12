# Phase 3 reasoning diagnosis — 2026-10-12

Original pinned bf16 Qwen, P2, official thinking, matched replicate-zero seeds,
guard disabled. Seven predeclared DEV cases completed staged diagnosis; all 13
emitted-prefix comparisons passed. These selected cases are diagnostic evidence,
not a population accuracy estimate. No response here is a TRAIN target.

## Natural continuation

Two cases naturally completed correctly: arithmetic at 498 reasoning tokens and
algebra at 6122. Five remained right-censored: arithmetic, Boolean logic, word
problem and ordering at 8192; state machine at 16384. Outcomes beyond their
predeclared ceilings are unknown. Additional output restored a final answer in
the algebra case, but much of its observed continuation repeated an already
correct derivation rather than discovering a new solution.

The following qualitative review uses saved reasoning, candidate-conclusion cues
and ending passages. It does not independently certify every intermediate step
or determine an optimal stopping point. Cue-qualified numbers remain heuristics.

| Case | Observed behavior | Interpretation |
|---|---|---|
| `(50 * 41) - 44` | Correct multiplication/subtraction, alternative calculation and repeated final-format checks; correct EOS | Useful initial calculation followed by redundant checks; naturally terminates |
| `59*x+56=2593` | Correct 43 appears at cue token 658; repeatedly substitutes, redivides, questions interpretations and revisits formatting; correct EOS at 6122 | Correct solution followed by substantial semantic re-derivation and format fixation; longer allowance permits eventual completion |
| `(49 * 57) - 58` | Correct 2735 appears at cue token 468; later repeatedly recomputes 2850−57, mixes subtraction states, remains unclosed at 8K | Re-derivation with intermediate arithmetic drift; correct-number appearance is insufficient for success |
| Modular state sequence | Correct 0 appears at cue token 3325; repeatedly derives 318 mod 64=62, then 64 mod 64=0, questions operation order and meaning until 16K | Correct path followed by persistent instruction doubt and equivalent-state revisits; natural completion unresolved |
| Boolean expressions | Correct count 3 appears at cue token 706; repeatedly reevaluates the same four expressions and operator semantics until 8K | Semantic re-derivation with occasional operator confusion; capped, incomplete answer |
| Discount word problem | Correct 12701520 appears at cue token 936; revisits boxes/items/discount interpretation and later drifts in percentage arithmetic until 8K | Useful initial multi-step calculation followed by low-value reconsideration and drift |
| Runner ordering | Repeatedly questions whether runner identifiers denote values or names; increasing nested bullets; no cue-qualified solution by 8K | Failure to resolve the representation within the tested budget; cannot infer inability at all larger budgets |

Raw evidence: [tail records](termination_tail_records.jsonl),
[censor-aware measurements](termination_tail_measurements.json),
[tail report](termination_tail.md). Original trajectories are unchanged.

## Forced-close recovery

The separate six-source experiment ran all 34 predeclared prefix probes, including
six zero-prefix controls. Three correct EOS answers occurred across two tasks;
31 final responses hit the 128-token cap. Arithmetic recovered at zero and 256
tokens, but failed at intervening tested prefixes; algebra recovered at 512.
Success is not monotonic in prefix length. The algebra source is a different task
from the tail algebra case.

Many capped final responses began another step-by-step explanation. Thus the
tested intervention is limited both by final verbosity and by its small final
budget. It does not prove that 31 prefixes were mathematically insufficient.
Forced closure is not a validated stopping policy, and observed savings for
capped source traces are relative to their observed cap, not natural completion.

Evidence: [early-exit report](early_exit_probe.md),
[measurements](early_exit_measurements.json), [raw probes](early_exit_records.jsonl).

## Implication for training

Local CPU MiniLM calibration completed on 36 reviewed DEV pairs from five tasks.
At the retained 0.90 threshold: TP 11, FP 1, TN 16, FN 8; precision 91.7%, recall
57.9%. Lexical cosine at 0.90 measured 81.8% precision and 47.4% recall. These
are curated, single-reviewer labels without held-out calibration; semantic
similarity is not an oracle of waste or a training-selection filter. On the
matched 144-record prompt study, mean semantic density was P0 11.39%, P1 10.61%,
P2 9.61%, P3 10.06%. These are descriptive DEV measurements, not independent
population estimates or evidence of trained-model improvement.

The three already-declared literal guard configurations replayed all 161 unique
saved prompt/tail generations at the real 16-token cadence: zero triggers, zero
observed savings. No new threshold sweep or live guard run was justified.
See [semantic calibration](metric_calibration.md) and
[full guard replay](runtime_guard_calibration.md).

The observed problem includes semantic re-derivation, instruction doubt,
format fixation and mathematical drift. Exact-cycle detection alone is poorly
matched to these traces. Correctness and complete natural targets remain the
first training-data gate; neither a correct number inside an unclosed trace nor
a shorter wrong response is an acceptable target. Preserve productive multi-step
reasoning and necessary correction when reviewing TRAIN candidates.
