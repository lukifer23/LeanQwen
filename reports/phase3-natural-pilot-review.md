# Completed natural TRAIN pilot — review checkpoint

2026-10-12. The original-Qwen generation pass completed all 64 responses and exited.
Training is held for user/independent-agent review. No optimizer steps, approved
SFT pool, adapter, solver corpus or external teacher dataset exists.

## Measured coverage

Sixteen procedural-v2 TRAIN tasks, eight families, one trivial and one medium
task per family, four seeds each. P2, official thinking, 4096 total-output tokens,
original pinned bf16 revision `2fc06364715b967f1860aea9cf38778875588b17`, guard off,
MLX, default `mac-128k` runtime ceiling. Run started on clean commit `13b958a`.
All public generation fields match their original local raw records; no generation
or historical score was rewritten for this review.

| Measure | Result |
|---|---:|
| Correct final responses / natural EOS | 4/64 (6.25%) |
| Capped responses / unclosed thinking | 60/64 (93.75%) |
| Distinct tasks with any eligible natural target | 1/16 (6.25%) |
| Trivial accuracy | 4/32 (12.5%) |
| Medium accuracy | 0/32 |
| Arithmetic accuracy | 4/8; only the trivial task succeeds |
| Each of the other seven families | 0/8 |
| Mean reasoning / total output tokens | 3883.19 / 3883.50 |
| Median, p90 and p95 reasoning tokens | 4096; right-censored |
| Exact output/reasoning cycles detected | 0 |
| Mean literal / lexical redundancy density | 11.94% / 12.97% |
| Generation errors | 0 |
| Sum of request latencies | 58.66 minutes |
| Aggregate output throughput | 70.62 tokens/s |
| Peak recorded MLX allocation / RSS | 1.772 / 2.155 GB; overlapping measures |
| Minimum sampled available system RAM | 4.534 GB (4.223 GiB) |

The request-time sum excludes initial model load and controller overhead. These
are selected TRAIN-task generation statistics, not a general benchmark or
stock-versus-adapter comparison. Fifteen tasks failed all four seeds; one succeeded
all four. There is no mixed correctness within a task in this small pilot, although
successful reasoning length varies substantially. No TEST model evaluation ran.
Literal/lexical densities are repetition proxies, not verified fractions of wasted
computation or a substitute for calibrated semantic and reasoning-quality review.

## What the responses show

All successful responses calculate `2 + (-3) = -1`. Their reasoning lengths are
302, 1605, 417 and 440 tokens; complete prompt/response lengths are 335, 1638,
450 and 473. All four final answers are correct and naturally closed. All repeat
format checking; the 1605-token response repeatedly questions a settled answer
and its negative-sign/whitespace formatting. The shortest selected response still
contains unnecessary final-format verification and imprecise sign-rule prose.
Replicate three entertains a wrong chained equation ending in 1, explicitly
rejects it, and restores -1. A correct final answer is not proof that every
intermediate claim was valid or necessary. Automatic tier A is an eligibility
proxy, not certification of efficient reasoning.

At 1024 tokens, three eligible trajectories fit and one is excluded whole. At
2048/4096, all four fit. Each bound still covers just **one distinct task**. These
are token-fit counts, not measured training-memory feasibility. No truncation or
reviewer text was applied to a target, and no quality-approved training manifest
was created.

Representative capped traces show:

| Family | Observation |
|---|---|
| Arithmetic | Re-derives the correct value 2, then treats subtracting a negative as a possible typo and reopens sign interpretation |
| Ordering | Repeatedly questions identifier/rank semantics despite the prompt explicitly distinguishing them; simple chain still caps |
| Sequence | Conflates the glossary for sort/rotate/negate with the requested operations; repeatedly questions supplied definitions |
| Word problems | Reconsiders item counts, discounts and fees; one trace uses 6×17=108 rather than 102, another spends its budget interpreting the fee list |
| Algebra | Finds -9 for the simple equation but revisits 18×12 and alternate parses; nested equation trace loses wrapper factors |
| Boolean logic | Misreads supplied values/expressions, corrects some, then repeatedly re-evaluates operator semantics |
| State machine | Repeatedly revisits reduction order and equivalent modular calculations after correct-state cues appear |
| Counting | Finds the three even singletons but digresses into size-two subsets; medium case repeatedly confuses odd/even membership |

Thirty-nine of sixty capped traces have a cue-qualified correct-number appearance.
These are heuristics, particularly weak for answers 0/1; they do not establish
39 solved problems, a safe stopping point or valid complete targets. Capped final
scores are failures to deliver an answer within this budget, not proof of inability
at every possible larger budget. TRAIN semantic embedding measurement was **not**
run; observed semantic re-derivation here is qualitative sampled review.

All sixteen rendered-task answers were recomputed in an assistant mathematical
review and agreed with the labels. This was not blinded or independently reviewed
by a human. Some rendering choices remain possible confounds: sequence prompts
define unused operations, fee lists use bracket notation, arithmetic uses adjacent
signs, and every task includes final-line wording. These patterns are hypotheses
for a future distinct prompt condition, not proof of a cause. Preserve the evaluated
v2 prompts and matched evaluation stimuli; do not rewrite them retrospectively.

## Narrow checker limitations

Four trajectories contain automatically flagged numeric lines. The ordinary
word-problem equations `6×17=108` and `60+42=108` are genuinely wrong. However,
the counting flags `1+1=0`, `2+0=0`, `5+1=0`, `6+0=0` occur under an explicit
modulo-2 context; treating them as ordinary integer equations yields false flags.
A state trace ends exactly at `8×9=7` because of the cap; its unfinished RHS is
not a certified claim that the product is 7. The checker also skips chained
equations, including the rejected wrong chain in a naturally completed response.

Do not interpret the flag count as a verified mathematical-error rate. No
scoring/selection rule was changed during this review. These issues do not change
coverage: the questionable flags occur in capped, independently ineligible traces.
Context-aware handling of modular claims, unfinished lines and rejected hypotheses
should be reviewed before using this checker more broadly as a quality gate.

## Recommendation for the next decision

**Do not run a model-improvement SFT pilot on this pool.** One elementary task
cannot demonstrate general reasoning adaptation; repeating it increases exposure
without adding task diversity. A technical optimizer smoke is a separate possible
future decision, not evidence of improvement and not authorized to start now.

If continuation is chosen, prioritize diverse solver-verified complete targets,
with exact intermediate checks and a separate fail-closed origin contract. Retain
productive longer reasoning and necessary correction. Consider a small reviewed
licensed source for language diversity, rather than automatically importing a
large verbose reasoning corpus. Keep teacher permissions/provenance explicit.
Use matched held-out accuracy, redundancy, non-thinking utility and the exact
supported context tests to reject regressions. More token allowance or lower
training loss alone does not establish success. No additional large guard sweep
or long-context stress run is justified by this pilot.

## Evidence and review scope

- [Original quality report](train_candidate_quality.json) and
  [raw responses/tokens](train_candidate_records.jsonl).
- [Original run summary with source hash](phase3_natural_pilot_summary.json).
- [CPU audit](phase3_natural_pilot_audit.json): source hashes, full configuration,
  per-task IDs/lengths, recomputed label review, checker flags and complete-fit counts.
- [Earlier partial snapshot](phase3_natural_pilot_snapshot.json), preserved as history.
- [Idle-process/lease checkpoint](phase3_idle_checkpoint.json).
- [Data strategy and capability safeguards](../docs/training-data-strategy.md).

Review covered all 64 statistical records, all 16 rendered problems/labels, all
four naturally completed responses in full, representative opening/ending passages
from replicate zero of all fifteen capped tasks, and focused contexts of flagged
claims. It does not certify every intermediate step in all 64 responses.
Reviewer calculations/prose are measurement metadata, explicitly not SFT targets.
