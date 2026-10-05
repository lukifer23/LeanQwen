# Measurement contract

Final task correctness is programmatic, using terminal numeric FINAL/boxed/plain answers
where present. Quoted formatting examples or markers followed by reconsideration
are not terminal answers. If those are absent, accept a unique terminal numeric conclusion after explicit
final-channel cues such as "integer solution is", "answer is", or "Total True
Expressions:". Conflicting numeric FINAL markers are rejected; nonterminal calculation cues
are not treated as final answers. The extraction never uses
ground truth to select an output number and never scores the thinking channel.
Labels must be valid exact rational values. This conservative grammar cannot
score every natural-language answer; its method/version are saved per record.

The historical `strict_final_correct` extractor searches numeric FINAL markers
anywhere in the final channel; it is retained only for audit and can credit
nonterminal formatting examples. Historical first-pass accuracy uses terminal v5;
new prompt-policy experiments use the separately frozen v6 contract documented below.
Keep `strict_final_correct` and `format_compliant` separate from task `correct`.
A response that says "Total True Expressions: 3" and then leaves `FINAL:` empty
can be task-correct and format-noncompliant. Earlier v1 measurements conflated
these; first-pass raw outputs were rescored with v5 and original metrics retained.
Those measurement histories are now preserved unchanged. A conclusion must end the final
channel, with only whitespace/punctuation or an empty FINAL marker afterward;
ongoing reconsideration after a numeric result is not a terminal answer.
Terminal standalone numeric lines and numeric equation results are accepted,
as are explicit total-count/count-of conclusions. A trailing empty FINAL marker
or an explicit notice that the answer follows the requested format may be
removed before terminal extraction; continued reconsideration is never removed.
These rules are independent of ground truth. Missing/truncated answers still count as incorrect unless the final channel
contains an unambiguous answer. Reasoning-only correct numbers receive no accuracy
credit. Format compliance requires a trailing numeric FINAL marker.

Token partitions use actual emitted IDs, not separately retokenized strings.
Reasoning excludes `<think>`, `</think>` and EOS. Final text excludes those controls.
Total = reasoning + final + controls, including the sampled EOS (one compute step).
The opening `<think>` in the prompt is not an output token. A missing `</think>`
leaves all output in reasoning. Non-thinking prompt precloses the empty block.
Wall-clock latency includes prefill/decode/synchronization and runtime guard costs,
excluding post-hoc scoring/metrics. Aggregate output throughput = emitted tokens /
summed inference latency; backend decode throughput is recorded separately.

Reasoning mean/p50/p75/p90/p95/p99 and rates strictly greater than 512/1024/2048/4096
are reported. A 2048-total-token cap censors the tail: zero values above the cap
cannot establish that unlimited traces would be short. Inspect cap-hit rate.

Literal n-grams use normalized words and mathematical operators, with n=5/8/12.
Require at least max(2,n/3) non-stopword units. Report repeated unique n-grams and
redundant occurrences / eligible occurrences. Repetition density is the union
of *subsequent* occurrences of repeated spans of at least eight words, divided by
all word units; the first occurrence is not charged. Longest repeated span is
exact and non-overlapping, in word units. These distinguish terminology reuse
from substantial copied blocks, but cannot establish uselessness by themselves.

Report reasoning-only `loop_rate` and all-output `output_loop_rate` separately.
The latter also inspects final-channel text, particularly important for non-thinking
controls. A zero thinking partition does not establish low total output compute.

Loop heuristic: exact repetition of a 24–128 emitted-token block at least three
consecutive times. Offline detection checks every endpoint. Runtime guard checks
every 16 output tokens, only during reasoning, logs period/repeats/start/end, and
stops without forcing an answer. False positives must be manually reviewed on
real traces before enabling it as a default. Narrow exact matching misses loops
that change wording or cycle over long periods. Zero detected loops is not zero
pathological behavior. Thresholds are exposed in YAML.

Semantic redundancy is initially a **lexical proxy**, not an embedding metric
or semantic oracle: sentence/line chunks of at least ten words, content-word
bag cosine >=0.90 with a previous chunk. Numbers are preserved. This catches
similar wording and reordered content but misses paraphrases and can flag
necessary re-verification. Keep separate from literal loop rate. Local embeddings
are deferred until these low-cost diagnostics are calibrated by inspection.

Candidate correct-conclusion distance: find the first exact ground-truth numeric
value after an equation RHS or an answer/result/total/therefore/thus cue in the
reasoning. Locate its endpoint by decoding prefixes of emitted IDs; count later
reasoning tokens. It is **not answer-reached ground truth**. A coincidental number,
intermediate calculation or later correction can mislead it. The saved evidence
must be inspected. `candidate_correct_then_wrong_final` excludes capped/guarded
traces; even EOS cases require human verification before claiming overthinking
caused answer damage. This metric uses labels for evaluation only.

Accuracy uncertainty: Wilson binomial intervals provide a non-degenerate boundary
interval even for zero/all correct. Also save deterministic-seed nonparametric sample bootstrap, 5000
resamples, 95% percentile intervals. Bootstrap intervals are degenerate at zero/all correct
and must not be interpreted as proof of zero population uncertainty. Comparisons use paired sample IDs and seeds.
Also report exact two-sided McNemar probability: under the paired null, the
number of gains among discordant samples is binomial with p=0.5; double the
smaller tail, capped at one. Four gains and zero losses give p=0.125, so a
positive small-sample percentile-bootstrap interval alone is insufficient to
claim a statistically established improvement. These reflect sample uncertainty, not multiple generation-seed variation.
Discrete 60-sample accuracy changes and six small families warrant cautious claims.
Primary comparisons remain accuracy versus reasoning compute; the supplementary
accuracy/p50 ratio is not a selection objective. No arbitrary reward coefficients.

Memory: MLX peak allocator bytes and sampled process RSS every 64 tokens plus
start/end. Both are imperfect and must not be summed (shared/unified accounting).
Neither is total machine usage. Available system RAM is recorded separately.

Latency observations were collected during interactive development, with some
CPU analysis and small unit-test operations in the session. Treat them as
preliminary hardware timings; dedicated latency comparisons should isolate
background activity and control power/thermal conditions. Model experiments ran
one at a time, with no overlapping model instances.


## Phase 2 contract (v6)

Historical v5 is frozen in `qwenlean/scoring/legacy_v5.py`; existing reports and
records are not overwritten. New rendered-policy experiments use
`terminal_numeric_policies_v6`, adding terminal numerical LaTeX fraction boxing,
display-math wrappers and explicit rejection of quoted/example hypotheses.
Task correctness is independent of policy. Format compliance is terminal box for
P1, integer-only final line for P2, legacy numeric FINAL for P3, and null for P0.
No label selects which model number to extract. Adversarial cases are unit-tested.

Replicate seeds match across policies. Statistics record unique tasks and total
generations. Bootstrap units are per-task replicate averages, not trajectories;
Wilson intervals are omitted when multiple observations share a task. Separate
prompt comparison enforces task/label/revision/weights/sampler/seed/cap/scorer/guard
invariants and permits only rendered prompt/policy differences. Its deltas include
correctness, applicable format compliance, EOS/cap, total/reasoning tokens, latency,
literal/lexical redundancy and format-meta reasoning. Within-task variability is
reported separately. The new format-meta line heuristic remains uncalibrated until
manual review; it is not a semantic redundancy metric.

## Phase 2 local calibration

`conclusions.py` separates correct numeric mentions, conclusion cues, later
conflicting cues, observed final correctness and unobserved closure. These are
text categories, not hidden-state claims. Structured DEV annotations reference
source generation IDs and are forbidden for training. Four complete matched
algebra traces have been inspected so far; aggregate conclusions await the study.

`semantic.py` implements local MiniLM chunk cosine independently of literal and
content-word overlap. CPU float32 inference is pinned and cached. Logical text
units are bounded to the encoder's window without truncating Qwen inputs or
changing its context. Numeric agreement is reported alongside similarity because
embeddings can confuse mathematically different statements. Default 0.90 and a
0.80/0.85/0.90/0.95 sensitivity sweep are pending actual measurement against 24
curated repeated-claim/operation pair judgments. These judgments do not certify
that a repeated step has zero expected value. The calibration is primarily one
algebra task and one assistant reviewer; no independent human labels are claimed.

`guard_replay.py` reproduces the runtime 16-token polling cadence on saved
reasoning IDs. Trigger position, observed tokens saved, original final correctness
and continuation text are retained. Every trigger needs individual review before
a live pilot. No triggers means precision cannot be estimated. Censored savings
are relative to observed output only.

### Format-meta calibration correction

Complete inspection of eight saved DEV traces found task-rule ambiguity mistaken
for output-format discussion by v1's generic `instruction`/`interpretation` cues.
A separate derived v2 metric anchors `FINAL:`, boxing, formatting and final-line
syntax, with bounded within-paragraph context. Raw run v1 fields are preserved,
and the predeclared policy-selection tie breaker remains v1; v2 is diagnostic.
Brief correct adherence to a final-line instruction is not pathological. Manual
annotation distinguishes presence from extended deliberation. Eight traces and
one reviewer provide limited calibration, not an oracle or population estimate.

The calibration pair pool now contains 36 judgments across five tasks, rather than
only algebra. It includes different truth-table rows/variables as hard negatives;
selection partly uses lexical similarity and remains nonrepresentative. Early-exit
probes include a predeclared zero-prefix control. A success at 128 tokens provides
weaker evidence if the same final decoder also succeeds without saved reasoning.
Forced closure and greedy/no-penalty final decoding remain combined interventions;
these results cannot isolate their causal contributions or prove internal certainty.

## Uncertainty at the boundary

Task-cluster bootstrap intervals can collapse to zero width when every observed
task has zero successes (or all succeed). Such an interval describes resampling
this small observed set; it does not establish that population accuracy is exactly
zero or one. The twelve-task prompt pilot has limited power, and its seed outcomes
are nested within tasks. Avoid interpreting boundary bootstrap intervals as strong
evidence of absence or guaranteed performance.
