# Measurement contract

Final correctness is strict and programmatic. The last unambiguous `FINAL:`
number, a single consistent boxed answer, or a bare numeric final response must
match exact rational ground truth. Conflicting FINAL values are failures.
Reasoning numbers never receive final-answer accuracy credit. Missing final
answers and truncated traces count as incorrect in unconditional accuracy.

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
These reflect sample uncertainty, not multiple generation-seed variation.
Discrete 60-sample accuracy changes and six small families warrant cautious claims.
Primary comparisons remain accuracy versus reasoning compute; the supplementary
accuracy/p50 ratio is not a selection objective. No arbitrary reward coefficients.

Memory: MLX peak allocator bytes and sampled process RSS every 64 tokens plus
start/end. Both are imperfect and must not be summed (shared/unified accounting).
Neither is total machine usage. Available system RAM is recorded separately.
