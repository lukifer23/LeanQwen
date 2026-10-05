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
nonterminal formatting examples. Main task accuracy uses the terminal v5 contract.
Keep `strict_final_correct` and `format_compliant` separate from task `correct`.
A response that says "Total True Expressions: 3" and then leaves `FINAL:` empty
can be task-correct and format-noncompliant. Earlier v1 measurements conflated
these; preserved raw outputs are rescored with v5 and original metrics retained. A conclusion must end the final
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


## PHASE2-AUDIT-014 — starting from main

Starting SHA: 8f67d3b2a1c217f5cad819ac2b8211b24db57e7d. Pulled main fast-forward;
working tree clean. All 32 starting tests and Ruff pass. Six completed local runs
match their stored summaries; token counts/config hashes/pinned revision agree.
Pinned model remains 2fc06364715b967f1860aea9cf38778875588b17. Exclusive process
lease available; no model workload active. Historical artifact hashes preserved
in reports/phase2_start.json. New environment artifact: phase2_environment.json.
Next decision: harden before collecting new model trajectories.

## HARNESS-015 — stream/resume/prompt contracts

Implemented explicit stream edge/error statuses with cleanup, valid EOS checks,
optional dependency recording, protected general resume, task/policy identities,
matched replicate seeds, clustered summaries, prompt-intervention comparison,
and a frozen v5 plus new policy-aware v6 scorer. No model context/position/RoPE
settings changed and no silent truncation is permitted. Unit/serialization tests
pass; full inference integration and the 144-trajectory prompt study are next.
No TRAIN targets or optimizer steps yet.


## EOS-016 — added-token vocabulary bounds

The first prompt-study launch stopped during model initialization, with zero
trajectories generated. New validation mistakenly bounded EOS by vocab_size248044,
which excludes33 added tokens; full tokenizer length248077 contains both EOS IDs.
Fixed the bound to len(tokenizer), retained fail-closed checks, and added a real
cached-tokenizer regression test. Failed-launch artifact is preserved separately.

### Phase 2 — device-free CI milestone

Added Linux GitHub Actions for Ruff and device-free tests, with explicit markers
for Apple-only and cached-tokenizer checks. Local verification: 50 passed, five
deselected; Ruff passed. This is local evidence, not a claim that remote CI ran.
MLX remains the Apple default; optional embedding dependencies are locked but
have not been installed during the active prompt study. The study is a single
serialized process and retains the native 262144-token context configuration.

### Phase 2 — diagnostic and structural-data infrastructure

Implemented staged-cap censor-aware summaries, exact prefix checks and labeled
forced-close probes. Added local MiniLM measurement code (not yet run), offline
exact-guard replay, conclusion categories, natural correctness-first candidate
ranking and a guarded masked-LoRA optimizer interface (not yet model validated).
Procedural-v2 produces 64 tasks per split with independently tested exact labels
and cross-split structural holdouts. Difficulty follows operation/AST structure,
not model outcomes. Historical v1 data and v5 report artifacts remain unchanged.
Predeclared TRAIN pilot: sixteen tasks, N=4, 4096 cap. No TRAIN generations or
optimizer steps have been executed at this milestone. Prompt study continues
under the exclusive model lease.

### Phase 2 — first matched qualitative calibration

Inspected the complete first algebra replicate under all four policies. P0 still
repeatedly rederives arithmetic without format wording. P1 both drifts arithmetically
and repeatedly plans boxed presentation. P2 solves, performs a substitution check
and terminates at 554 reasoning tokens. P3 recovers the solution then deliberates
at length about literal FINAL syntax until the cap. This is a single matched case,
not an aggregate prompt-effect estimate. Source-linked DEV annotations and 24
curated chunk-pair judgments are recorded with training permission false. They
will calibrate local lexical/semantic measurement; actual embedding inference is
pending until Qwen releases the exclusive lease.
