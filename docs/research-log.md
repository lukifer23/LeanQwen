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

### Phase 2 — documentation and historical-preservation checkpoint

Full Mac unit suite: 70 passed; Ruff passed. README now distinguishes implemented
interfaces from pending real validation and keeps first-pass findings explicitly
historical/censored. All report builders write new destinations; first-pass report
and split hashes remain unchanged. Four completed P3 replicate-zero controls
reproduce the historical emitted token IDs exactly. Remote CI is queued, so no
remote pass is claimed. Substantial SFT remains gated and TEST has no responses.

### Phase 2 — format-meta heuristic confound

Inspected four further complete P2 trajectories: ordering, modular state, logic
and word problem. They repeatedly question task semantics without prolonged
answer-format discussion. The generic v1 format-meta cues falsely capture this.
Added derived anchored v2 while preserving all v1 run fields and the original
policy-selection rule. State trace never reaches its expected 11; word trace
never reaches its expected discounted price; ordering never assembles the chain.
Logic trace corrects initially wrong XOR/NOT definitions but truncates before an
aggregate final count. Long continuation includes both necessary error recovery
and redundant reconsideration; it cannot all be labeled waste.

### Phase 2 — dataset-quality report completeness

Candidate summaries now include measured raw/selected length distributions,
correctness, exact response duplicate counts, family/difficulty distributions and
source/license distributions. No candidate measurements exist yet. Format-meta
calibration on the eight inspected DEV traces: v1 has three true positives, four
false positives, one true negative and zero false negatives. Anchored v2 separates
these eight examples, but was designed on them: this is in-sample evidence, not
independent validation. Full suite at this checkpoint: 74 passed; Ruff passed.

### Phase 2 — inspectable target approval interface

Added natural dataset assembly from separate individual TRAIN reviews, with
structural contamination checks, exact parent preservation, length exclusion
without truncation and a hashed quality manifest. Reviewer prose is excluded from
supervised responses. Optional TRAIN embedding analysis runs in its own sequential
process after Qwen generation; it remains diagnostic rather than an automatic
filter. These interfaces have not yet produced data or optimizer evidence.

### Phase 2 — calibration breadth and zero-prefix control

Expanded individually inspected redundancy pairs to 36 across five DEV tasks,
including ordering, state, logic and word-problem content. Useful distinct
truth-table rows and different variables are explicit negative examples. This is
curated, partly selected for lexical similarity, and not population calibration.
Before any early-exit generation, added a zero-prefix forced-close control to the
committed protocol. Without it, a correct 128-token probe might simply reflect
existing direct-answer ability. Greedy/no-penalty closure remains an intervention,
not ordinary model behavior. No source generations or historical artifacts change.

### Phase 2 — v2 stimulus wording audit before generation

Patched the unqueried v2 pool as generator version `procedural-v2.0.1`: counting
questions now say “whose sum is…” and runner numbers explicitly mean identifiers,
not ranks. This addresses a grammar defect and a known task-parsing ambiguity
before collecting TRAIN/DEV model responses. Labels, semantic IDs and structural
signatures are unchanged; pool hashes/provenance now record the patch. Earlier
v2 drafts remain in Git history. Historical v1 and every measured run are untouched.

### Phase 2 — model-instance lifecycle hardening

Future CLI/diagnostic/candidate invocations now close the MLX backend (synchronize,
release model/tokenizer references and allocator cache) before releasing the OS
lease. Training explicitly destroys its first instance before reload and closes
the reloaded instance before unlocking. The currently running prompt study uses
its original committed code; its process will be fully reaped before another
model is launched, so this change does not alter its generations or environment.

### Phase 2 — durable diagnostic recovery and CI evidence

Staged-tail jobs now use a dedicated task/cap run directory and recover an already
fsynced trajectory if interrupted before controller-cache publication. Contract,
model, seed and scoring mismatches, explicit generation failures and multiple
attempts are rejected. This avoids regenerating a completed trajectory on resume.
The local embedding instance is released before its lease is unlocked. The SFT
interface records actual adapter/base tensor dtypes when executed; no optimizer
smoke has run yet. Validation: 76 local unit tests and Ruff passed. GitHub Actions
run 37369683787 passed its weight-free Linux checks for commit 48eb9b7; subsequent
commits have their own CI status and are not covered by that result.

### Phase 2 — matched arithmetic trace review

Individually reviewed replicate-zero arithmetic under all four policies, extending
conclusion annotations to twelve source-linked DEV traces. Bare P0 recovers 2006
then returns to incorrect subtraction without finalizing. P1 reaches 2006 early,
reconsiders arithmetic and operand interpretations, and remains capped. P2 answers
2006 and reaches EOS after a shorter trace. P3 derives 2006 then spends most of its
remaining budget reinterpreting FINAL syntax. These are qualitative matched cases,
not the still-pending full-study aggregate or proof of wrong-final degradation.
Added measured-report plotting and documented degenerate boundary bootstrap
intervals. Semantic policy averages now explicitly exclude the uneven selected
higher-cap cohort. No new model process, TRAIN generation or optimizer steps.

### Phase 2 — completed prompt-policy checkpoint and requested stop

Experiment: `runs/20261005T193609-phase2-prompt-policy-8fdc19d6`, source Git
`a4cd3d92259a0a4f94d8ad135b3a34ae122c25de`. Completed 144 DEV trajectories:
twelve balanced historical tasks × three matched seeds × four policies, original
pinned weights, official thinking sampler, guard off, 2048 total-output cap,
scoring v6. P0/P1/P2/P3 correct counts: 3/0/7/3 out of 36 each. P2 provisional
selection reduces mean total output 6.5% versus P3, while median/p95 reasoning
remain capped at 2048 and 29/36 P2 trajectories truncate. Paired task-bootstrap
accuracy delta: +11.1 points, interval 0–22.2 points; no established general gain.
P2 has mixed seed correctness on five of twelve tasks. All twelve P3 replicate-zero
controls reproduce historical stock token IDs exactly.

Twelve individually reviewed DEV traces and 36 curated lexical pairs are now
source-linked. Lexical pair cosine @0.90: TP9/FP2/TN15/FN10, limited by curated
selection and one assistant reviewer. Anchored format-meta v2 remains diagnostic;
its initial eight-trace development calibration is preserved separately from the
expanded checkpoint. No semantic encoder inference or complete guard replay ran.
All thirteen natural EOS completions are correct; intermediate contradiction in
capped arithmetic does not establish wrong-final degradation.

Aggregate output throughput: 67.60 tokens/s including prefill, with uncontrolled
interactive timing. Peak recorded MLX allocation 1.684 GB; sampled process RSS
0.394 GB, overlapping accounting. Context remains native 262144 with unchanged
RoPE and no truncation; long-context behavioral retention is untested.

Stopped at the user’s requested good milestone. Worker 15712 and launcher 15596
exited normally and were reaped; no QwenLean model workload remains, OS lease
available, no follow-on job queued. All starting historical file hashes and six
run-record digests match. No new TRAIN responses, approved SFT dataset, optimizer
steps or adapter exist. TEST has no model evaluation. Validation: 76 local tests,
Ruff and relative document links pass; GitHub CI run 37371123362 passed for
fd4360e. Subsequent checkpoint publication has its own CI status.

Next decision: resume the seven-case predeclared P2 staged tail, then early-exit
and semantic/guard calibration. Candidate generation and real optimizer smoke
remain gated; substantial SFT is not ready to select or launch. Full measured
status, raw artifact map and exact next command: `reports/phase2_checkpoint.md`.
