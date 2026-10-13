# Research log

Chronological entries retain earlier decisions and measurements. Later scoring
corrections supersede older accuracy counts; current results are in
[the first-pass report](../reports/first-pass.md).

## ENV-001 — 2026-10-05

Hypothesis: native Mac inference can support the first controlled research loop.
Configuration: original Qwen/Qwen3.5-0.8B pinned weights, no quantization; isolated
Python 3.12; MLX 0.32.3/MLX-LM 0.32.0, Transformers 5.18.0/Torch 2.14.1 on MPS.
Dataset: one diagnostic arithmetic prompt, outside all split pools.
Result: both inference paths run. MLX 512-token thinking diagnostic 7.15 s;
MPS 64-token greedy thinking diagnostic 7.42 s. MLX trace reaches cap after
multiple recalculations; sampler omitted presence penalty, so not baseline A.
Interpretation: select MLX for evaluation; diagnostic confirms parse behavior.
Next decision: implement measured DEV baseline with full official parameters.

## COMPAT-002 — 2026-10-05

Hypothesis: adapter gradients can pass through Qwen3.5's hybrid MLX architecture.
Configuration: rank 4 adapters in last two blocks; fixed 32 token IDs; numerical
logit diagnostic; no optimizer steps.
Result: first attempt fails because snapshot request expected uncached metadata;
restricting the request to the same files as inference fixes that download issue.
Second attempt fails with CustomKernel VJP: model was in inference mode.
Root cause: the architecture selects non-differentiable fast kernels when not
training. Correct fix is `model.train()`, using the existing differentiable path.
Final result is in `reports/mlx_lora_probe.json`; preserve failed probe artifacts.
Interpretation: backward compatibility is distinct from a validated training run.
Next decision: real bounded SFT smoke only after dataset quality gates.

## EOS-003 — 2026-10-05

Hypothesis: a tokenizer/model EOS mismatch could masquerade as poor termination.
Configuration: inspect checkpoint text config and official tokenizer metadata.
Result: chat EOS is `<|im_end|>` (248046); text config EOS is `<|endoftext|>`
(248044). Added both to the stop set. Aborted the initial three-sample partial
run and retained it locally with an explicit aborted status. Restarted A.
Interpretation: subsequent cap failures cannot be attributed to omitting either
checkpoint-declared terminal token. Parser treats both as control tokens.
Next decision: complete the corrected run and predeclared sampler sweep.

## PROCESS-004 — 2026-10-05

User constraint: avoid duplicate, orphaned or concurrent model processes.
Result: process audit found exactly one active evaluator with a live supervisor;
all earlier probes exited. Added repository-wide nonblocking OS lock, PID and
process creation-time metadata, stale-owner recovery and lock regression test.
Registered the already-running evaluator; verified a second lease is rejected.
Next decision: model workloads remain strictly sequential, with final process audit.

## CONTROL-005 — planned before full baseline completion

Hypothesis: thinking-mode failure can differ from task-solving capability.
Configuration: official default non-thinking mode with its distinct official
text sampler (temperature 1, top-p 1, top-k 20, presence 2). Same DEV tasks/seeds
and total-output cap. This changes mode and recommended sampler together, so it
is a mode control, not a single-variable sampler ablation.
Result: pending; do not populate until measured.
Next decision: compare total output compute as well as reasoning-token partitions.

## BASELINE-A-006 — 2026-10-05

Hypothesis: the official thinking sampler provides a measurable accuracy/compute baseline.
Configuration: original bf16 text weights, temperature 1, top-p .95, top-k 20,
presence 1.5, repetition 1, thinking enabled, both checkpoint EOS IDs, 2048 total
output cap, no guard. Run `20261005T170152-A-official-thinking-0d62e9fa`.
Dataset: procedural-v1 DEV, 60 problems across six families, one stable seed each.
Result: 0/60 final accuracy; all capped; mean reasoning 2044.13, p50/p95 2048;
mean latency 33.90 s; aggregate 60.42 output tokens/s; MLX peak 1.684 GB, sampled
RSS peak 2.140 GB. Exact-cycle rate 0%; mean repeated-content density 7.76%.
Reanalysis fixes sentence punctuation matching, preserving original samples and
all generation IDs; final accuracy remains unchanged. Candidate correct-conclusion
cues in 40/60 are heuristic, not verified solution states.
Interpretation: termination is severely impaired at this cap and with these
format-constrained prompts. Do not infer uncapped or general reasoning accuracy.
Manual inspection includes correct arithmetic followed by formatting reconsideration.
Next decision: default-mode control and bounded single-variable sampler sweep.

## PARITY-007 — 2026-10-05

Hypothesis: severe baseline behavior could reflect an MLX architecture implementation error.
Configuration: same original bf16 checkpoint and diagnostic prompt; greedy, no
penalties, 64-token cap. Compare sequential MLX and prior MPS generations.
Result: first 53 tokens agree. Divergence is newline versus double newline.
Teacher-forced cached logits at this context have cosine .9999007, RMSE .03784,
and 19/20 top-token overlap. MPS logits tie at 22.5; MLX double-newline logit 22.625.
Interpretation: divergence is consistent with bf16 numerical differences near a
tie; this diagnostic supports architecture fidelity but is not universal backend
equivalence. Pins/seeds reproduce within a backend, not across devices.
Next decision: continue MLX experiments and retain explicit backend identifiers.

## CONTROL-005 — completed

Configuration: `configs/nonthinking.yaml`, same 60 DEV tasks and stable per-problem
seeds, original weights, official non-thinking sampler, 2048 total-output cap.
Run: `20261005T173939-A0-official-nonthinking-723496fd`.
Result: 15/60 correct (25%; Wilson 95% interval 15.78–37.23%); median total output
381.5 tokens, mean 815.13, p95 2048; 17/60 cap failures; mean latency 12.39 s;
aggregate throughput 65.76 tokens/s. Median explicit thinking partition is zero,
but final-channel text includes calculations and sometimes very long reconsideration.
Task accuracy: algebra 50%, arithmetic 30%, boolean 30%, state 20%, word problems
20%, ordering 0%. Exact all-output cycle heuristic detects none.
Interpretation: mode changes bounded accuracy and output compute substantially,
while many very short answers are wrong. This is a control, not an optimized
reasoning-preserving model; no claim about uncapped capability or TEST performance.
Next decision: finish the predeclared thinking sampler pilots before dataset selection.

## SCORE-008 — correctness versus format, manual calibration

Hypothesis: strict FINAL extraction can conflate task accuracy and format compliance.
Evidence: a completed repetition-penalty response correctly concludes "Total True
Expressions: 3" but leaves `FINAL:` empty. Another completes with `ANSWER: 2006`.
Decision: separate task correctness, strict extraction correctness and numeric
FINAL compliance. Accept only terminal explicit conclusion cues, independent of
label matching. Reject values followed by continued reconsideration. Test all
observed edge cases. Preserve original metrics and versioned scoring backups.
Result: baseline A remains 0/60. A0 is 19/60 task-correct, 15/60 strict-correct.
Manual review rejects a capped intermediate-correct total and accepts a terminal
state answer after earlier incorrect computations. All four A0 recoveries are EOS.
Pilot task accuracy: cooler 1/12, presence-zero 0/12, repetition-1.05 2/12; the
last two successes are format-noncompliant but have correct terminal answers.
Interrupted the controller after 11/12 repetition samples to fix scoring;
resumed only the final pending sample, with identical config/hash/ordered dataset.
No model generation was fabricated or replaced. Candidate selection remains
repetition-1.05 under the corrected definition; full DEV confirmation is running.
Interpretation: report format separately from reasoning ability, and inspect
intermediate reasoning quality separately from final-answer accuracy.
Next decision: rescore full confirmation under the same v3 scorer before comparison.


## SCORE-009 — terminal numeric markers

Hypothesis: marker extraction can mistake quoted formatting examples for an answer.
Evidence: `dev-algebra-45a5b6efe1bc` in the full repetition candidate quotes
`FINAL: 43` while continuing format deliberation and reaches the cap.
Decision: v4 requires terminal numeric FINAL/boxed/plain answers or terminal
explicit conclusion cues. Preserve legacy extraction as `strict_final_correct`
for audit, independently of format compliance. Versioned backups retain all
previous measurements. No regenerated or modified model output.
Result: completed A/A0 and pilot task counts remain 0/60, 19/60, and 1/12,
0/12, 2/12. The observed quoted example is rejected. Thirty tests pass.
Next decision: apply the same v4 contract to full B before final comparison.


## SCORE-010 — calibrated terminal numeric forms

Hypothesis: a conservative marker/cue grammar can miss clear completed solutions.
Evidence: A0 outputs terminal standalone numbers, x=43, total count=2, and a
count-of-3 conclusion; B correctly calculates 2871 in its final channel before
an empty FINAL marker. None should be mistaken for answer degradation.
Decision: v5 accepts terminal standalone numeric lines, numeric equations,
explicit count conclusions, and terminal equations followed only by a format
notice/empty marker. Still reject quoted examples or continued reconsideration.
All extraction is label-independent; correctness compares the extracted result
with the algorithmic label. Thirty-one regression tests pass. Preserve v1–v4
metrics and original emitted IDs/raw outputs.
Result: uniformly rescored A0=24/60, A=0/60, B=4/60. Pilot counts remain 1/12,
0/12, 2/12, so candidate selection is unchanged. No wrong EOS final result in B;
the earlier heuristic candidate was a correct mathematical solution with a
missing numeric FINAL field. No causal overthinking-to-wrong-answer rate claimed.
Next decision: freeze this documented grammar for the next prompt-control experiment.

## SAMPLING-011 — completed first decoding confirmation

Hypothesis: single-variable sampling changes can improve bounded thinking behavior.
Configuration: three balanced 12-example DEV pilots: temperature .6, presence 0,
repetition 1.05, all other A settings fixed. Promote repetition 1.05 to all 60 DEV.
Dataset: procedural-v1 DEV; no TEST/model evaluation or TRAIN trajectory generation.
Result: pilots 1/12, 0/12, 2/12; full B 4/60, 56/60 capped, median/p95 reasoning
2048. A 0/60, 60/60 capped. Paired accuracy +6.7 percentage points, bootstrap 95%
interval +1.7 to +13.3 points. Exact paired McNemar two-sided p=0.125:
this small gain is not statistically established at 0.05. B mean total tokens 2007.33 versus A 2048: 2.0%
reduction. Mean repeated-content density A 7.76%, B 5.67%. All-output exact-cycle
rate 0% under the documented narrow heuristic. Twelve of twelve repeated pilot
token sequences are identical. B 62.08 output tokens/s; mean latency 32.34 seconds.
MLX allocator peak 1.684 GB; sampled B RSS 0.515 GB (distinct accounting measures).
Interpretation: small bounded accuracy gain, unchanged censored reasoning tails;
no established material efficiency improvement. Default-mode control 40% is far
better on this small suite but changes mode and sampler together. Format
reconsideration is a prominent manually observed confound; a learned-behavior
versus prompt/decoder attribution is not yet resolved.
Next decision: matched-seed DEV prompt-format control before training, then a
small higher-cap diagnostic if needed. First dataset should be Qwen TRAIN best-of-N
only after semantic split fixes and reasoning-quality gates.

## PROCESS-012 — first-pass cleanup

Result: full sampler controller exited normally; process audit found no registered
QwenLean model workloads and an inactive lease with a dead former owner. Exclusive
lock acquisition/release succeeds. No unrelated application was terminated.
See reports/process_audit.json. All model experiments were serialized; offline
scoring/report jobs are separate from model inference.


## FOUNDATION-013 — validation milestone

Result: 32 tests pass; Ruff and whitespace checks pass. Machine-generated reports
use complete preserved records under the calibrated v5 scoring contract. Docs
state implemented commands and measured limits; no adapter, CUDA backend, semantic
embedding metric, quality-approved training corpus or efficiency breakthrough is
claimed. GitHub About/description/homepage and eight research topics are configured.
Milestone tag: v0.1.0, first-pass research foundation. Main-only development.
Next decision: the DEV prompt-format control described in reports/first-pass.md.

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

### Phase 3 — repository health and resumed authorization (2026-10-12)

Local and fetched remote main matched `3d32475d660a9f2ac4c8d157f7794bb0c35392d6`
at start; clean checkout. All 76 starting local tests and Ruff passed. GitHub run
37372433674 had no executed steps: its failure annotation says the job was not
acquired by a hosted runner after multiple attempts. Retried that run and pinned
the next workflow to Ubuntu 24.04/Python 3.12 with a bounded job timeout. Fresh
remote CI must be observed before claiming recovery. Raw failure classification:
`reports/phase3_ci_initial_failure.json`.

The user authorizes reversible mac-128k/mac-100k/native-262k profiles, sequential
diagnostics, reviewed solver targets if natural coverage is poor, the optimizer
smoke, and one conditional real SFT pilot. This supersedes the prior requested
stop and previous natural-target-only/128K-floor restrictions for this phase.
Historical findings and protocols remain preserved.

The user confirmed PonderScope PID 23131 is the competing Qwen3.5-4B GPU workload.
It is still active during CPU-only preparation. No LeanQwen model experiment has
started or unrelated process been terminated. macOS observed swap use was
4825.50 MiB during that competing workload; this is not a LeanQwen measurement.

### Phase 3 — reversible context implementation and CPU restoration audit

Implemented mac-128k/mac-100k/native-262k runtime admission profiles with native
checkpoint metadata preserved. Generations now record actual input, reserved
output, emitted/combined tokens and context fingerprint; prefill chunk selection
uses the inspected MLX-LM API. No lossy cache or quantization is introduced.
CPU SHA256 checks of actual pinned checkpoint, config, tokenizer and template
files before and after profile selection all match. This verifies configuration
reversibility, not 262K inference or memory savings. No adapter exists yet.
Initial offline audit resolution failed because unrelated cached model-card/license
files were absent; the exact runtime file patterns fixed it, with the failure
retained. See `phase3_context_configuration.json` and
`phase3_metadata_initial_failure.json`.

The evaluation-only document suite constructs exact 2K/8K/32K combined envelopes,
with 256 reserved output tokens and five retrieval/reasoning questions per length.
Offline official-tokenizer tests verify construction, labels and fact positions.
No model context benchmark has run while PonderScope is active. Restoration,
training/inference budget distinctions, backend/cache limits and real commands
are documented in `docs/context-management.md`. Historical v1/v5 and all 144
prompt-study records remain unchanged. Current local tests: 92 passed; Ruff passed
including pending training-integrity changes. The retried original CI and fresh
CI for `91a0bb9` both passed.

### Phase 3 — adapter/base integrity audit before GPU availability

Replaced selected adapted-base-matrix hashing with name/shape/dtype/raw-byte hashes
of every current frozen parameter, including unadapted blocks, embeddings and
norms. Empty/duplicate verification sets fail. Hashing uses bounded host chunks
and re-enumerates parameters after training, so replacing arrays cannot evade an
old-reference check. Fresh reload repeats both adapter and base verification.
Exact intended full-attention q/v and linear-attention qkv tensor names are checked;
all-zero gradients, missing/early EOS, unclosed or incorrect targets, and failed
reload generation prevent success. Added the explicit four-step smoke config.

CPU tests use real tiny MLX Linear/LoRA modules to demonstrate that adapter changes
leave the frozen-base fingerprint unchanged and an unadapted-weight replacement
changes it. These are unit contracts, not Qwen optimizer evidence. After the user's
strict no-GPU steering, MLX unit tests were explicitly forced to the CPU. Latest
validation: 92 non-MLX plus six CPU MLX tests passed; Ruff passed. PonderScope still
active, zero new LeanQwen model runs/optimizer steps. Context milestone `81db1ae`
is pushed on main and its GitHub run 38284065338 passed.

### Phase 3 — bounded prefill/cache measurement readiness

Inspected installed `KVCache` (256-token allocation growth) and `ArraysCache`
state/byte accounting. The wrapper now explicitly owns the same fresh ordinary
caches that MLX-LM would construct and records per-chunk prefill and sampled decode
allocation/RSS, attention-cache bytes and linear-state bytes. No state is evicted
or reset during a request. Context probes apply their declared available-memory
floor at those observation boundaries; violations become durable explicit failures.
Cache state is retained through the final snapshot and released on return, so
end-state bytes and unused allocator cache are distinguished in the docs. Profile
validation runs before loading weights. No resource savings, prefix reproducibility
or long-context behavior is claimed before native execution. PonderScope remains
active; all verification continues on CPU only (98 tests, Ruff passed).

### Phase 3 — saved-trajectory exact-guard replay (CPU only)

Reused existing replay logic on all 144 archived prompt trajectories, at all three
predeclared settings and the live 16-token polling cadence. Zero triggers and zero
observed savings at every threshold. Precision/recall remain unidentified; these
trajectories have no independent exact-cycle labels, and semantic rederivation
can evade literal matching. Guard stays off. No source trajectory or historical
result was changed, no model was loaded, and this partial replay cannot satisfy
the full diagnostic gate before tail/semantic evidence. Source hash matches the
completed prompt-study checkpoint. New source-linked outputs live in
`reports/phase3-prompt-replay/`.

Reproduce into a new protected destination:
`uv run --frozen --no-sync python scripts/calibrate_metrics.py --guard-only --prompt-only --output-dir reports/<new-prompt-replay>`.
The full default command still requires tail records and runs semantic analysis
under the existing process lease. PonderScope is active and untouched.

### Phase 3 — stock context and prefill evidence after natural GPU handoff

PonderScope exited naturally with EVIDENCE_COMPLETE before LeanQwen model work;
no external process was signaled. Stock `context_document_v1` completed 25 total
question/length/chunk observations before the later audit addendum arrived.
Original bf16 weights, greedy non-thinking, mac-128k, full history and 256 reserved
output tokens. At 2K/8K/32K combined envelopes (1792/7936/32512 actual input):
3/5, 4/5 and 4/5 correct, with all three positional retrievals correct at each
length. The inventory-reasoning task failed in every condition, including a
wrong naturally terminated 32K answer. These fixed-fact/repeated-distractor
questions do not establish general long-context reasoning capability.

Measured 512-token prefill chunks reduced peak MLX allocation at 8K from 3.016
to 2.473 GB (18.0%) and at 32K from 5.016 to 2.960 GB (41.0%). Accuracy unchanged;
four of five 8K outputs and all five 32K outputs match exactly. Mean 32K prefill:
16.274 versus 16.381 seconds. KV capacity: 24/96/384 MiB; linear state 18.6 MiB.
Measurements are preliminary sequential interactive observations, without thermal
controls. Savings come from chunk size, not a reduced operating ceiling.
Short live 128K/262K switching preserved all 320 loaded tensors (752393024
elements), model args and emitted IDs. Full 100K/128K/262K remains unverified.
Evidence: `phase3-context-stock.md`, `phase3_context_stock.json`,
`phase3_context_stock_records.jsonl`, `phase3_live_profile_restoration.json`.

### Phase 3 — historical log restoration after independent audit

The independent addendum identified a real documentation regression: the log
began with duplicated metric definitions and had lost the Phase 1 experiment
entries. Restored the complete original log from Git commit
`8f67d3b2a1c217f5cad819ac2b8211b24db57e7d`, then appended the existing Phase 2/3
entries unchanged. ENV-001 through FOUNDATION-013, both planned/completed
CONTROL-005 entries, original scoring corrections, failure narratives and measured
results remain verbatim. Formal definitions remain in `docs/metrics.md`.
Source hashes, entry inventory and preservation/link checks:
`reports/phase3_research_log_restoration.json`. No measurement was reconstructed
from memory and no historical result was rewritten.

At addendum arrival, local and remote main were both `e5b5b49`; the working tree
contained completed context/restoration evidence and its pending documentation.
PonderScope had exited naturally with EVIDENCE_COMPLETE. The 2K/8K/32K suites,
512-chunk comparison and short 128K/262K invariance check had already completed.
The staged tail had begun before the addendum: algebra completed its 2K/4K/8K
jobs, reaching correct EOS after 6122 reasoning tokens at the 8K cap. This is a
selected case, not a completed seven-case study. The next job was interrupted
with SIGINT sent only to the verified own LeanQwen controller to hold escalation
for the missing small-Metal/cache/three-profile/historical-prefix preflights.
Completed job archives are retained; the interrupted job has zero durable rows
and an explicit KeyboardInterrupt completion record. Full partial emitted tokens
were not captured for that interrupted job. Hold evidence:
`reports/phase3_tail_preflight_hold.json`. PonderScope was never signaled.

### Phase 3 — actual Metal/backend/historical compatibility gates

After the audit hold, tested small real Metal bf16 tensors before loading Qwen
for the new compatibility preflight. Stable hashes across repeats/chunk sizes,
value-sensitive hashes, safe GPU/CPU transfer, and host copies bounded to six
bytes covering all 128 input bytes passed. No full Qwen model was loaded for the
primitive test. `phase3_metal_integrity_preflight.json` retains actual evidence.

Short loaded-model checks passed correct EOS and exact one-token cap behavior,
six full-attention KV/eighteen linear-state caches, nonnegative consistent byte
counts, progress callbacks, prefill/decode timing and request cleanup. All three
profiles produce identical short IDs without changing any loaded tensor, model
argument or tokenizer/template metadata. First 128 tokens match two saved P2
replicate-zero arithmetic/algebra trajectories exactly under their original
prompt IDs, seeds and sampling. After adding explicit seed/config/hash fields to
all backend generations, repeated the bounded checks into
`phase3_backend_preflight_attributed.json`, preserving the initial artifact.

Available RAM before that load: 7.10 GiB; short-generation MLX peak 1.605–1.609 GB;
cleanup active allocation approximately 1.505 GB, within 22 bytes of the baseline.
Frozen-base integrity now compares the complete canonical inventory with the
original loaded parameters before attachment, and verifies trainables belong to
the unique full set. Unit contracts pass; actual optimizer steps remain pending.

User requested all current evidence/docs be saved remotely before the next long
run. Full Mac suite: 98 passed; Ruff/whitespace checks pass. Pre-run checkpoint:
`reports/phase3-preflight-checkpoint.md`. The tail remains held until this update
is pushed, CI is observed and main is clean. No TRAIN generations, solver targets,
optimizer steps, adapter or model improvement are claimed. Next: resume the
existing seven-case tail, then early exit, semantic calibration and 16×4 TRAIN.

### Phase 3 — completed staged reasoning diagnosis

Resumed the preserved controller after commit `8156239`, passing CI and the clean
pre-run checkpoint. Seven predeclared DEV cases completed without prefix drift:
13/13 exact comparisons. Natural correct completions: arithmetic 498 reasoning
tokens, algebra 6122. Four cases remain censored at 8192; state machine remains
censored at 16384. No inference about their unobserved larger-budget outcomes.
Repeated correct derivations, renewed interpretation doubts, formatting fixation
and intermediate drift are visible in the saved trajectories. No TEST used.
Evidence: `reports/termination_tail_records.jsonl` and censor-aware measurements.

All 34 six-source forced-close probes completed, including six zero-prefix
controls. Three correct EOS answers across two tasks, one zero-prefix success;
31 outputs capped at the predeclared 128-token final allowance, often while
writing another explanation. This budget confounds mathematical insufficiency
with incomplete finalization. Prefix success is not monotonic; this is not a
validated stopping policy. Full outputs and seeds/configs are retained in
`reports/early_exit_records.jsonl`.

Installed the locked semantic extra only between Qwen workloads. CPU float32,
one-thread pinned MiniLM calibration: 36 reviewed pairs across five DEV tasks;
@0.90 TP11/FP1/TN16/FN8, precision91.7%, recall57.9%. Curated single-reviewer
annotations lack held-out calibration and cannot certify useless reasoning.
All 161 unique prompt/tail trajectories were measured without re-generating the
144 prompt-study responses. Full exact-cycle replay at the same three thresholds
again yielded zero triggers and zero savings. Guard remains disabled.

`reports/phase3-reasoning-diagnosis.md` connects observations to the original
post-training objective. No TRAIN generations, solver targets or optimizer steps
have yet run. Next: the predeclared sixteen TRAIN tasks × four natural trajectories,
then complete-target review and conditional optimizer/SFT gates.

### Phase 3 — active natural pilot and user decision checkpoint

After diagnostic commit `13b958a` and passing CI run 38292517555, launched the
predeclared sixteen TRAIN tasks × four natural Qwen generations, P2, cap4096,
original pinned bf16 weights, guard off, one leased process. Partial snapshot:
53/64 saved; 4 correct/EOS; 4 automatically eligible responses across
1 distinct task(s). This is incomplete evidence, not final coverage or a reviewed
training pool. Snapshot/hash: `reports/phase3_natural_pilot_snapshot.json`;
raw outputs remain durable in `runs/20261012T184117-phase2-train-natural-n4-1af34da1`.

The user now explicitly requests generation finish and a decision before
training. Optimizer smoke and real SFT are held; the current controller only
publishes candidate reports and exits. No optimizer job is queued. This checkpoint
supersedes earlier automatic conditional training progression. Do not implement
the conditional solver path before final natural coverage is assessed and
continuation is chosen.

Documented new discussion in `docs/training-data-strategy.md`: exact solver
reasoning with checked intermediates; reviewed licensed public worked examples;
coding/review assistants separated from response targets; closed-model teacher
permission/provenance restrictions; diverse complete examples and structural
splits; conservative removable adapters and measured held-out capability gates.
These are proposed interventions, not generated targets or demonstrated retention.
No external dataset or proprietary teacher response has entered training.

### Phase 3 — completed natural pilot, negative coverage and idle handoff

The preserved controller completed 64/64 generations across sixteen TRAIN tasks
and exited naturally. Four correct EOS responses, all for `2 + (-3) = -1`;
sixty capped/unclosed traces. Automatic target coverage is one distinct task
(6.25%), insufficient for a meaningful SFT improvement pilot. Successful reasoning
lengths: 302/1605/417/440; complete sequence lengths: 335/1638/450/473.
Mean reasoning3883.19; median/p90/p95 censored4096. No generation errors.
Sum of request latencies58.66min, aggregate70.62 output tokens/s; peak recorded
MLX1.772GB/RSS2.155GB overlap, minimum sampled available RAM4.534GB.
Original raw/public generation fields match exactly. No scores were rewritten.

Reviewed all64 statistical records, all16 task prompts/labels, all4 natural
responses in full, representative replicate-zero opening/ending passages for
the fifteen capped tasks, and flagged-claim contexts. Recomputed labels agree16/16
in an assistant mathematical review; not blinded or independent human validation.
Patterns include semantic re-derivation, instruction doubt, formatting fixation,
parsing confusion and arithmetic drift. Thirty-nine capped traces contain heuristic
correct-number cues; these are not completed solutions. Natural tierA remains
automatic eligibility, not a certificate of productive intermediate reasoning.

Found context limitations in the narrow equation checker: parity equations in
a modulo2 passage are flagged as ordinary integer arithmetic; a cap-cut RHS is
treated as a complete claim; rejected chained equations are not parsed. Do not
count these flags as a verified error rate. No checker change occurred in this
review, and these flags do not change complete-target coverage. Potential prompt
confounds (unused operation glossary, bracketed fees, adjacent signs, final-line
wording) remain hypotheses; evaluated v2 stimuli must stay unchanged.

Evidence: `reports/phase3-natural-pilot-review.md`, source-bound CPU audit/summary,
`reports/train_candidate_records.jsonl` and original quality report. Review prose
and calculations are measurement-only, not training targets. TRAIN semantic model
analysis was not run. No approved corpus, solver path, optimizer step or adapter.

User requested this documentation/remote milestone and then a stop for their and
another agent's review. Verified no project model workers and successful acquisition/
release of the OS lease (`reports/phase3_idle_checkpoint.json`). No processes were
terminated during this review. All new model work, data-source implementation and
training remain held until the user's next instructions. No automatic continuation.
Local validation for this documentation/evidence checkpoint: 98 tests pass,
Ruff and whitespace checks pass, ten current Markdown files have no missing
local link targets. No new model experiment was launched for verification.

### Phase 4 — starting audit and independently verified corpus

User explicitly authorized continuation beyond the October 12 review hold. Clean
`main` started at `160f7e0065249af41520c485a6e69ffc063070b8`; 98 starting tests
and Ruff passed. Historical artifact SHA256 values are frozen in
`reports/phase4_start.json`. No historical runs are overwritten. An unrelated
LM Studio model was initially loading, so Qwen work was deferred; it subsequently
unloaded. No unrelated process was terminated.

Implemented `solver-reasoning-v1` with eight operation-producing algorithms,
independent exact intermediate-state checks, structural deduplication, ID/OOD
DEV holdouts and a sealed TEST generator namespace. A 256-problem pilot passed
CPU verification before expansion to 512 problems / 512 distinct normalized
TRAIN structures. Exactly 64 per family; difficulty counts 38 trivial, 124 easy,
190 medium, 160 harder. Whole-example fit: 480 at 1024, 508 at 2048, 512 at 4096.
No truncation, external dataset, teacher trace or review prose.

The separate solver approval path reproduces its entire source archive from the
seed/recipe, checks every proof and preserves exact pinned official thinking
prefill/response/EOS tokens. Natural-parent contracts stay intact. Intermediate
verification uses parsed Fraction arithmetic, Boolean truth tables, unreduced
modular simulation, sequence invariants, transitive closure ranks, affine symbolic
substitution, exact-cent accounting and exhaustive combinatorial enumeration.
CPU property tests vary seeds and corrupt every intermediate after-state.
Initial cents/tokenizer mistakes and a lease-refused launch are retained in
`reports/phase4-development-failures.json`. Metal integrity and short backend
preflights passed. A 32-task, one-seed, 2048-output-token paired thinking/nonthinking
DEV protocol is frozen before training candidate selection; no TEST inference.
Optimizer and improvement results remain pending at this milestone.

### Phase 4 — approved smoke pool and stock baseline memory stop

The 512-task corpus and source-aware approval milestone was committed/pushed as
`135f5d8`; local 109 tests and Ruff passed, GitHub CI 38306454283 succeeded.
Eight solver targets (one/family, full lengths 247–268) independently passed archive
reproduction, proof, exact template/token/EOS and source approval. Parent archive
reproduction now runs inside the training lease; the pool-approval CLI also takes
the mandatory lease.

The frozen stock DEV baseline attempted its first non-thinking request and stopped
at prefill with zero emitted tokens: available RAM 3,323,740,160 bytes versus the
configured 4 GiB floor, MLX active/peak 1,504,791,388, observed RSS 943,620,096.
Explicit stream_error is saved in its raw sample; it is not a task-solving outcome.
Qwen was released; no unrelated application was terminated. User was asked to
free background memory. No floor was weakened and no model escalation followed.

No optimizer, adapter, capacity measurement, improvement pilot or before/after
capability comparison exists. Full status/length distributions/stopping evidence:
`reports/phase4-verified-reasoning.md`. Historical artifact hashes still agree.
This is an incomplete intervention experiment blocked at resource admission,
not a measured negative SFT result. Retry the fixed baseline in a new destination
when memory is available, then run actual optimizer smoke and capacity gates.

### Phase 4 — stricter structural/response deduplication correction

A final independent CPU normalization audit found word signatures retained discount
values: 64 initial v1 tasks mapped to 52 number-free word structures. Ordering
transitive-edge variants mapped to 50 size/query response schedules among64 tasks.
Maximum initial multiplicity3; mathematical proof verification still passed.
Retained the entire v1 corpus/manifests and published the derived audit in
`reports/phase4-structural-audit-correction.json`; no optimizer used the initial data.

`solver-reasoning-v1.1` / generator1.0.1 excludes numerical rates from word shapes
and deduplicates ordering responses as well as constraint structure. Regeneration
produced512 distinct problems,512 normalized task shapes and512 response shapes,
64/family. Difficulty45 trivial/120 easy/206 medium/141 harder. Reasoning min9,
median210, mean315.0977, max2134; whole-example fit480 at1024,511 at2048,512 at4096.
Fresh v1.1 DEV protocol frozen before training; newly approved smoke lengths
263/259/263/250/236/254/262/273. Older approvals intentionally fail the current
source-version/hash gate. The initial zero-token memory failure is preserved.
No new model job, optimizer, adapter or improvement evaluation followed the stop.

### Phase 4 — v1.1 baseline retry, resource stop and engine constraint

After background memory recovered to6,257,328,128 bytes, retried the corrected
frozen cohort in a new directory under the unchanged4 GiB floor. One non-thinking
arithmetic request completed incorrectly at15 output tokens. The next stopped
at448 output tokens, available RAM4,230,922,240 bytes (3.940 GiB); raw outputs
were preserved. No matched thinking cohort, optimizer, adapter or SFT followed.
This is an incomplete baseline, not measured cohort accuracy or an SFT failure.
Corrected data milestone07a2e21 passed GitHub CI38307356441.

User reiterated that LM Studio/LFM is forbidden. Earlier lms status commands
were an operational mistake; one woke its service. No LFM inference or target
generation ran for LeanQwen, and no LM Studio/Qwen output entered the new TRAIN
pool. Added an explicit rule forbidding even LM Studio status commands and new
Qwen self-distillation. Direct Python/MLX runs benchmark/train Qwen as the student;
all512 target proofs originate in independently verified deterministic solvers.

### Phase 4 — external teacher correction and human-source pilot

The critical user addendum supersedes natural/self-distillation and solver-primary
strategies. Starting main SHA0a62326, clean before work. Preserved historical raw
Qwen results, solvers, corpora and reports; current SFT rejects all their targets.
No LM Studio/LFM or Qwen teacher/rewrite/judge is allowed. Coding-agent solutions
cannot be relabeled algorithmic or human. Permission validation precedes training
record/tensor loading at all SFT entrypoints, with exact source/template checks.

Investigated original GSM8K, OpenR1-Math-220k, OpenMathInstruct-2 and DeepMath-103K.
Actual bounded sampling:64 problems per large source,7,554,902 response bytes;
measured full original GSM8K TRAIN. Preserved raw pages, immutable card pins,
permission snapshots and hashes. Large viewer samples are diagnostic only, with
no immutable row-revision attestation. OpenR1/DeepMath underlying source rights
remain pending; OpenMathInstruct upstream Llama output-trained-model obligations
prevent automatic enabling. No paid calls. Current DeepSeek API models are V4,
so do not assume an attested R1 provider endpoint from the R1 model license.

User directed selection and execution. Chose768 intact original human GSM8K TRAIN
solutions under MIT, plus64 reserved DEV questions. Approval uses documented
published human worker-agreement QA and exact screening, not invented fresh
individual review or formal proof of prose semantics. Human-only source100%;
Qwen/LFM/solver-authored/paid generation0%. Annotated operation bins256 each;
reasoning min33/median124/max346, full sequences97–499; no cropping. All annotations
recognized and consistent, final-label linked; unrecognized/unlinked/flagged rows
excluded. Limited contamination checks passed, with paraphrase/all-benchmark and
pretraining exposure explicitly outside scope. Eight human targets approved for
technical smoke. Original candidate manifest remains pending as import evidence;
the new separately hashed approved manifest records the actual approval basis.

Frozen56-task DEV protocol:24 source questions across three operation-count bins
plus existing32 procedural retention tasks, identical stock/adapter tasks, both
modes,2048 output cap,one matched seed,guard off. Relative to GSM8K, procedural
cohorts test retention/domain shift; old solver-ID/OOD names do not define current
training generalization. No model outcomes used for cohort/target selection.
First CPU freeze failed because source tasks lacked prompt-renderer facts; fixed
with source identity/annotation-count metadata, then froze successfully. Raw source
and historical cohorts were not modified. Audit records the failure and hashes.

No optimizer step or adapter exists at this corpus milestone. The four-step MLX
technical smoke is configured, then integrity/reload, actual capacity and complete
stock baseline remain prerequisites for a bounded improvement pilot. No reduction
in overthinking or capability retention can yet be claimed. See external source
comparison, corpus audit and training-data strategy for reproducible commands.

Corpus milestone validation:140 local tests pass;124 device-free CI-equivalent
tests pass (16 device/tokenizer tests deselected); Ruff and diff whitespace checks
pass. Historical tracked reports/split files match starting Git objects exactly;
counts and hashes are saved in external-source-start-audit.json. Raw datasets,
permission archives and weights remain ignored. Remote CI is verified after push.

### Phase 4 — external human technical smoke resource stop

Corpus/code/docs milestone df51885 pushed to main; GitHub CI38313003963 succeeded.
Attempted the actual four-step MLX smoke on eight intact approved human TRAIN
solutions under the unchanged4 GiB floor and512 MiB swap-out limit. No competing
student process; model lease acquired. First backward evaluation completed, then
available RAM2,773,237,760 bytes hit the floor before optimizer update1. MLX
active1,520,987,980 / peak2,896,505,106 bytes; RSS2,076,753,920; swap-outdelta4,161,536.
Loss and gradient finiteness checks follow the memory gate and were not reached;
no passing finite-loss claim, completed step time or adapter exists. Original-base
and attached frozen-base fingerprints match before optimization. Post-training
integrity/reload/fresh inference and capacity checks were not reached.

Saved failure completion, raw console, resource snapshot, exact approved-source
manifest and base fingerprints in ignored adapter directory; compact measured
report tracks their hashes. Resource-stop logging now preserves allocation, RSS,
swap and completed-update count. Qwen exited and released the lease; no unrelated
application was killed and no floor was relaxed. Stop model work at the resource
gate. Same bounded smoke is the next technical action once memory permits; no
external review/teacher call is needed for this retry. No SFT efficacy conclusion.

### Local training memory fixes

Starting main65c03dd, clean. Replaced the full-sequence vocabulary projection with
complete response-only hidden-state projection and checkpointed32-token head/loss
chunks. Same float32 full-vocabulary token-level objective, prompt mask, first
response/EOS supervision and frozen head; no target truncation or base changes.
Twelve small CPU MLX equivalence/contract tests pass, including actual tiny Qwen
mixed-attention LoRA gradients and tied/untied heads. Initial checkpoint test found
integer-label VJP attempted; fixed with stop_gradient on labels within the chunk.
This was a unit-test failure, not a model training run or teacher example.

Bound reusable MLX cache64 MiB and clear unused buffers before RAM admission checks;
keep4 GiB RAM/512 MiB swap-out gates unchanged. Active example, chunk size, live/cache
allocation and elapsed-stop time are now saved. Full suite152 passed before four
additional invalid-config cases were added. New measured retry follows separately.

### First real optimizer update and further bounded-prefix fix

Memory-fix commit0c219d5 pushed; GitHub CI for the exact0c219d5 commit succeeded. Retried eight approved human targets with32-token
checkpointed head chunks and64 MiB reusable cache. One real AdamW update executed:
loss1.066536665, finite gradients,1.215982 seconds,171 supervised tokens,281 complete
sequence tokens, MLX peak2,810,513,840 bytes; sampled RAM4,533,387,264 bytes. Second
backward (146 complete tokens) stopped before update2 at4,033,495,040 bytes available,
below the unchanged4 GiB floor. The failed run has no saved adapter or full integrity/
reload approval. Loss/source/raw hashes are preserved in the separate attempt1 report.
No efficiency or capability claim follows this optimizer-only evidence.

Added bounded execution of the frozen prefix: reproduce the pinned complete-input
Qwen core with unchanged training-mode recurrence, masks, layers and final norm;
materialize and detach only layers before the first trainable block. Embeddings
must be frozen; multi-device pipelines are rejected. No mathematical shortening,
inference-kernel substitution or architecture/weight changes. CPU tests compare
loss and every LoRA gradient against the original full decoder with two frozen and
two adapted layers, tied/untied heads. Per-step adapter hashes and failed-run tiny
adapter preservation are added for exact update evidence. New bounded-prefix
config retains all original resource gates and exact human targets.

### GPU pressure incident and independent CPU supervision

Bounded-prefix commit553d1c3 passed159 local tests and remote CI. GPU trial on the
same281-token human example stalled inside mx.eval during the gradient transform;
macOS sample reported26.9G peak physical footprint. CPU equivalence had not tested
GPU resource behavior. Verified PID35811/PPID35810/exact command, took a one-second
sample, then sent SIGINT only to the owned trainer; it exited with saved
KeyboardInterrupt completion. SIGTERM follow-up found it already exited. No other
application was signaled. Run emitted no optimizer/loss record. Earlier one-update
run and all historical artifacts remain preserved and independently hashed.

User reiterated system-pressure limits. No subsequent GPU model workload launched.
Occupied swap after termination was7,171,538,944 bytes, available12,148,146,176 from
a read-only tool observation. Removed unsafe materialization and reject that option
in quality validation before model loading. Added independent CPU watchdog with
strict parent PID/create-time ownership, confirmed startup,100 ms polling through
native calls, snapshot-before-SIGINT and one-second SIGTERM fallback for native
stalls. Guards include starting RAM/swap, running reserve, occupied-swap growth,
swap-out delta, RSS and whole-job/gradient deadlines. Native MLX memory_limit is
only a guideline and not represented as a hard cap. The new supervised profile
keeps complete targets and unchanged4 GiB reserve, plus8 GiB admission and2 GiB
starting-swap maximum. CPU-only current admission refused at8,074,756,096 available
bytes with4,583,915,520 occupied swap bytes; no model was loaded for that check.

Ownership/admission/timeouts tested with CPU fixtures, including an owned sleeping
process interrupted during its operation. No model weights, teacher responses or
GPU allocations in those watchdog tests. Safe GPU optimization, integrity/reload,
capacity and full matched baseline remain unfinished. No improvement result exists.

Safety milestone:171 complete local tests passed, Ruff/diff checks passed. All
added MLX numerical tests use small CPU fixtures; the independent monitor tests
load no model/MLX. Added an explicit CPU-only admission CLI for the guide. Public
and private training calls require a live owned watcher before base loading.
Documentation distinguishes the historical zero-step data milestone, one measured
update, failed four-step gates and unsafe-prefix rejection. No GPU rerun after
pressure correction. Compact evidence and original raw failures are preserved.

### Phase 4C — derived failure reconciliation and refused resource admission

Started at independently reviewed a9f6bf1, main, clean;171 tests/Ruff and latest
remote CI passed. All specified source/template/revision files, Phase4 failure
artifacts and56-task cohort hashes audited without model loading. Human768 TRAIN
pool hashfab6da950a7bbd7ad5ee81911ea7d2129359a93bcdd2ae71bf8680b65972d1b1;
smoke8 hash50a47dac6b91d78bb5d5fe3569a159e74794bc8e382d08c71fd4f0bd3e038a8c.
Model2fc0636 pinned file hashes recorded via bounded CPU reads, native262144/RoPE
unchanged. No new data or teacher generation.

CORRECTION to the earlier GPU-incident accounting: the whole unsafe-prefix run did
emit three completed loss/update records. Update1 took91.064s, update2 7.166s,
update3 17.917s; numeric logged MLX peak27,863,031,210 bytes. At interruption the
active microbatch was step4 with333 complete tokens. Its221,943-byte step-two
safetensors file has six float32 LoRA tensors/55,296 finite parameters. Independent
CPU byte fingerprint7cf287f9fddb85dc13bdde9fc18989a15443748a6dce9ab97e0ccba1a1697ef1
matches the completed step-two log. Original file/report hashes remain intact;
new derived audit corrects the earlier zero-step/empty-loss claim. Quarantine marker
prevents unsafe/partial attachment before model loading. No inference from it,
no full-smoke acceptance and no post-training base/reload validation. One ordinary
update plus three unsafe updates are separate failed initializations, not one smoke.

Fresh CPU resource command runs/phase4c-resource-admission-01 refused:8,432,910,336
available bytes below8 GiB,3,278,503,936 occupied swap bytes above2 GiB. Repository
lease available; no competing model observed. Ordinary Chrome/Codex app consumption
and retained swap are the current admission problem; no process was killed, cache
cleared or Mac restarted. NO GPU workload or optimizer update launched this pass.
User-controlled quiet session/closing unneeded tabs/windows may help; retained swap
may warrant a user-controlled restart. No indefinite resource retry loop.

Fixed monitor-failure detection during blocking work with a parent liveness guard,
fresh session/heartbeat identity checks, atomic fsynced stop records and explicit
phase journals. Success publication moves after watchdog exit/final acceptance.
Current deadlines unchanged:120s whole smoke including load/integrity/reload/short
inference,15s active gradient. CPU-owned timeout/dead-monitor cleanup fixtures pass;
no heavy watchdog stress or supervised Metal validation. Sampled safety remains
limited for arbitrarily fast spikes. No more gradient implementation experiments.

Baseline driver now has CPU inventory, explicit unique retry attempts and at most
two fresh jobs/invocation under the unchanged supervised resource policy. It never
regenerates a valid durable row, rejects contract drift/duplicate success, preserves
failed records and withholds aggregate metrics until all112 jobs complete. Real
historical EOS/error records were used for CPU recovery validation; new56-task
baseline remains0/112 because admission failed. Original32-task partial measurements
are incompatible historical diagnostics, not substituted cohort outcomes.

DecisionB: address shared-host admission and verified watchdog/controller issues,
then exactly one supervised technical smoke when healthy. No basis yet for A; a
clean admitted Mac failure has not been measured, so no CUDA backend/migration was
started. No workstation hardware/VRAM/access assumption. No128-update SFT authorized.

### Phase 4D — fresh qualification refused; adapter paths audited

Started on clean main at b82ea945e2d7d53dfdc14ff6d62a67fc224e9179, matching
remote main and successful GitHub CI. Starting 175 local tests and Ruff passed.
Revalidated the approved eight-example smoke and 768-example human GSM8K pools,
exact source/license/review/template hashes and all cached pinned model files by
CPU-only checks. Native 262144 context, RoPE and tokenizer unchanged. All 127
starting tracked report/split files and historical raw failure inventories remain
byte-identical; unsafe checkpoint quarantine remains present. Historical optimizer
counts remain zero / one / three across separately initialized failed attempts.

One fresh admission at runs/phase4d-resource-admission-01 exited 2:
7,418,003,456 available RAM bytes (6.909 GiB) against 8,589,934,592 required;
2,665,873,408 occupied swap bytes (2.483 GiB) against 2,147,483,648 allowed.
Both gates fail. Model lease free, no project model process observed. CPU system
snapshot records Mac15,6, 18 GiB RAM, vm_stat and kernel pressure value 1; that
pressure indicator does not override admission. Chrome/Codex/ChatGPT ordinary app
RSS was recorded without private command arguments. No application signaled,
cache cleared, global memory limit changed or Mac restarted. No model loaded,
optimizer update, inference or automatic admission retry. An initial CPU process
inspection encountered a missing memory_info value; the error is retained beside
the corrected CPU inspection, separate from model results.

Reviewed watchdog session/parent identity, heartbeat, owned signals, fsynced stop
journals, cleanup/reaping and final acceptance order. Existing 120-second deadline
covers loading through integrity/reload/fresh inference; 15 seconds applies only
while phase=gradient. No limit weakened. CPU fixtures pass, Metal behavior remains
unvalidated and sampled controls cannot guarantee arbitrarily fast spikes.

Closed remaining adapter validation paths: offline context metadata now requires
complete adapter approval; pending training reload requires current live owned
watchdog, initializing exact-config attempt, source permission, matching metadata
and updated tensor hash. Weight-free CPU tests verify rejection of quarantine,
failed/partial and foreign pending attempts. This is not a new training algorithm.
Final validation: 178 local tests, 148 device-free tests (30 deselected), Ruff and
whitespace checks pass.

CPU baseline inventory still 0/112 completed, 56 pending per mode, no new
infrastructure attempt and no accuracy/EOS/cap/token/latency result. Earlier failed
cohorts remain historical diagnostics. No replacement task set was generated.

Next decision: user-controlled quiet Mac session, one new admission and only then
one supervised smoke. Current evidence cannot establish Mac SFT feasibility or
justify a CUDA migration; no workstation/backend was assumed or built. A conditional
128-update rank-4/scale-2 final-two-layer proposal is saved in Phase 4D manifests,
with all 768 human examples eligible under a 512-token complete bound (actual
97–499). At batch/accumulation one it consumes 128 distinct examples, not a whole
corpus epoch. Its throughput/deadline needs qualification; one historical step's
156-second linear extrapolation is explicitly uncalibrated. No improvement run
is authorized or executed. See reports/phase4d-review.md and phase4d-progress.json.
