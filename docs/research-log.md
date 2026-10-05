# Research log

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
