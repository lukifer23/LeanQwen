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
