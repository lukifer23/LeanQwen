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
