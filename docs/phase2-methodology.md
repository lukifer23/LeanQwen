# Phase 2 — deconfounding before post-training

Starting main: `8f67d3b2a1c217f5cad819ac2b8211b24db57e7d`.
The first-pass stimuli, original generations, v5 scoring module and reports remain historical.
The audit is in `reports/phase2_start.json`; environment validation is a separate artifact.

## Implemented measurement contract

Tasks now have a body and stable task ID; rendering adds a policy and distinct
rendered-prompt ID. Historical procedural-v1 files remain unchanged. P3 rendering
of a v1 body reproduces the original prompt exactly. Replicate zero retains the
historical seed; replicates one and two hash the same seed key plus replicate index.
Policy never influences the seed. Every new trajectory records its replicate,
generation ID, actual prompt token IDs, policy/version and scoring version.

| Policy | Instruction after task body |
|---|---|
| P0 | None |
| P1 | Reason step by step and put the final answer in `\boxed{}`. |
| P2 | On the final line, write only the integer answer. |
| P3 | Historical FINAL instruction, verbatim |

P1 follows the [Qwen model card's math guidance](https://huggingface.co/Qwen/Qwen3.5-0.8B#best-practices).
The card warns about small-model thinking loops and recommends much larger output
budgets than 2048. Our initial cap is a controlled bound, not a natural completion
length or architectural context limit. The higher-cap experiment will be targeted.

Correctness is terminal numeric extraction, independent of ground truth during
extraction. New policy experiments use v6; historical v5 remains callable exactly.
P0 format adherence is not applicable (null), rather than artificially perfect.
Prompt comparisons enforce task content/label, original weights/revision, sampler,
seed, cap, guard, thinking mode and scorer equality. Ordinary sampler comparison
still requires identical rendered prompts. Replicates are clustered by task;
bootstrap intervals average replicate outcomes inside each task first. Generations
are not independent benchmark problems. A small twelve-task study limits certainty.

Format-meta reasoning is a line-cue heuristic for formatting/instruction/endpoint
language. Numeric FINAL mentions are recorded separately. These are proxies that
require manual calibration; ordinary final-answer words do not automatically
count as formatting deliberation.

## Stream failures and resumption

Empty output, iterator exhaustion, unknown finish reasons, premature length stops,
stream errors and cleanup errors have explicit statuses. Valid EOS IDs are required;
no token list or response is indexed before it exists. The iterator closes in a
finally block. Invalid generation is preserved and halts the run rather than
masquerading as EOS or a model accuracy failure. Resume appends pending jobs only;
failed attempts already recorded are not regenerated automatically.

Resumption checks configuration hash, pinned revision, task/rendered dataset hash,
ordered sample/replicate prefix, per-record seeds/labels/prompts and scoring contract.
Completed and explicitly aborted runs cannot be resumed. Partial/corrupt JSONL
is not silently repaired. Resumption events are append-only. The entire model
workload stays under the repository process lock.

Optional package versions are null when absent. Ordinary MLX inference does not
require the compatibility extra. Environment validation takes a pinned revision
and refuses to overwrite existing output.

## Context preservation

The original model's 262144-token configured context window, positional limits and
RoPE settings remain unchanged. Prompt tokenization does not truncate. A prompt
plus output budget exceeding the original window is refused. Output caps and
bounded future smoke-training sequence lengths are experimental budgets, not
changes to model context capacity. Full source traces remain saved; counterfactual
prefixes will be separately labeled interventions. The user permits a documented 128K limit if useful, with a floor of 131072 tokens;
current experiments keep 262144. Lowering a configured ceiling alone does not save
compute for short inputs. Long-context behavioral retention
has not been tested and cannot be inferred from unchanged configuration alone.

## Remaining gates

The prompt-policy experiment, targeted termination tail, manual metric calibration,
early-exit probes, semantic redundancy and guard replay must complete before TRAIN
candidate generation. Procedural-v2 must pass structural contamination checks.
A real optimizer smoke requires quality-approved TRAIN targets and adapter reload
verification. No substantial SFT is authorized until the final review gate.
