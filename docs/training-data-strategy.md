# Training data and capability preservation

Status: review checkpoint, 2026-10-12. The sixteen-task × four-trajectory natural
pilot completed with four correct natural responses from one task; sixty traces
capped/unclosed. No optimizer step, solver corpus, external dataset import or
closed-model teacher run has occurred. See the
[generation review](../reports/phase3-natural-pilot-review.md).
The project is idle for user/independent-agent review. New model jobs, data-source
implementation, optimizer smoke and real SFT are held until the user's next
instructions. This supersedes earlier automatic conditional progression.

## Current natural pilot

Deterministic procedural-v2 generators supply independently computed answers.
The predeclared cohort contains sixteen TRAIN tasks: one trivial and one medium
task from each of eight families. Original pinned bf16 Qwen produces four sampled
responses per task, P2, official thinking, a 4096 total-output-token allowance and
the existing sampler/seed contract. Generation is serial under the model lease.

Every response is retained, including capped/incorrect ones. Automatic selection
requires correctness, natural EOS, complete reasoning/final partition, no detected
literal cycle and no detected invalid arithmetic. Parsed intermediate checks are
narrow; individual review remains mandatory. Neither a correct number inside an
unclosed trace nor a short incorrect answer qualifies. Selection returns at most
one natural target per task. Four correct repeats of one task are not four
independent training problems.

The partial snapshot in
[phase3_natural_pilot_snapshot.json](../reports/phase3_natural_pilot_snapshot.json)
is explicitly incomplete and preserved as history. Final
[quality evidence](../reports/train_candidate_quality.json) and
[raw responses](../reports/train_candidate_records.jsonl) are now published.
Only one distinct task qualifies automatically; no reviewed training pool is
approved. The script exited without launching training.

## Proposed data sources

| Source | Role | Current boundary |
|---|---|---|
| Natural Qwen responses | Preserve successful native reasoning styles | First pilot accepts only exact, reviewed natural parents |
| Deterministic solvers | Reproducible answers and valid intermediate operations across families/difficulties | Implement a separate fail-closed source contract only if completed natural coverage is inadequate and continuation is chosen |
| Coding/review assistants | Help develop generators, audit verifier logic and identify missing cases | Reviewer prose and directly authored solutions stay outside response targets; retain provenance and do not relabel teacher text as algorithmic |
| Licensed public worked examples | Add linguistic/task diversity and productive reasoning | No import yet; source/license/verification/contamination review and a separate ingestion contract required |
| Closed-model teacher responses | Possible separately identified distillation condition | Disabled without verified provider permission and explicit source/experiment contracts |

For the solver path, require exact independently checked ground truth, validation
of every intermediate operation, an explicit algorithmic origin, solver/version
hashes, reproducible seeds, TRAIN-only permission, complete official response/EOS
formatting, duplicate checks, split-contamination checks and an exact manifest.
Do not bypass the existing `qwen_self` validator or mix sources invisibly.
Use the actual algorithm's operations as evidence; passing teacher-authored text
through a script does not change its origin.

Open sources considered, not imported:

- [Reasoning Gym](https://github.com/open-thought/reasoning-gym): Apache-2.0
  procedural generators/verifiable environments. Verified final answers alone
  do not certify an intermediate reasoning trace.
- [GSM8K original training data](https://github.com/openai/grade-school-math):
  human-written multi-step problems/worked solutions with calculation annotations.
  Consider original `train.jsonl`, preserving test data and excluding the separate
  model-generated/Socratic variants from a human-origin claim.
- [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2):
  CC-BY-4.0 Llama-generated solutions. Augmented answer labels use majority voting,
  which is not independent exact verification. Consider a bounded audited subset.
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k):
  Apache-2.0 DeepSeek-R1 traces with completeness/verification metadata. Its 16K
  generation allowance and potentially long traces make efficiency suitability
  a question to measure, not an assumed property.

Output ownership is not itself permission for model distillation. Review the
applicable account/service agreement before introducing GPT/Codex/Claude outputs
as response targets. [OpenAI services agreement](https://openai.com/policies/services-agreement/)
restricts developing competing models with outputs subject to defined exceptions;
[Anthropic guidance](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model)
also restricts training competing generative models. No permission determination
or external teacher generation has been performed here.

## Proposed safeguards for a future bounded SFT pilot

These are gates to implement/measure if continuation is chosen, not completed
retention results or guarantees against forgetting.

1. Prefer distinct, verified tasks over repeated epochs on a tiny successful
   subset. Balance families/difficulty and include longer productive reasoning,
   necessary self-correction and appropriate ordinary-utility preservation data.
2. Deduplicate before splitting. Separate task structures and sources where
   possible; changed numbers alone do not establish generalization. Freeze DEV
   selection/evaluation stimuli before training; TEST remains reserved.
3. Measure complete-example fit/exclusions at 1024/2048/4096 tokens. Reject whole
   oversized examples rather than truncate. These training bounds do not change
   the native positional range or the operational context ceiling.
4. Run actual optimizer/save/reload smoke first after an approved target pool.
   For real adaptation, start with one conservative removable LoRA configuration,
   a bounded exposure budget and meaningful checkpoints. Select by held-out
   behavior, not decreasing training loss. Avoid a large configuration sweep.
5. Compare stock and adapter under matched prompts/seeds/sampler/scorer/precision,
   output allowance and runtime profile. Report accuracy by family/difficulty,
   reasoning-token distributions, EOS/cap rates, latency and calibrated redundancy.
6. Test ordinary non-thinking utility, difficult multi-step reasoning and the exact
   supported stock context suite. Reject/stop on instability, answer-quality
   collapse, serious capability regression or excessive resource pressure.

Frozen original weights and an unfused adapter preserve rollback. They do not
guarantee that applying the adapter preserves behavior: forgetting is possible
under parameter-efficient tuning too. See
[research on forgetting in parameter-efficient tuning](https://arxiv.org/abs/2402.18865).
100K/128K/262K inference and adapter context retention remain unverified.
The objective is correctness-preserving reduction of wasted reasoning, never
shortness alone.
