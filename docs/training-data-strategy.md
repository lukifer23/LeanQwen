# Training data and capability preservation

The user's critical teacher/data-source correction supersedes both natural
self-distillation and the subsequent solver-primary plan. Qwen is the student
and evaluation target. It never supplies, rewrites, compresses or judges targets.
The same restriction applies to every Qwen-family model, LFM, weak local models
and LeanQwen. No LM Studio commands. Historical evidence and exact-parent
validators remain available for offline reproduction; their approvals cannot
authorize current SFT.

## Selected first source

Use **768 distinct original human-written GSM8K TRAIN solutions**, pinned to
`3101c7d5072418e28b9008a6636bde82a006892c`, under MIT. This excludes Socratic
variants, model-generated solutions and official TEST responses. A separate
64-question DEV holdout comes from original TRAIN, assigned before selection.
All source reasoning and calculator annotations remain intact. Only the source
final delimiter is replaced with Qwen's thinking/final boundary and turn EOS;
the final integer is canonicalized. No generated replacement prose.

[Source comparison](../reports/external-source-comparison.md) records actual
bounded measurements of GSM8K, OpenR1-Math-220k, OpenMathInstruct-2 and DeepMath-103K.
The first pilot is 100% human reference data. Stronger synthetic sources remain
separate until their upstream rights, benchmark provenance and quality pass.
No paid teacher is needed for this bounded pilot.

## Approval and its limits

The explicitly selected approval basis is **published human QA plus exact
screening**, limited to at most 1,000 original GSM8K reference examples. The
[GSM8K paper](https://arxiv.org/html/2110.14168v2) documents worker-answer-agreement
quality control. Its exact archived publication hash is required by the code.
This basis does not claim new individual project review or formal verification
of every natural-language claim. Both remain explicitly false in the records.

Every accepted row must reproduce its exact pinned human source, contain a
complete final label, pass all recognized calculator annotations using exact
rational arithmetic, link its last verified annotation to the source label,
and pass deterministic completion/repetition screening. Unknown, unfinished or
invalid annotations fail closed. These checks do not prove that each equation
correctly models the problem's prose. Flagged examples are excluded, not rewritten.
Unannotated symbolic/contextual claims are not silently certified.

An alternative individual human review path requires source-bound judgments
for correctness, mathematical consistency, relevance, completion and productive
effort. A future independent frontier reviewer requires explicit provider/spend
authorization and archived permissions and raw reviews. Coding-agent judgments
cannot be relabeled human review. Reviewer notes never enter targets.

## Source-aware enforcement

`licensed_external_reasoning` has its own registered permission and quality
contract. Source/license files, immutable revision, raw source row, task identity,
normalized structure, checks, review basis, tokenizer contract, exact rendered
prompt and response, token IDs and dataset hashes are bound together. Every SFT
entrypoint checks permission before reading the tokenized pool or loading model
tensors, then reproduces each source and its exact tokenization.

`authorized_frontier_reasoning` has an optional permission schema; actual target
ingestion is disabled. Full DeepSeek-R1 has permissive model licensing, and the
audited official API terms permit distillation, but current API aliases are V4
models rather than an attested R1 endpoint. A model license alone cannot authorize
a different provider/model. No paid calls without the user's provider/spend approval.
GPT/Codex, Claude and Gemini targets remain excluded without verified authorization.

Historical `qwen_self` exact-parent checks are preserved in historical validators.
Historical `solver_verified_reasoning` proofs and datasets are retained but excluded
from current training. Deterministic computation now verifies external solutions;
solver-authored targets require a separately authorized future ablation.

## Diversity, partitioning and lengths

The pilot has 768 task identities and 768 normalized structure fingerprints.
Three annotation-count bins contribute 256 examples each: 1–2, 3–4, and 5–8
operations. This is a structural proxy, not measured student difficulty or eight
independent domains. Linguistic variety comes from the original human source.
Reasoning lengths span 33–346 tokens; full prompt-plus-target lengths span 97–499.
All examples fit under 512/1024/2048/4096 tokens without cropping. This naturally
short corpus does not cover long productive CoT or prove long-sequence feasibility.
See [complete compact audit](../reports/external-human-pilot.json).

Official GSM8K TEST questions and repository stimuli are inspected only for
contamination hashes. No TEST answer tokenization or model evaluation occurs.
Exact/number-free wording and operation fingerprints have zero detected TRAIN/DEV
overlap; semantic paraphrases, all other benchmarks and base pretraining exposure
are outside the check's scope. The frozen 56-task comparison uses 24 of the 64 GSM8K DEV questions (eight per
operation-count bin) and all 32 procedural retention tasks. GSM8K DEV measures
within-source generalization.
The previously frozen procedural DEV cohort measures retention/domain shift
relative to GSM8K; its old solver ID/OOD labels are not current training labels.

## Optimizer and improvement gates

The first technical experiment uses `external-human-smoke.yaml`: four AdamW
steps, rank 4/scale 2, last two mixed-attention blocks, batch/accumulation 1,
learning rate 1e-4 and a 1024 maximum complete-example bound. Require finite loss
and gradients, nonzero adapter updates, all frozen-base tensor hashes unchanged,
save/reload equivalence and fresh inference. The actual examples are shorter than
512 tokens; a passing smoke will not establish 1024/2048/4096 capacity.

No improvement run starts before a complete matched stock DEV baseline and measured
hardware feasibility. Initial proposed pilot: one removable rank-4 two-block
adapter, sequence bound 512, 128 optimizer updates with accumulation 4 and
checkpoints every 32 updates. This schedule is conditional, not an executed run.
Periodic matched DEV checks and resource evidence must support the final schedule.
Stock/adapter thinking and non-thinking comparisons use the same tasks, prompts,
seeds, decoding, budgets and scoring. Report accuracy, EOS/caps, tokens per correct
completion, latency and calibrated redundancy; shorter wrong answers are failure.

Retain useful hard-task reasoning, non-thinking utility and the supported stock
context suites through 32K envelopes with 512-token prefill. No new 128K stress
campaign. Native 262144 position/context, RoPE, tokenizer and original weights
remain unchanged. Runtime profiles mac-128k/mac-100k/native-262k are reversible;
training sequence bounds are separate. Adapters stay unfused and removable.

## Preserved negative result

The completed natural pilot produced 4/64 correct natural EOS responses, all
from `2 + (-3)`, and 60 capped/unclosed traces. One distinct eligible task and
zero optimizer steps are historical evidence, not a reason to increase N or
relabeled teacher data. See the unchanged
[review](../reports/phase3-natural-pilot-review.md) and
[raw records](../reports/train_candidate_records.jsonl).
