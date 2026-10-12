# Historical Phase 4 solver reasoning contract

**Superseded source strategy.** The critical teacher correction makes licensed
external worked solutions primary. This document describes the preserved solver
implementation, not current SFT authorization. Solver/natural corpora cannot enter
current SFT. See [current strategy](training-data-strategy.md).

The goal is useful reasoning per token with correctness and capability retention.
Natural-only best-of-four coverage failed in Phase 3; that evidence remains intact.
This experiment uses algorithmic operations, not assistant-authored worked answers.

`solver-tasks-1.0.1` / `operation-solver-1.0.0` generates arithmetic and Boolean
trees, modular chains, sequence transformations, unique ordering constraints,
nested affine algebra, exact-cent purchases/discounts/fees and bounded subset
counts. Each proof has an initial state, complete typed operations with before/after
states and contexts, and a final answer. A fixed grammar renders those computed
states. No prose compression model, rejected-hypothesis trace or fabricated
correction example is used. Longer dependencies receive their complete derivation.

`independent-proof-verifier-1.0.0` checks every operation and final result through
separate routes: Python AST/Fraction arithmetic, 64-assignment Boolean truth tables,
unreduced modular simulation, index/multiset invariants, graph transitive closure,
symbolic affine coefficients plus substitution, exact discount subtraction, and
exhaustive combinations for every prefix and count-table cell. Unknown operations,
missing/extra steps and unrecognized claims fail closed. This verifier certifies
typed proofs, not arbitrary natural-language mathematical claims. The historical
narrow numeric-line checker is not used as a proof certificate or rewritten.

Every target retains generator seed/attempt/version, task/family/complexity,
independent expected answer, solver/verifier versions, full proof and certificate
hash, rendered prompt/response, exact token IDs, model/tokenizer/template pins,
Apache-2.0 permission, split/namespace, source-file hashes and source-archive hash.
Approval reproduces the entire archive from the seed and recipe. Natural Qwen
records retain their historical exact-parent path. This historical contract does not
authorize current external-only training.
Uniform solver sampling is explicitly recorded as 100% solver / 0% natural.

Pinned Qwen thinking template ends with `<think>\n`. The supervised suffix starts
with computed reasoning, has one `</think>`, the integer final answer and
`<|im_end|>`; no second opening tag. Prompt prediction positions are masked; the
first response token and final EOS receive loss. Rendering, encoding, decoding and
final-channel parsing must agree exactly. No target is silently cropped.

TRAIN uses structural hash buckets 0–2, OOD DEV bucket 3 and TEST bucket 4.
Within TRAIN, normalized task and response structures, semantic problems and exact
prompts are unique. Word signatures exclude discount values; ordering response
schedules are deduplicated by list size/query position as well as graph structure.
Initial v1 manifests are preserved; corrected v1.1 supersedes them for training.
ID DEV deliberately shares TRAIN topology with fresh constants/entities (counting
shifts the integer domain); it measures numeric generalization. OOD DEV measures
new topology under the same family algorithms. Neither is a conceptual-domain
holdout. Ordering transitive-edge patterns and word baskets share algorithms even
when their signatures differ. TEST remains a reserved seed/bucket namespace; no
TEST proof/answer/tokenized pool or model evaluation is emitted.

```bash
uv run --frozen python scripts/build_solver_corpus.py --historical-reproduction --output data/generated/<new-corpus> --per-family 64
uv run --frozen python scripts/phase4_dev.py --freeze --corpus data/generated/<new-corpus> --cohort data/generated/<new-cohort>
uv run --frozen python scripts/phase4_dev.py --cohort data/generated/<new-cohort> --output runs/<stock-baseline>
uv run --frozen python scripts/approve_solver_pool.py --historical-reproduction --corpus data/generated/<new-corpus> --output data/processed/<new-pool> --config configs/training/smoke.yaml --selection smoke
```

All data generation and model work holds `model_process_lock`. New destinations
are required. Large exact datasets and adapters remain ignored; compact manifests,
source recipes and results are tracked. The frozen 32-task DEV cohort has simple
and harder tasks in eight families, both ID/OOD, matched seeds, P2, official
mode-specific samplers, 2048 output tokens and guard off. One seed per task limits
sampler-variance inference. Stock and adapter each run both modes on identical tasks.

Actual optimizer smoke, complete-example capacity, bounded SFT, paired redundancy,
utility and supported long-context retention must be measured before any improvement
claim. Rank 4 / scale 2 in the last two mixed-attention blocks is a technical
starting point. Base parameters, native 262144 position/RoPE and tokenizer remain
untouched, and adapters remain removable. A short target or lower training loss is
not evidence of improved reasoning efficiency.
