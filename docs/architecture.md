# Architecture and compatibility decision

Phase 3 resumes the completed prompt-study checkpoint; the user's PonderScope run
finished naturally before LeanQwen model work. See [current progress](../reports/phase3-progress.md)
and [reversible runtime context contracts](context-management.md). The backend
and compatibility evidence below remain historical unless explicitly updated.
Higher-cap diagnosis, DEV semantic measurement and natural TRAIN generation
completed. No optimizer ran in Phase 3. Phase 4 now uses licensed external human/strong-model reference solutions;
Qwen is exclusively the student/evaluation target and historical solver/natural
approvals cannot authorize current SFT;
see [generation review](../reports/phase3-natural-pilot-review.md) and
[data strategy](training-data-strategy.md).

2026-10-05, M3 Pro (11 CPU cores), 18 GiB unified RAM, macOS 27.2.
See `reports/environment.json` for the measured environment. Python 3.12.12 is
isolated in `.venv`, dependencies resolved in `uv.lock`. The pre-existing global
Python was 3.14.8 with MLX 0.32.3 but no PyTorch/Transformers/PEFT/TRL installation.

Use MLX 0.32.3 + MLX-LM 0.32.0 for this first text-only experiment. Load original
Qwen weights at pinned revision `2fc06364715b967f1860aea9cf38778875588b17` without
quantization. MLX-LM's Qwen3.5 implementation sanitizes HF tensor names and drops
the unused vision tower; this is text inference, not multimodal inference. Its
hybrid gated-delta/attention implementation is native to MLX.

Compatibility audit:

| Stack | Exact-checkpoint evidence | Remaining uncertainty |
|---|---|---|
| Transformers 5.18.0 | AutoModelForCausalLM loads Qwen3.5 and generates on MPS | CUDA performance not measured |
| PyTorch 2.14.1 MPS | MPS available; bf16 eager inference works; 64-token diagnostic took 7.42 s | Hybrid kernels use reference PyTorch implementations; training backward not tested |
| PEFT 0.21.2 | Real r=4 q_proj/v_proj adapter attachment succeeds: 159,744 trainable parameters | Backward/optimizer not tested; explicit modules needed |
| TRL 1.14.1 | Installed code includes Qwen3.5 thinking/non-thinking and training templates | No SFT/DPO run; template support alone is not end-to-end training validation |
| MLX 0.32.3 / MLX-LM 0.32.0 | Original checkpoint runs; 512-token diagnostic took 7.15 s; adapter backward passes in training mode: 55,296 adapter parameters, 1.00 s, 1.73 GB MLX peak | Long sequence training still needs a real SFT smoke test |

These diagnostic timings use different lengths and sampling; they justify a
practical backend choice, not a rigorous performance benchmark. MLX adapter
probe uses fixed token IDs, a numerical scalar loss, and zero optimizer steps.
No training targets or trained checkpoint are produced.

[Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[MLX-LM architecture](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py),
[MLX-LM LoRA conversion](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py),
[PEFT module mappings](https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py).
MLX users have reported Qwen3.5 [long-sequence backward memory failures](https://github.com/ml-explore/mlx/issues/3539)
and [resource-limit failures](https://github.com/ml-explore/mlx-lm/issues/1185) on
larger models. These reports motivate bounded future smoke tests; they do not
establish that the 0.8B model will fail on this machine.

## Backend boundary

Apple device imports are isolated in `inference/mlx_backend.py` and
`training/mlx_sft.py`. Optional semantic inference imports its CPU framework only
when the local encoder is explicitly constructed.
Diagnostic scripts import the frameworks they probe directly. The runner consumes a backend
with `generate(prompt, seed, config)` and `tokenizer`. Scoring, provenance, procedural
tasks and statistics are independent of device and model framework. A future
Transformers/CUDA backend can implement that interface; macOS/Linux process
locking currently uses POSIX `flock`. Windows is not a validated target. No CUDA implementation
is claimed in this milestone. The implemented MLX training command awaits its
actual optimizer/save/reload smoke validation.

## Thinking and sampler semantics

The [official checkpoint card](https://huggingface.co/Qwen/Qwen3.5-0.8B) states
that non-thinking is default and warns specifically about this model's thinking
loops. Enable thinking through the official template parameter. Thinking prompt
suffix: `assistant\n<think>\n`; non-thinking precloses an empty block. Generated
`</think>` is the reasoning boundary. The tokenizer ends turns at `<|im_end|>` (248046); text config also specifies
`<|endoftext|>` (248044). Honor both and record the actual stop token.
A missing boundary means an unclosed trace;
reasoning never gets scored as a final answer. Record EOS separately from token
cap termination, and never interpret a token cap as clean termination.

A uses the official text-thinking sampling parameters, with a practical 2048
*total output* cap. This cap is an experimental bound, not an official recommended
limit. Penalties apply to all generated tokens only, excluding prompt tokens.
Order: additive presence penalty, sign-aware repetition penalty, temperature,
top-k, top-p (renormalized after top-k), min-p, categorical sampling. MLX-LM's
convenience sampler uses a different ordering and 20-token penalty windows by
default; we define these semantics explicitly and test them. Run seeds are
stable per sample ID; identical IDs receive identical seeds across variants.
Reproducibility is scoped to the pinned software/backend, not bitwise equivalence
with every serving engine. Runs save templates, actual token IDs, model revision,
configuration hashes, dependency versions, dataset hashes and Git revision.

## Data provenance

Current teacher data consists of original human-written GSM8K TRAIN under MIT,
with its own source permission, immutable revision and exact source/template/hash
contract. Permission is checked before tokenized data or model tensors are loaded.
Published human QA plus exact screening is a bounded approval basis with prose
semantics explicitly unverified. Individual human/frontier reviews are a separate
contract. No Qwen/LFM/student/weak-local targets, coding-agent solutions or unapproved
closed-model outputs. Historical natural exact-parent and solver proof contracts
are retained for offline reproduction, excluded from current SFT. DEV/TEST responses
are forbidden as targets. See [current data strategy](training-data-strategy.md).

## Phase 2 implementation boundaries

Device code now also lives in `training/mlx_sft.py`; framework-independent
training contracts live in `training/quality.py`. The real optimizer interface
now has approved external human TRAIN data and one finite optimizer update,
but the full technical smoke/integrity/reload validation remains incomplete.
Explicit targets are full-attention `self_attn.q_proj`/`v_proj` and gated-delta
`linear_attn.in_proj_qkv` in selected final layers. Attachment verifies the expected
adapter tensor count instead of silently falling back to generic modules.
Phase 3 strengthens this to exact target tensor names and bitwise fingerprinting
of every frozen base parameter, re-enumerated from the live model before/after
training and on fresh adapter reload. Host copies are chunked; base weights are
not duplicated. CPU tests exercise dtype/shape/bit sensitivity and replacement of
an unadapted tensor. One update is measured; full post-training integrity/reload
validation remains pending.

Local semantic analysis uses Apache-2.0
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
pinned to `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU float32 and one thread.
Its model card discloses public sentence-pair training sources. It is used only
as a local measurement encoder, not a response teacher or source of training
content. Encoder-sized chunk subdivision is separate from Qwen context handling.
Qwen's native positional context remains 262144. Phase 3's reversible runtime
ceilings are 131072 by default, 102400 as an authorized fallback, and 262144 for
restoration. Long-context behavior after any future adapter is unmeasured.

## Training memory and controller correction

Training decoder inputs and native architecture remain intact. The loss selects
supervised hidden positions before projecting the frozen head, uses full-vocabulary
float32 cross entropy in checkpointed chunks and preserves all target tokens/EOS.
Tiny CPU float32/bf16 and Qwen LoRA-gradient equivalence checks are passing.
The experimental frozen-prefix materialization path caused unsafe GPU pressure
and was removed; its option is rejected before weights load. CPU numerical
agreement is not GPU resource approval.

An independent CPU process now supervises SFT across native evaluation, with
verified parent PID/create-time ownership, resource admission, RAM/swap-occupancy/
swap-out/RSS checks and gradient/job deadlines. Both public and private training
paths require the live watchdog before loading weights. Watchdog startup must be
confirmed; stops preserve snapshots before interrupting only the owned trainer.
Native MLX's memory setting is a guideline, not a hard allocation cap. CPU watchdog
fixtures pass, but the supervised GPU smoke has not been attempted after the
pressure correction. See [measured safety report](../reports/local-training-resource-safety.md).
