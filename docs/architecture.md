# Architecture and compatibility decision

Current checkpoint (2026-10-05): the matched prompt study is complete and work is
stopped with no active QwenLean model worker. See [measured status and next steps](../reports/phase2_checkpoint.md).
Higher-cap diagnosis, semantic inference, TRAIN generation and optimizer smoke
remain pending; implemented interfaces are not evidence that those experiments ran.

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

Only deterministic code-generated problems are included in this milestone.
No externally licensed benchmark or proprietary model response enters a data
pool. The generators and generated stimuli use this repository's Apache-2.0
license. Responses originate from pinned Qwen weights. DEV/TEST permissions are
false for training. Unknown or closed-model origins fail the future training
gate; derived records must retain parent IDs. Numerical labels are algorithmic.
A future imported dataset must be reviewed individually before import.

## Phase 2 implementation boundaries

Device code now also lives in `training/mlx_sft.py`; framework-independent
training contracts live in `training/quality.py`. The real optimizer interface
is implemented but awaits approved TRAIN data and a measured save/reload smoke.
Explicit targets are full-attention `self_attn.q_proj`/`v_proj` and gated-delta
`linear_attn.in_proj_qkv` in selected final layers. Attachment verifies the expected
adapter tensor count instead of silently falling back to generic modules.

Local semantic analysis uses Apache-2.0
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
pinned to `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU float32 and one thread.
Its model card discloses public sentence-pair training sources. It is used only
as a local measurement encoder, not a response teacher or source of training
content. Encoder-sized chunk subdivision is separate from Qwen context handling.
The Qwen context remains 262144; the documented acceptable floor is 131072.
Long-context behavior after any future adapter is unmeasured.
