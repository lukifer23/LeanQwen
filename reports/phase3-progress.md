# Phase 3 progress — 2026-10-12

Active work continues on main. No improved model or Phase 3 model experiment
is claimed. The user's PonderScope Qwen3.5-4B GPU workload (PID 23131) must finish
naturally before LeanQwen launches any model workload. It is never terminated by
this work. Original 144-trajectory results and historical artifacts are preserved.

## Repository health

Starting local/fetched remote HEAD: `3d32475d660a9f2ac4c8d157f7794bb0c35392d6`.
Starting checkout clean, 76 local tests and Ruff passed. Original failed GitHub
run 37372433674 was a hosted-runner allocation failure with no executed steps,
not a test failure. Its retry passed. Commit `91a0bb9` pins Ubuntu 24.04/Python
3.12; fresh CI run 38283493219 passed. See `phase3_ci_initial_failure.json`.

## Reversible context preparation

Implemented default `mac-128k` (131072), fallback `mac-100k` (102400), and
`native-262k` (262144). Admission enforces actual templated input plus reserved
output, with no truncation. Each new generation records full budget accounting
and a profile/positional/cache fingerprint. CLI supports profile and prefill chunk
selection; legacy configs/raw trajectories remain untouched.

The CPU-only actual cached-checkpoint audit verifies unchanged original weight,
configuration, tokenizer and chat-template file SHA256 before/after selecting
all profiles. Native maximum remains 262144; original RoPE theta 10000000 and
partial rotary factor 0.25 remain intact. No model was loaded for this audit.
See `phase3_context_configuration.json` and the initial offline snapshot-resolution
failure explanation in `phase3_metadata_initial_failure.json`.

Five deterministic DEV questions per length cover positional retrieval, similar
distractors, distant addition and a basic document calculation. Pinned-tokenizer
tests verify exact 2K/8K/32K combined envelopes with a 256-token reservation.
These tests construct stimuli only; **no context accuracy or memory benchmark
has run**. The suite preserves identical tasks/seeds for later adapter comparison.

Installed MLX-LM 0.32.0 supports prefill chunk selection via `stream_generate`.
Default 2048; smaller chunks are ready for matched measurement. Qwen uses 18
persistent linear-state blocks and six growing full-attention KV caches.
No sliding cache, quantization, destructive position edits, or system Metal-limit
override was introduced. No resource-saving claim follows from a ceiling change.

Prefill/decode telemetry samples actual allocator/RSS and separate attention-KV
and DeltaNet-state sizes. Each request owns fresh full-history caches, with no
cross-request cache reuse. The benchmark's declared memory floor also stops after
sampled prefill/decode boundaries, retaining explicit failure evidence. None of
these fields is a measured resource result until a model run is executed.

Exact implemented commands and progressive native-window restoration recipe:
[context-management.md](../docs/context-management.md). 100K, 128K and 262K
behavior remain unverified; no adapter exists to validate restoration with one.

## Training integrity preparation

The former `.linear.weight` filter only checked adapted base matrices and did not
assert a nonempty verification set; it omitted embeddings, norms and unadapted
blocks, and retained old references that could miss replaced arrays. Verification
now hashes all current frozen parameters by name/shape/dtype/raw bytes with bounded
host copies, rejects empty sets, and repeats on fresh reload. Adapter attachment
checks exact expected layer/module tensor names. Training also rejects all-zero
gradients, unclosed/incorrect targets and early EOS, and records memory/swap and
finite gradients per optimizer step. Reload generation must succeed before the
smoke can report success. The four-step rank-4/two-layer configuration is now
committed at `configs/training/smoke.yaml`.

These are implementation and small CPU unit-test results, not actual Qwen training.
Latest validation: 92 non-MLX tests and six MLX unit tests forced to the CPU pass;
Ruff passes. No GPU test or model run starts while PonderScope is active.

## Pending model evidence

1. After PonderScope exits naturally and memory is suitable: stock 2K/8K/32K
   capability checks and bounded prefill comparison.
2. Existing seven-case termination-tail protocol, own-prefix early exit including
   zero controls, semantic calibration and offline guard replay.
3. Existing 16 TRAIN tasks × four trajectories at P2/4096; measure and review
   natural clean-target coverage. Solver targets are conditional on poor coverage.
4. Hardened optimizer smoke, sequence-feasibility measurements and one conditional
   real SFT pilot, followed by matched reasoning, non-thinking and context evaluation.

No new natural TRAIN generations, solver corpus, optimizer steps, trained adapter,
accuracy/efficiency comparison or context-retention result exists yet. Historical
P2 has 7/36 correct and 29/36 capped at 2048; this remains a diagnosis input, not
post-training success. Recommendations about real adapter coverage/data quantity
must await the actual gates.
