# Phase 3 progress — 2026-10-12

Active work continues on main. No improved model is claimed. The user's PonderScope
Qwen3.5-4B workload (PID 23131) exited naturally with `EVIDENCE_COMPLETE` before
LeanQwen model runs. It was never terminated by this work. Original 144-trajectory
results and historical artifacts are preserved. After the audit preflights passed,
the staged tail, early-exit probes and semantic calibration completed successfully.

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
The suite preserves identical tasks/seeds for later adapter comparison. Native
stock probes have now completed: 3/5 at 2K, 4/5 at 8K and 4/5 at 32K, with all
beginning/middle/end facts retrieved. The inventory-reasoning question failed at
every length, including an incorrect naturally terminated answer at 32K. Context
retrieval and useful long-document reasoning therefore remain separate findings.

Installed MLX-LM 0.32.0 supports prefill chunk selection via `stream_generate`.
Default 2048; smaller chunks are ready for matched measurement. Qwen uses 18
persistent linear-state blocks and six growing full-attention KV caches.
No sliding cache, quantization, destructive position edits, or system Metal-limit
override was introduced. No resource-saving claim follows from a ceiling change.
Actual 512-chunk peak allocation fell from 3.016 to 2.473 GB at 8K (18.0%) and
5.016 to 2.960 GB at 32K (41.0%); correctness stayed 4/5. Mean prefill at 32K:
16.274 versus 16.381 seconds. Four of five 8K outputs and all five 32K outputs
matched exactly. No representative population or controlled thermal estimate.

Prefill/decode telemetry samples actual allocator/RSS and separate attention-KV
and DeltaNet-state sizes. Each request owns fresh full-history caches, with no
cross-request cache reuse. The benchmark's declared memory floor also stops after
sampled prefill/decode boundaries, retaining explicit failure evidence. None of
these observations establishes a 100K/128K/262K feasibility or adapter-retention claim.

Exact implemented commands and progressive native-window restoration recipe:
[context-management.md](../docs/context-management.md). 100K, 128K and 262K
behavior remain unverified; no adapter exists to validate restoration with one.
Short live profile restoration now also verifies every loaded tensor and model
argument unchanged, with identical 64-token emissions. See
`phase3_live_profile_restoration.json` and `phase3_gpu_handoff.json`.

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
Latest full Mac validation: 98 tests and Ruff passed. Small actual Metal bf16
hashing passed with stable hashes, value sensitivity and six-byte maximum host
copies in the primitive test. The loaded stock model also passed live short
EOS/cap generation, 6 KV/18 state caches, consistent telemetry, cleanup and
identical outputs in all three profiles. Two historical P2 trajectories reproduce
their first 128 emitted IDs exactly. Full tensor/position/tokenizer metadata remains
unchanged. These are compatibility results, not optimizer or quality improvement.
The integrity contract now independently compares all canonical frozen parameters
against the full original loaded parameter inventory before LoRA attachment.

Historical research-log restoration is committed independently at `466041f`,
whose CI run 38286571600 passed. All original Phase 1 entries from `8f67d3b2`
and later Phase 2/3 entries were merged verbatim, with source/entry/link checks.
See [preflight checkpoint](phase3-preflight-checkpoint.md) for the exact gates and
next commands. No GPU test or model run starts while a competing project is active.

## Completed reasoning diagnostics and next gates

CPU-only guard replay of the saved 144 prompt trajectories is now measured:
zero triggers at all three predeclared threshold settings, using the actual
16-token polling cadence. Observed token savings are zero; precision/recall remain
unidentified. This rules out a benefit from these literal-cycle guards on those
saved trajectories, while saying nothing about paraphrased redundant reasoning.
The original partial prompt-only result remains preserved. Full replay now covers
161 unique prompt/tail trajectories, again with zero triggers/savings at the same
three configurations. See [full replay](runtime_guard_calibration.md).

Seven tail cases completed: correct EOS at 498 and 6122 reasoning tokens; five
right-censored (four at 8192, state machine at 16384). All 13 exact prefix checks
passed. Forced-close recovery: 3/34 correct EOS across two of six tasks, including
one successful zero-prefix control; 31 capped final responses. The 128-token
final budget limits interpretation: incomplete explanation is not proof of
mathematical inability. Semantic @0.90: precision 91.7%, recall 57.9% on 36 curated
pairs across five DEV tasks. No held-out calibration or training-filter claim.
See [diagnosis and sources](phase3-reasoning-diagnosis.md).

1. Completed stock 2K/8K/32K capability checks, bounded prefill comparison and
   live short-request profile restoration. Full-length larger contexts deferred.
2. Completed seven-case termination-tail protocol, own-prefix early exit including
   zero controls, semantic calibration and offline guard replay.
3. Active 16 TRAIN tasks × four trajectories at P2/4096; measure and review
   natural clean-target coverage. Solver targets are conditional on poor coverage.
4. Hardened optimizer smoke, sequence-feasibility measurements and one conditional
   real SFT pilot, followed by matched reasoning, non-thinking and context evaluation.

Natural TRAIN responses are being saved in
`runs/20261012T184117-phase2-train-natural-n4-1af34da1`; see the explicitly partial
[snapshot](phase3_natural_pilot_snapshot.json). Four eligible responses in that
snapshot come from one task, so repetitions do not establish corpus diversity.
Final coverage remains unknown until all 64 responses and quality reports publish.

The user requested generation finish before deciding whether to continue into
training. Optimizer smoke and SFT are now held for that decision, superseding
earlier automatic conditional progression. Documented
[data options and retention safeguards](../docs/training-data-strategy.md) are
proposals, not implemented sources or proven capability preservation.

No solver corpus, optimizer steps, trained adapter,
stock-versus-trained efficiency comparison or context-retention result exists yet. Historical
P2 has 7/36 correct and 29/36 capped at 2048; this remains a diagnosis input, not
post-training success. Recommendations about real adapter coverage/data quantity
must await the actual gates.
