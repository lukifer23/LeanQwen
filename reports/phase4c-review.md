# Phase 4C — corrected evidence, refused admission, and CPU fixes

GPU work was refused by fresh resource admission. No model was loaded, no new
optimizer update executed, and the frozen stock baseline was not run in this pass.
Starting HEAD: `a9f6bf1f9d66124f71175beb655e166557b18985`, main, clean;
starting local tests171/Ruff/remote CI passed. The source/model/data hashes are
saved in [starting audit](phase4c-start-audit.json). Native262144/RoPE/tokenizer,
original bf16 files and human-source contracts are unchanged.

## Historical failure reconciliation

| Failed run | Completed updates | Evidence / outcome |
|---|---:|---|
| Original external-human smoke | 0 | Backward stopped at RAM floor; no completed update |
| Chunked-loss memory-efficient retry | 1 | Finite loss1.066536665/gradients;281 complete tokens,171 supervised;1.215982s; next backward RAM stop |
| Unsafe frozen-prefix retry | 3 | Consecutive update1–3 records and changed adapter hashes; fourth333-token microbatch interrupted |

The unsafe run's original whole-run zero/empty-loss report was wrong. A new
[derived CPU audit](phase4c-historical-training-audit.json) corrects it without
modifying the original report. Update timings91.0644/7.1663/17.9174s and losses
1.06654/1.19691/1.30372 are on different examples, not an efficacy learning curve.
Logged MLX peak27,863,031,210 bytes is distinct from macOS's26.9G displayed peak
footprint. Neither is missing from the corrected accounting.

The step-two file is221,943 bytes, SHA256
`f755acdbae7afb457660d1de480b56cc986545944c17c8d506c04b0b68a4aa80`.
Independent CPU parsing verifies complete payload, six float32 LoRA tensors,
55,296 finite parameters and exact tensor fingerprint
`7cf287f9fddb85dc13bdde9fc18989a15443748a6dce9ab97e0ccba1a1697ef1`, matching
step two. It is an intact **unapproved partial artifact**, retained in its original
ignored directory with quarantine metadata. No inference from it and no publication
of weights. New attachment checks reject quarantine/partial/failed or mismatched
source/integrity/reload evidence before base loading. Frozen-prefix code remains
disabled. Four updates across separate failed initializations are not one smoke.

## Resource admission and diagnosis

The actual CPU CLI matched the requested command and saved
`runs/phase4c-resource-admission-01/resource-admission.json`, exiting2.
Available RAM8,432,910,336 bytes (**7.854 GiB**) was below8 GiB;
occupied swap3,278,503,936 bytes (**3.053 GiB**) exceeded2 GiB.
Both criteria fail. The profile's4 GiB running reserve,128 MiB swap growth,
512 MiB swap-out,3 GiB RSS and bounded15s-gradient/120s-total limits are unchanged.

The model lease was available, and no competing model was observed. Normal Chrome
and Codex app processes appear in the RSS snapshot; RSS is not additive unified
memory ownership. Existing swap pressure and ordinary app consumption block
admission. No process, other project's cache or global system configuration was
changed. User-controlled closing of unused Chrome tabs/windows and inactive app
sessions may help. Retained swap may warrant a user-controlled restart and quiet
work session. No automatic admission retry or GPU attempt followed refusal.

## Watchdog and acceptance corrections

Confirmed exact parent/monitor PID and create-time identities, fresh session IDs,
atomic heartbeat/status files and stale-state rejection. Stop records are atomically
published and fsynced before signaling only the owned trainer. A parent liveness
guard interrupts on monitor death/heartbeat failure during blocking work; CPU-owned
fixtures validate death detection, timeout, cleanup and unrelated-PID rejection.

Explicit phase journaling covers loading, integrity, gradient, optimizer, original
release, fresh reload, reload integrity and short inference. The120-second deadline
runs from admission through the whole smoke, including those stages;15 seconds
applies only to active gradient work. Existing ordinary step1 timing1.216s and
historical short inference throughput make this a bounded technical allowance,
not a guarantee that every quiet-Mac run finishes. It has not been relaxed.
Success is now published only after watchdog exit/final resource acceptance.

These controls have CPU validation, not a successful full Metal session. Sampled
checks cannot guarantee protection from arbitrarily fast memory spikes, and native
MLX's memory setting is a guideline, not a hard cap. No new gradient implementation,
heavy watchdog stress, or frozen-prefix rerun was attempted.

## Training and evaluation gates

The same eight approved human examples were revalidated: dataset hash
`50a47dac6b91d78bb5d5fe3569a159e74794bc8e382d08c71fd4f0bd3e038a8c`, complete
sequence lengths137/281/172/146/333/262/202/237. The768-example TRAIN hash remains
`fab6da950a7bbd7ad5ee81911ea7d2129359a93bcdd2ae71bf8680b65972d1b1`.
No source research, paid API call, target rewriting or teacher generation occurred.

The supervised smoke was **not attempted** because admission failed. No new loss,
gradient, allocation, adapter/save-reload/logit equivalence, frozen-base-after or
fresh-inference result exists. The four-step acceptance gate remains incomplete.

The56-task frozen DEV protocol hash is
`5bc818fe5e780e6e5715b6380991639bcc9ec540a53ae1c8e45c7d8124343efd`;
task hash`871ac0862005ed764c094282d6829c7e00f6aa8d83aaa0a149792fb306c11e10`.
CPU inventory: **0/112 jobs complete**,56 pending per mode. Accuracy/EOS/cap rates
are null; no partial benchmark aggregate is published. Earlier32-task failures
remain distinct. The updated driver supports monitored one/two-job batches and
explicit unique retries. Genuine completed EOS/cap jobs are reused; changed
config/model/scoring contracts or duplicate success are rejected. Infrastructure
attempts remain separate, and all-cohort aggregates require112 accepted jobs.
Actual saved EOS/error records passed CPU recovery checks; no new inference ran.

Validation:175 full local tests,145 CI-equivalent tests (30 deselected), Ruff and
whitespace checks passed. All123 prior tracked report/split artifacts remain
byte-identical. CI remains weight-free. See [machine-readable current gates](phase4c-progress.json).

## Decision and exact next experiment

**B: address shared Mac resource admission and the verified controller issues.**
A is premature: no complete smoke or baseline exists. C is not yet supported:
healthy admitted ordinary-path Metal training has not failed in this pass, and
no accessible CUDA workstation/hardware/VRAM inventory was provided. No second
backend was built. If a future admitted single smoke fails, use its measured cause
before considering a workstation and equivalent source/precision/LoRA contracts.

In a quiet session, make one fresh CPU admission check using the existing supervised
profile and a new destination. Only if it passes and remains healthy, execute one
four-step smoke on the same approved eight-example pool, new output directory,
rank4/scale2, final two mixed-attention blocks, AdamW1e-4, batch/accumulation1,
1024 complete-example bound and checkpointed full-vocabulary loss. Exact commands
and no-retry conditions are recorded in the [progress manifest](phase4c-progress.json)
and [operating guide](../docs/guide.md). Then complete the fixed baseline in safe
batches. A128-update improvement run is not authorized. Its configuration must be
chosen only after all technical, persistence, capacity and stock-baseline gates pass.
