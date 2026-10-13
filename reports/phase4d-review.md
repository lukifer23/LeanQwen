# Phase 4D — fresh resource refusal and adapter validation

Fresh resource admission failed. **No model loaded, GPU job, optimizer update or
baseline generation ran in this pass.** Training viability and the complete stock
baseline remain unverified. No improvement or approved adapter is claimed.

Starting HEAD was `b82ea945e2d7d53dfdc14ff6d62a67fc224e9179` on clean `main`,
matching remote main; its GitHub CI passed. Detailed measurements, raw artifact
paths/hashes, revisions and the conditional SFT proposal are in
[phase4d-progress.json](phase4d-progress.json). The repository commit containing
this report records the resulting milestone; ending SHA/CI are reported after push.

## Fresh resources

One CPU admission command used the unchanged supervised configuration and new
`runs/phase4d-resource-admission-01` destination. It exited 2.

| Measure | Actual | Gate | Result |
|---|---:|---:|---|
| Available RAM | 7,418,003,456 bytes / 6.909 GiB | at least 8 GiB | refused |
| Occupied swap | 2,665,873,408 bytes / 2.483 GiB | at most 2 GiB | refused |

The RAM shortfall is 1.091 GiB; occupied swap exceeds admission by 0.483 GiB.
These are fresh observations, not reused Phase 4C values. Mac15,6 has 18 GiB
physical RAM. CPU inspection captured vm_stat, swapusage and kernel pressure value
1. The pressure indicator does not override the explicit admission gates.
The model lease was available and released; no project model workload was observed.
Ordinary Chrome, Codex and ChatGPT processes appear in the saved RSS inventory.
RSS is not additive unified-memory ownership or proof of GPU activity.

No application was killed, another project's cache cleared, global memory setting
changed, or reboot performed. No automatic retry followed refusal. User-controlled
closing of unused browser tabs/windows and inactive application sessions may help.
If occupied swap persists, a user-controlled restart and quiet work session may be
needed. Both gates must pass anew; freeing RAM alone is insufficient.

The running policy is unchanged: 4 GiB available-RAM reserve, 128 MiB additional
swap occupancy, 512 MiB swap-out delta, 3 GiB process RSS, 15-second gradient and
120-second total technical-smoke limits. No heavy watchdog stress was performed.

## Safety and code

Reviewed monitor startup before weights, exact parent/monitor PID/create-time,
fresh session/heartbeat binding, durable stop journals, owned-process signaling,
normal cleanup/reaping and final acceptance before success publication.
The total deadline includes loading, fingerprinting, backward/update, checkpoints,
release, fresh reload, integrity and inference. Gradient timeout applies only to
active gradient computation. CPU-owned fixtures pass; a full successful supervised
Metal session is still absent. Sampled controls cannot guarantee protection from
arbitrarily fast memory spikes; MLX memory settings remain guidelines.

Two adapter paths were tightened. The offline context metadata audit now requires
complete adapter approval instead of accepting file hashes/model pins alone. The
trainer's pending-save reload now requires the current live owned watchdog,
initializing exact-config attempt, approved source, matching metadata and a CPU
file fingerprint matching the just-updated tensors. This private acceptance step
confers no public inference approval. Quarantined/failed historical attempts cannot
enter it. Normal model loading retains its complete-approval requirement. Forensic
CPU byte inspection remains read-only and gives no permission to use weights.

No gradient implementation, architecture, data selection, precision, objective,
training layer coverage or resource limit was changed.

## Source, historical integrity and technical smoke

Revalidated all eight smoke and 768 TRAIN examples against the same source,
license, published QA, arithmetic screening, complete target/template/EOS and
manifest contracts. Source: original human GSM8K TRAIN revision
`3101c7d5072418e28b9008a6636bde82a006892c`, MIT. Dataset hashes:

- Smoke: `50a47dac6b91d78bb5d5fe3569a159e74794bc8e382d08c71fd4f0bd3e038a8c`.
- TRAIN: `fab6da950a7bbd7ad5ee81911ea7d2129359a93bcdd2ae71bf8680b65972d1b1`.

Cached original bf16 Qwen revision
`2fc06364715b967f1860aea9cf38778875588b17` files were rehashed with bounded CPU
reads, including weights. All match Phase 4C. Native 262144 context, RoPE,
tokenizer and reversible runtime profiles are unchanged. All 127 starting tracked
report/split artifacts and the historical raw failure inventory remain unchanged.
Unsafe-prefix quarantine is present. No teacher/model response became training data.

Historical optimizer counts remain **0 / 1 / 3** in separate failed runs. The
unsafe three-update run and intact step-two file remain unapproved; the corrected
[Phase 4C audit](phase4c-historical-training-audit.json) is unchanged. There is no
successful four-step smoke or accepted adapter.

Phase 4D smoke was **not attempted**, because admission failed. Therefore new
loss/gradient records, MLX peak/active allocation, training swap changes, adapter
hashes, post-training frozen-base verification, save/reload/logit equivalence and
fresh inference are unavailable. No safety or technical success is inferred.

## Frozen stock baseline

CPU inventory of `data/generated/external-human-v1-frozen-dev` against
`runs/phase4c-stock-dev` found **0/112 completed**, 56 pending per mode and zero
new infrastructure attempts. Protocol hash remains
`5bc818fe5e780e6e5715b6380991639bcc9ec540a53ae1c8e45c7d8124343efd`.
No task, prompt, seed, scorer, sampler or 2048-token allowance changed.
Thinking/non-thinking accuracy, EOS/cap rates, token distributions and latency are
null. Prior failed cohorts remain separate; no incomplete cohort aggregate is
published. The one-seed design limits certainty of future paired comparisons.

## Validation and decision

178 full local tests, 148 device-free tests (30 deselected), Ruff and whitespace
checks passed. Added tests are CPU-only rejection checks; no model weights or GPU
stress fixtures. Existing watchdog CPU fixtures also passed. CI remains weight-free.

The next decision is a **quiet Mac qualification session**, followed by one fresh
admission and, only if healthy, the same four-step smoke. Current refusal shows
shared system resources are insufficient; it does not establish that a properly
qualified ordinary-path Mac run is intrinsically infeasible. CUDA migration is
not yet warranted by an admitted failure, and no workstation/backend was assumed
or built. Exact next admission/smoke commands are in the progress manifest and
[guide](../docs/guide.md). No automatic repeat attempt or full SFT is authorized.

## Conditional first improvement experiment

This is a design proposal, **not an executable or approved training schedule**.
Prerequisites remain full smoke/persistence acceptance, complete 112-job stock
baseline, measured capacity for complete 499-token examples and independent
justification of a longer supervised training deadline.

| Setting | Proposed value |
|---|---|
| Eligible pool | unchanged 768 approved human GSM8K TRAIN solutions |
| LoRA | rank 4, scale 2, dropout 0; final layers 22 and 23 |
| Targets | layer 22 `linear_attn.in_proj_qkv`; layer 23 `self_attn.q_proj`, `self_attn.v_proj` |
| Optimization | AdamW, LR 1e-4, weight decay 0.01; batch 1, accumulation 1 |
| Full example bound | 512 tokens; actual entire pool 97–499, no truncation |
| Objective | unchanged response-only full-vocabulary chunked loss, chunk 32; prompt masking, first response/EOS supervision |
| Schedule | 128 updates, seed 20261012, checkpoints at 32/64/96/128 |
| Periodic DEV | 16 source/family/difficulty-stratified tasks selected and frozen before training; both frozen modes at updates 32/64/96 |
| Final DEV | all 56 fixed tasks × both modes; identical seeds/samplers/budgets |

Batch/accumulation one consumes 128 distinct examples, one-sixth of the eligible
pool, rather than a full epoch. Broader adaptation is a later controlled decision.
Keep adapters unfused/removable. The one historical 1.216-second step extrapolates
to about **156 seconds for updates alone**, but this is uncalibrated for longer
examples and excludes load/integrity/reload/evaluation. No defensible complete-run
estimate exists yet. The unchanged 120-second smoke timeout cannot authorize this
schedule; the longer-run deadline remains unset pending measured justification.

Stop on existing RAM/swap/RSS limits, monitor failure, nonfinite loss/gradients,
zero or invalid updates, frozen-base/source/context/tokenizer drift or persistence
failure. A periodic subset drop of at least two task completions against its fixed
stock reference triggers stop and review; this prospective alarm is not a proven
statistical threshold. Final acceptance requires no accuracy decrease in either
mode, preserved harder-task success, paired token/repetition improvements that
retain correctness, and no material regression in matched supported context checks
through approximately 32K. Context jobs require independently admitted monitored
execution; no 128K campaign. Shorter wrong answers or lower loss cannot establish
success. None of this improvement experiment ran in Phase 4D.
