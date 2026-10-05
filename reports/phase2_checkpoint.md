# Phase 2 checkpoint — prompt confound milestone

2026-10-05. Stopped at the user’s request after the complete bounded prompt study. This is a partial Phase 2 checkpoint, not a completed post-training pass. All model processes have exited; no follow-on model study was launched.

## Completed evidence

Original unquantized Qwen3.5-0.8B, pinned revision `2fc06364715b967f1860aea9cf38778875588b17`, MLX-LM, official thinking sampler, guard off, 2048 total-output cap. Twelve historical balanced DEV tasks, three deterministic matched seeds, four prompt policies, 144 saved trajectories. Scoring v6 separates task correctness from format compliance.

| Policy | Correct | EOS | Cap hits | Mean output tokens | Median / P95 reasoning | Mixed-seed tasks / 12 |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 3/36 (8.3%) | 3 | 33/36 | 1988.2 | 2048 / 2048 | 2 |
| P1 | 0/36 (0.0%) | 0 | 36/36 | 2048.0 | 2048 / 2048 | 0 |
| P2 | 7/36 (19.4%) | 7 | 29/36 | 1852.0 | 2048 / 2048 | 5 |
| P3 | 3/36 (8.3%) | 3 | 33/36 | 1981.7 | 2048 / 2048 | 2 |

P2 is the provisional DEV choice under the predeclared correctness-first rule. Relative to P3: +11.1 percentage points observed accuracy (task-bootstrap 95% interval 0–22.2 points), 6.5% fewer mean observed total output tokens, and 2.04 seconds lower mean measured generation latency. Its 29/36 cap rate and unchanged median/p95 leave most continuation failures unresolved. This is modest bounded-pilot evidence, not an established general improvement or natural-completion efficiency claim.

P0 (bare task) still capped 33/36. P1 (official-style step-by-step plus boxed answer) capped all 36 and did not improve this pilot. P3 replicate zero reproduced all twelve historical stock-control emitted token sequences exactly. P3’s three successes occur under other seeds, demonstrating why the previous single-seed baseline was insufficient. Replicates are nested within twelve tasks; zero-width bootstrap intervals for P1 do not prove zero population accuracy.

### Format discussion and failure modes

Derived anchored format-meta word densities: P0 0.5%, P1 1.8%, P2 10.0%, P3 25.9%. Raw v1 fields and the original policy-selection rule remain preserved. Format presence includes normal brief adherence; these densities do not directly measure wasted compute. Twelve reviewed DEV traces yielded v1 confusion {'fn': 0, 'fp': 4, 'tn': 1, 'tp': 7} and v2 {'fn': 0, 'fp': 0, 'tn': 5, 'tp': 7}. V2 was designed on the initial eight traces; the four subsequent arithmetic cases are curated extension evidence, not independent human validation.

- Matched arithmetic P3 solves 2006 early, then spends most remaining tokens reinterpreting FINAL syntax without closing. P2 answers 2006 after 498 reasoning tokens. P0 recovers 2006, returns to incorrect subtraction and caps; P1 reaches 2006 early but rechecks operands/arithmetic and caps.
- Ordering and word-problem traces repeatedly deliberate about task interpretation. State-machine traces misread operations; logic traces need real error recovery as well as containing redundancy. Removing output-format instructions does not remove these failures.
- All thirteen EOS completions were correct. No correct-intermediate-to-wrong-final case was observed. Reviewed capped arithmetic has intermediate contradiction, which is a weaker and distinct finding.

### Repetition and conclusion measurement

| Policy | Mean literal repeated-content density | Mean lexical redundancy density | Exact cycles | Candidate correct-conclusion cue |
|---|---:|---:|---:|---:|
| P0 | 5.3% | 3.5% | 0.0% | 21/36 |
| P1 | 5.0% | 4.4% | 0.0% | 19/36 |
| P2 | 5.2% | 3.9% | 0.0% | 20/36 |
| P3 | 7.7% | 3.5% | 0.0% | 23/36 |

Candidate correct-conclusion heuristics fired on 83/144 traces. Among EOS completions, 12/13 had a candidate, with median 512 further reasoning tokens and six gaps exceeding 512. Among 131 capped traces, 71 had a candidate; median observed remainder was 1049 tokens, with 56 exceeding 512. Capped gaps end at the observation cap, not EOS. Numbers, equation right-hand sides and ordinal mentions can be false positives; none of these counts certify that useful computation had ended. Source-linked observable categories and twelve manual annotations are saved separately.

Lexical pair cosine at the predeclared 0.90 threshold: {'fn': 10, 'fp': 2, 'precision': 0.8181818181818182, 'recall': 0.47368421052631576, 'tn': 15, 'tp': 9}. This is 36 curated DEV pairs across five tasks, with one assistant reviewer and selection partly based on lexical similarity. Low measured recall reinforces that word overlap misses paraphrased rederivation. Semantic encoder inference and complete guard calibration have not run.

Aggregate generation throughput: 67.60 output tokens/s, including prefill; summed generation latency 4191.0 seconds. Maximum recorded MLX allocation 1.684 GB and sampled process RSS 0.394 GB. These overlap and are not total machine RAM; interactive background work and thermals were uncontrolled.

![Measured prompt accuracy and capped lengths](figures/prompt_compute_accuracy.png)

## Repository work and validation

- Hardened empty/failed streaming and EOS metadata, optional-package environment recording, exact-contract resume and completed-run protection.
- Added separate task/policy IDs, frozen v6 alongside historical v5, prompt-only comparisons and task-clustered replicates.
- Implemented staged-tail/prefix checks, counterfactual closure with zero-prefix controls, local semantic/guard replay interfaces, natural target ranking, exact-parent approval and guarded masked LoRA SFT. The remaining model-dependent interfaces are smoke-unvalidated.
- Procedural-v2.0.1 contains eight families × four structural difficulty bins × two tasks = 64 per split. Fifty structural signatures per split; no detected cross-split signature or exact semantic-task collisions. Structural holdouts create distribution shift and do not prove conceptual independence or empirical generalization.
- Added weight-free Linux CI, meaningful unit tests, protected derived-report destinations and model-instance lifecycle cleanup. See [exact tracked files](../docs/repository-files.txt) and main’s commit history.
- Historical file SHA256 checks and all six historical run-record digests match the starting audit. Existing first-pass reports, raw generations and v1 splits were not rewritten.
- Native context remains 262144 with unchanged RoPE and no input truncation. The allowed 131072 floor has not been applied. No long-context behavioral-retention claim.
- TEST was not model-evaluated or used for training selection. Generator metadata alone was audited. No new TRAIN responses, approved SFT pool, optimizer steps or trained adapters exist.

Validation: 76 local tests, Ruff and relative document links passed. GitHub Actions
run 37371123362 passed for fd4360e; checkpoint publication has its own subsequent
CI status. No current-commit CI success is assumed before it is observed.

## Remaining gates / next session

1. Read this checkpoint, `docs/guide.md`, `docs/phase2-methodology.md` and the committed protocol; verify main is current, clean, and the model lease is available.
2. Run the predeclared seven-case tail with provisional P2: staged 2048/4096/8192; at most two selected still-unresolved cases at 16384. Reuse matching 2048 records, skip larger caps after EOS, verify exact same-seed prefixes. Report completed and right-censored lengths separately.
3. Run six-source own-prefix forced-close probes including zero-prefix controls. Complete local MiniLM calibration and exact-cycle replay; manually review every trigger before any live guard pilot.
4. Only after those diagnostic gates: generate the predeclared sixteen TRAIN tasks × N=4 at 4096, inspect natural coverage and actual intermediate validity, approve whole fitting targets without truncation.
5. Run the conditional four-step technical LoRA smoke: rank 4, scale 2, last two mixed-attention layers, full-attention q/v and linear-attention qkv, batch 1, accumulation 1, LR 1e-4, AdamW, maximum example sequence 1024, checkpoints every two steps. Save, destroy, reload, hash-check and DEV-sanity-generate. This is not the substantial SFT experiment.
6. Review completed diagnostic/dataset/smoke evidence before choosing the real SFT composition and hyperparameters. Those are not scientifically ready to freeze today.

Exact next model command, when work is resumed:

```bash
uv run --frozen --no-sync python scripts/termination_tail.py
```

Run sequentially, with no other model worker. The controller saves each task/cap durably and can resume without duplicating completed generations. No automatic follow-on job is queued.

## Artifact map

- `prompt_policy_measurements.json`, `prompt_policy_records.jsonl`, `prompt_policy_ablation.md`: complete quantitative study and raw evidence.
- `control_reproduction.json`: twelve bit-identical historical controls.
- `conclusion_calibration_annotations.jsonl`, `prompt_conclusion_states.jsonl`: manual DEV judgments and weak observable categories.
- `format_meta_calibration.json`: preserved initial eight-trace development calibration; `format_meta_checkpoint.json` / `prompt_format_meta_v2.jsonl`: expanded derived analysis.
- `lexical_pair_checkpoint.json` / `redundancy_calibration_pairs.jsonl`: actual curated lexical calibration; semantic inference is pending.
- `procedural_v2_quality.json/.md`, `phase2_protocol.json`: audited generator diversity and predeclared remaining design.
- `phase2_start.json`, `phase2_environment.json`, `phase2_initial_launch_failure.json`, `phase2_stop_audit.json`: starting audit, native context/environment, preserved launch failure and clean-stop checks.
- `runs/20261005T193609-phase2-prompt-policy-8fdc19d6`: ignored complete local run, config, environment, dataset hash, scorer, Git SHA and emitted token archive.
