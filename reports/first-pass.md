# First-pass decision gate — 2026-10-05

The evaluation foundation works. These capped DEV measurements do **not** establish an efficient reasoning model. Keep the original weights for the next diagnostic experiment; do not start SFT yet.

| Variant | Accuracy | Median Think Tokens | P95 Think Tokens | Exact Output Loop Rate | Max-Out Rate | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| A: official thinking | 0.0% | 2048 | 2048 | 0.0% | 100.0% | 33.90 s |
| B: repetition penalty 1.05 | 6.7% | 2048 | 2048 | 0.0% | 93.3% | 32.34 s |
| A0: official non-thinking control | 40.0% | 0 | 0 | 0.0% | 28.3% | 12.39 s |

All rows use the same 60 DEV prompts and per-problem seeds, original bf16 weights, a 2048 **total output** token cap, and no runtime guard. A0 changes both thinking mode and its official sampler. Scoring is terminal_numeric_conclusions_v5, calibrated on observed outputs; historical extraction and format compliance are saved separately. TEST remains reserved.

## 1. What was built

A real MLX inference backend; exact token-ID thinking/final parsing; both declared EOS tokens; six procedural families; TRAIN/DEV/TEST provenance and duplicate checks; exact task grading; repetition, lexical redundancy and candidate-conclusion diagnostics; optional exact-cycle runtime guard; JSONL runs with configs/hashes/seeds/timing/memory; staged sampler selection; paired statistics, reports and plots. CLI commands: build-dataset, generate, eval, analyze, compare. Thirty-two tests pass. No training CLI or CUDA backend is claimed.

## 2. Exact repository structure

See [tracked file manifest](../docs/repository-files.txt) for every tracked file, [README layout](../README.md#layout) for directory purposes, and [operating guide](../docs/guide.md) for commands. Generated weights/caches and complete raw runs are excluded from Git. Compact measurements, trace excerpts and run identifiers are committed. Raw local artifacts are under the paths in [run_sources.json](run_sources.json).

## 3. Backend decision

MLX 0.32.3 / MLX-LM 0.32.0 supports this exact hybrid Qwen3.5 text architecture natively. Transformers 5.18.0 / PyTorch 2.14.1 MPS also loads and generates, but uses reference hybrid kernels. PEFT adapter attachment succeeds; MPS backward is untested. MLX adapter backward succeeds in training mode with 55,296 adapter parameters, zero optimizer steps. TRL templates exist; no TRL training is validated. See [architecture decision and primary sources](../docs/architecture.md). Device-specific inference is isolated; portability is an interface, not an implemented CUDA path.

## 4. Does it run on this Mac?

Yes: original Qwen/Qwen3.5-0.8B, pinned revision 2fc06364715b967f1860aea9cf38778875588b17, bf16 text tower, no quantization, Apple M3 Pro, 18 GiB RAM, macOS 27.2, Python 3.12.12. Official thinking is opt-in; default is non-thinking. Generated </think> separates reasoning from final text. Missing closure is saved explicitly. Stop IDs 248044 and 248046 are both honored. A sequential MPS/MLX greedy/logit probe supports architecture fidelity but does not prove universal device equivalence.

## 5. Actual generation speed

Aggregate emitted output tokens divided by inference wall time (includes prefill/synchronization): A 60.42, B 62.08, A0 65.76 tokens/s. Mean latency: A 33.90, B 32.34, A0 12.39 seconds. These are preliminary interactive-session timings, without controlled thermals/background CPU. Decode-only timing is retained per generation.

## 6. Actual memory

MLX allocator peak: A 1.684, B 1.684, A0 1.689 GB. Sampled process RSS peak: A 2.140, B 0.515, A0 0.360 GB. Measures overlap and must not be summed; neither is total machine use. RSS is sampled every 64 tokens and at boundaries, so peaks can be missed. [Environment artifact](environment.json).

## 7. Baseline accuracy and uncertainty

A 0.0%, Wilson 95% interval 0.0%–6.0%. B 6.7%, interval 2.6%–15.9%. A0 40.0%, interval 28.6%–52.6%. B minus A: +6.7%, paired bootstrap interval 1.7%–13.3%, gained 4, lost 0. Exact paired McNemar two-sided p=0.125; these four wins do not establish significance at 0.05, despite the positive discrete bootstrap interval. These describe this small, sampler-tuned DEV set and one generation seed per task. They do not estimate uncapped capability, seed variance, or general reasoning performance. Family counts are in each summary.

## 8. Reasoning and total token distributions

Reasoning median/p95: A 2048/2048, B 2048/2048. Mean reasoning: A 2044.13, B 1975.77. Mean total output: A 2048.00, B 2007.33, A0 815.13. A0 median total output is 381.5; its zero median thinking partition does not mean zero reasoning-like calculations or compute. The cap censors p90–p99; no inference about the uncapped tail is justified. [Measured distribution plot](reasoning_accuracy.png).

## 9. Repetition and loop statistics

Exact consecutive-cycle detector: A 0.0%, B 0.0%, A0 0.0% of full outputs. This narrow 24–128-token block repeated three times misses paraphrased and longer cycles. Mean reasoning repeated-content density: A 7.76%, B 5.67%. Mean lexical cosine proxy: A 4.83%, B 2.78%; neither establishes uselessness. A0 reasoning-only repetition metrics are not informative about its final-channel text. Runtime guard was not benchmarked: there is no C comparison or measured false-positive rate.

## 10. Major observed failure modes

[Manual trace review](trace_review.md) includes actual outputs: correct 2770 by reasoning token 240 followed by 1808 additional tokens and cap failure; repeated modulo-rule parsing and arithmetic confusion; FINAL: 56 instead of remainder 11; omission of a final multiply operation despite lengthy checking; correct boolean answer with an initially invalid negation; and quoted answer-format examples during an unfinished final channel. These demonstrate several distinct failures, not a single uniform literal-loop problem. A finds candidate correct-conclusion cues in 40/60; this heuristic does not mean all those problems were solved. No causal rate of correct-then-wrong answer degradation has been established.

## 11. Initial sampler changes

Three predeclared one-variable pilots on the same 12 balanced DEV tasks: temperature .6, presence penalty 0, repetition penalty 1.05. See [sampling report](sampling_ablation.md) for counts, token tails, latency and paired results. Repetition 1.05 was promoted because its pilot scored 2/12 versus A’s 0/12; cooler scored 1/12 and presence-zero 0/12. The finalist repeats 12/12 pilot token sequences exactly. No model parameters changed. B is a provisional non-trained candidate, not a validated production decoder.

## 12. Does decoding alone materially solve it?

At this cap and with these prompts, B changes bounded accuracy by +6.7% and cap failures from 100.0% to 93.3%, but median and p95 thinking tokens remain 2048. Mean total-output reduction is 2.0%. This falls short of a substantial efficiency improvement. Prompt-induced format deliberation and cap censoring still confound the diagnosis. The much better default-mode control warrants investigation but does not establish preservation on harder tasks.

## 13. Recommendation for the first fine-tuning data

After diagnostic controls and split fixes, sample N=4 trajectories per TRAIN task from Qwen itself. Prefer naturally correct, nonredundant traces; require final correctness plus spot-checked valid intermediate operations, clean termination, complete provenance and parent IDs for transformations. Keep legitimately long hard solutions; a short wrong answer loses to a correct longer one. Start with best-of-N self-distillation before algorithmic compression. No DEV/TEST trajectories, closed-model teachers, or aggressively rewritten proofs. This is a proposed dataset, not one already constructed or quality-approved.

## 14. Blockers and limitations

No inference blocker remains. Training is gated by diagnosis and absent trajectory/dataset quality pipelines. Procedural-v1 prompt hashes are disjoint, but the small boolean truth domain repeats semantics across pools; introduce a richer generator and explicit semantic split checks before training. Task wording and formatting need control, multiple seeds and harder tasks are missing, runtime-guard false positives and SFT optimizer memory are untested. Existing provenance permission checks establish source/split admissibility, not trajectory quality. No external benchmark has been imported. TEST has never been model-evaluated.

## 15. Exact next experiment

Run a DEV-only output-format control on the same twelve balanced pilot problems, with original weights, official thinking sampler, guard off, three matched per-task seeds, 2048 total-output cap. Compare the current instruction with the explicit sentence: “Write the numeric answer on the last line as FINAL: <integer>.” Retain the mathematical task body, labels and sample pairing; record the prompt variant separately. This requires explicit prompt-variant pairing support: the existing compare command intentionally rejects different prompt text. Measure terminal task accuracy, format compliance, cap rate, total/reasoning tokens and format-deliberation excerpts. Then, if needed, run a small 8192-token diagnostic on preselected failures to distinguish cap censoring from persistent nontermination. These experiments are proposed, not implemented or executed. Do not start SFT until these controls clarify which behavior should become a training target.

## Reproduction and audit

See [guide](../docs/guide.md), [metric contract](../docs/metrics.md), [research log](../docs/research-log.md), machine-readable [sampling measurements](sampling_measurements.json), and compact [per-sample evidence](per_sample_measurements.json). Full runs are local/ignored; committed excerpts and summaries cannot reconstruct every full output. Fresh runs use pinned software/model and deterministic seeds; hardware numerics may differ. Scoring fixes preserve original IDs/raw outputs in versioned backups. All Git development is on main. GitHub About/topics describe the implemented foundation. v0.1.0 names the first-pass research foundation, not an improved model release.
