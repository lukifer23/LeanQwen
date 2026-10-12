# QwenLean working rules

- Keep every update on `main`. Do not create branches or worktrees. Do not force-push.
- Remote: https://github.com/lukifer23/LeanQwen.git. Make logical commits and push completed milestones.
- Keep exactly one model workload active. Use `model_process_lock` for inference,
  probes, data generation and future training. Check process ownership before
  terminating anything; do not kill unrelated applications.
- Never invent measurements. Retain raw outputs and failed-run explanations.
- TEST is reserved; use DEV for sampler selection and TRAIN for candidate generation.
- No proprietary closed-model outputs in training data without verified permission.
- Do not use LM Studio, including status commands. Never use Qwen-family, LFM, weak/small local models or LeanQwen as a teacher, target rewriter/compressor, subjective reasoning judge or dataset-cleaning model. Qwen is the student/evaluation target through direct Python/MLX. Historical Qwen targets are reproduction-only and excluded from current SFT.
- No Docker. No long training until baseline/decoding diagnosis and dataset quality gates.
- The critical teacher/data-source addendum supersedes the earlier solver-primary Phase 4 plan: prioritize licensed external human/strong-model worked reasoning, starting with original GSM8K TRAIN and audited OpenR1. Deterministic solvers verify data; their authored targets are excluded unless a future separate ablation is explicitly authorized. Preserve all previous corpora/reports/raw evidence. No new natural-Qwen teacher collection or automatic fallback targets.
- Verify dataset license AND upstream output permissions, immutable revision, provenance and benchmark contamination. No GPT/Codex, Claude or Gemini authored solutions without verified authorization. Coding-agent prose/solutions never become targets. Subjective quality uses individual human or provider-authorized independent strong review. For the bounded original GSM8K human-reference pilot (at most 1,000 examples), published worker-agreement QA plus exact screening is an explicitly scoped approval basis: do not claim fresh individual review or formal prose verification. Unknown/unverified claims stay separate.
- No paid API calls without explicit user approval of provider and spend. Source permission and complete-record quality approval must pass before any SFT tensor loading. Freeze matched DEV, validate exact template/EOS, then four-step MLX smoke, integrity/reload, measured capacity and one bounded SFT pilot only after all gates pass. Current solver/natural approvals cannot authorize training under the new policy.
- Verify changes with `uv run --frozen pytest -q` and Ruff as appropriate.

- Preserve the native 262144-token model/position/RoPE configuration. Reversible runtime profiles are mac-128k (131072, default), mac-100k (102400, explicitly authorized fallback), and native-262k (262144). Refuse excess combined input/output budgets; never truncate. Output and training sequence budgets are separate.
- Preserve historical v1/v5 artifacts; new prompt experiments use independent IDs/scoring contracts and task-clustered replicate statistics.
