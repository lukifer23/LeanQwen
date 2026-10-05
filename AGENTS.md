# QwenLean working rules

- Keep every update on `main`. Do not create branches or worktrees. Do not force-push.
- Remote: https://github.com/lukifer23/LeanQwen.git. Make logical commits and push completed milestones.
- Keep exactly one model workload active. Use `model_process_lock` for inference,
  probes, data generation and future training. Check process ownership before
  terminating anything; do not kill unrelated applications.
- Never invent measurements. Retain raw outputs and failed-run explanations.
- TEST is reserved; use DEV for sampler selection and TRAIN for candidate generation.
- No proprietary closed-model outputs in training data without verified permission.
- No Docker. No long training until baseline/decoding diagnosis and dataset quality gates.
- Verify changes with `uv run --frozen pytest -q` and Ruff as appropriate.

- Keep the original context window by default; an explicitly documented limit may be lowered to 131072 tokens (128K), never below. Preserve RoPE/position settings; refuse excess input instead of silently truncating. Output and smoke-training sequence budgets are separately documented.
- Preserve historical v1/v5 artifacts; new prompt experiments use independent IDs/scoring contracts and task-clustered replicate statistics.
