# Runtime exact-cycle guard replay

Replayed 144 saved DEV trajectories at three threshold settings and the actual 16-token polling cadence. No configuration triggered. Estimated observed token savings: zero. Precision/recall are unidentified because there were no triggers and these traces are not independently cycle-labeled.

No live guard pilot was justified. Guard stays off. Exact-cycle failure to detect paraphrased reconsideration is an expected limitation, not evidence of healthy reasoning. Details: runtime_guard_replay.json.

Scope: prompt study only; tail replay and full diagnostic gate remain pending. Reproduce into a NEW directory: `uv run --frozen python scripts/calibrate_metrics.py --guard-only --prompt-only --output-dir reports/<new-prompt-replay>`.
