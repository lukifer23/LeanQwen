# Natural best-of-N TRAIN pilot

64 original-Qwen trajectories, sixteen TRAIN tasks, N=4, policy P2, 4096 total-output cap, guard off. No compression, DEV/TEST responses, or proprietary teacher responses.

Tasks with any correct final: 1/16. Clean eligible natural target coverage: 1/16. Tier counts: {'E': 60, 'A': 4}.

Selected reasoning lengths: [302]. Per-task median candidate minus selected tokens: [126.5].

Quality precedes length. Eligibility requires correct final, clean EOS/partition, no exact loop and no detected invalid numeric claim. Recognized intermediate claims are narrowly checked; absence of a detected error is not proof of reasoning validity. Selected targets remain a review pool until individual examples are approved in a hashed quality manifest.

Reproduce: `uv run --frozen python scripts/train_candidates.py`. Resume with `--resume runs/<incomplete-run>`.
