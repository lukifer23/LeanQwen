# Procedural-v2 structural diversity

Eight families × four structural difficulty bins × two records = 64 tasks per split. Exact labels are independently unit-tested. No output-policy wording is baked into task bodies.

V2 structural signature counts: {'train': 50, 'dev': 50, 'test': 50}. Cross-split structure collisions and exact semantic-task duplicates: zero. V1 logic used the same four-formula template in all splits; matching truth-assignment patterns across splits: {'train/dev': 3, 'train/test': 2, 'dev/test': 4}.

Signatures remove numeric values and variable aliases for expression trees. Algebra/word templates and ordering sizes are deliberate structural holdouts. This creates distribution shift and does not establish conceptual independence, matched empirical difficulty, or generalization. Counting has a bounded finite space and is not a broad combinatorial benchmark. Generator namespaces, labels and structural checks are audited without model-evaluating TEST.

Reproduce: `uv run --frozen qwenlean build-dataset --version v2 --per-bin 2 --output data/splits/<new-directory>` and `uv run --frozen pytest -q tests/test_procedural_v2.py`.
