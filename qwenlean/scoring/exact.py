"""Policy-independent terminal numeric correctness, with frozen historical v5."""

import re

from qwenlean.scoring import legacy_v5
from qwenlean.scoring.legacy_v5 import NUMBER as NUMBER
from qwenlean.scoring.legacy_v5 import extract_answer as extract_answer
from qwenlean.scoring.legacy_v5 import extract_conclusion, normalize

SCORING_VERSION = "terminal_numeric_policies_v6"
LEGACY_SCORING_VERSION = legacy_v5.SCORING_VERSION


def _terminal_context_is_example(final):
    tail = final.strip().splitlines()[-2:]
    return any(
        re.search(r"\b(?:example|hypothetically|maybe|perhaps)\b|\be\.g\.", line, re.I)
        for line in tail
    )


def _extract(final):
    if not final.strip():
        return None, "unscorable_final"
    if _terminal_context_is_example(final) or re.search(r'[`"\']\s*$', final):
        return None, "ambiguous_quoted_or_example_answer"
    clean = (
        final.replace("**", "")
        .replace("\\[", "")
        .replace("\\]", "")
        .replace("\\(", "")
        .replace("\\)", "")
    )
    # Canonicalize a numeric LaTeX fraction, independent of the task label.
    clean = re.sub(r"\\frac\{([-+]?\d+)\}\{(\d+)\}", r"\1/\2", clean)
    return (
        legacy_v5.extract_terminal_answer(clean)
        if legacy_v5.extract_terminal_answer(clean)[1] != "unscorable_final"
        else extract_conclusion(clean)
    )


def format_compliance(final, policy):
    if policy == "P0":
        return None  # no imposed format, so the statistic is not applicable
    if policy == "P1":
        return bool(
            re.search(r"\\boxed\{(?:" + NUMBER + r"|\\frac\{[-+]?\d+\}\{\d+\})\}[\s.$]*$", final)
        )
    if policy == "P2":
        lines = final.strip().splitlines()
        return bool(lines and re.fullmatch(r"[-+]?\d+", lines[-1].strip()))
    if policy == "P3":
        return bool(re.search(r"FINAL\s*:\s*(" + NUMBER + r")\s*$", final, re.I))
    raise ValueError("Unknown prompt policy")


def score(final, expected, policy="P3", version=None):
    version = version or SCORING_VERSION
    if version == LEGACY_SCORING_VERSION:
        return legacy_v5.score(final, expected)
    if version != SCORING_VERSION:
        raise ValueError("Unsupported scoring contract")
    target = normalize(expected)
    if target is None:
        raise ValueError("Exact-answer ground truth must be a valid rational number")
    answer, method = _extract(final)
    historical = legacy_v5.score(final, expected)
    return {
        "correct": answer is not None and normalize(answer) == target,
        "strict_final_correct": historical["strict_final_correct"],
        "format_compliant": format_compliance(final, policy),
        "extracted_answer": answer,
        "scoring_method": method,
        "scoring_version": version,
    }
