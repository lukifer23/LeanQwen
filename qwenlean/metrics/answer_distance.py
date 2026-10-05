"""Candidate correct-conclusion distance, explicitly not an oracle for solved state."""

import re

from qwenlean.scoring.exact import NUMBER, normalize


def conclusion_distance(text, reasoning_ids, expected, tokenizer):
    pattern = re.compile(
        r"(?:\b(?:answer|result|total|therefore|thus)\b[^\n]{0,60}?|=\s*)("
        + NUMBER
        + r")(?![\d./])",
        re.I,
    )
    matches = [m for m in pattern.finditer(text) if normalize(m.group(1)) == normalize(expected)]
    if not matches:
        return {
            "candidate_correct_conclusion_token": None,
            "tokens_after_candidate": None,
            "evidence": None,
            "method": "conclusion_cue_or_equation_rhs_v1",
        }
    match = matches[0]
    # Decode prefixes of actual emitted IDs; avoid retokenization drift at boundaries.
    target_char = match.end()
    lo, hi = 0, len(reasoning_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(reasoning_ids[:mid])) >= target_char:
            hi = mid
        else:
            lo = mid + 1
    return {
        "candidate_correct_conclusion_token": lo,
        "tokens_after_candidate": len(reasoning_ids) - lo,
        "evidence": match.group(0),
        "method": "conclusion_cue_or_equation_rhs_v1",
    }
