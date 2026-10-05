"""Transparent format-deliberation cues; manually calibrated, not an oracle."""

import re

from qwenlean.metrics.repetition import words

PATTERN = re.compile(
    r"\b(?:format(?:ting)?|instructions?|literal(?:ly)?|ambiguit(?:y|ies)|ambiguous)\b|"
    r"\b(?:end (?:your|my|the) response|followed by|last (?:line|characters)|"
    r"output (?:structure|constraint)|interpretation)\b",
    re.I,
)


def format_meta_reasoning(text):
    chunks = [m for m in re.finditer(r"[^\n]+", text)]
    hits = [m for m in chunks if PATTERN.search(m.group())]
    total = len(words(text))
    return {
        "method": "format_meta_line_cues_v1",
        "flagged_lines": len(hits),
        "word_density": sum(len(words(m.group())) for m in hits) / max(1, total),
        "literal_FINAL_colon_mentions": len(re.findall(r"\bFINAL\s*:", text)),
        "evidence": [
            {"start_char": m.start(), "end_char": m.end(), "text": m.group()} for m in hits
        ],
    }
