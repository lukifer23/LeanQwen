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


FORMAT_ANCHOR = re.compile(
    r'\bFINAL\s*:|\\boxed|\bboxed\b|\bformat(?:ting)?\b|'
    r'\b(?:final|last)\s+line\b|\bend (?:your|my|the) response\b|'
    r'\bfollowed by (?:just |only )?the (?:integer )?answer\b', re.I)
FORMAT_CONTEXT = re.compile(r'\b(?:interpret(?:ation|ed)?|literal(?:ly)?|space|markdown|'
                            r'extra text|last line|end|followed by|output|phrase|instruction)\b', re.I)


def format_meta_reasoning_v2(text):
    """Narrow anchors separate task-rule ambiguity from output-format discussion.

    Derived measurement only: v1 fields in original records remain unchanged.
    Context expansion is limited to eight following lines inside a paragraph.
    """
    lines = list(re.finditer(r'[^\n]+', text))
    hits, last_anchor, previous_end = [], -100, 0
    for i, line in enumerate(lines):
        if '\n\n' in text[previous_end:line.start()]:
            last_anchor = -100
        anchor = bool(FORMAT_ANCHOR.search(line.group()))
        if re.search(r'\b(?:logic|problem) formats?\b', line.group(), re.I):
            anchor = False
        if anchor:
            last_anchor = i
        if anchor or (i-last_anchor <= 8 and FORMAT_CONTEXT.search(line.group())):
            hits.append(line)
        previous_end = line.end()
    return {'method':'format_meta_anchored_lines_v2', 'flagged_lines':len(hits),
            'word_density':sum(len(words(m.group())) for m in hits)/max(1,len(words(text))),
            'literal_FINAL_colon_mentions':len(re.findall(r'\bFINAL\s*:',text)),
            'evidence':[{'start_char':m.start(),'end_char':m.end(),'text':m.group()} for m in hits]}
