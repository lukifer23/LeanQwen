"""Strict final-answer scoring; never give final accuracy credit for reasoning alone."""

import re
from fractions import Fraction

NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?"


def extract_answer(final):
    markers = re.findall(r"FINAL\s*:\s*(" + NUMBER + r")(?![\d./])", final, re.I)
    if markers:
        # Conflicting final markers are not unambiguous evidence.
        if len({normalize(x) for x in markers}) != 1:
            return None, "conflicting_final_markers"
        return markers[-1], "final_marker"
    boxed = re.findall(r"\\boxed\{(" + NUMBER + r")\}", final)
    if boxed and len({normalize(x) for x in boxed}) == 1:
        return boxed[-1], "boxed"
    plain = re.fullmatch(r"\s*(" + NUMBER + r")\s*[.!]?\s*", final)
    return (plain.group(1), "plain") if plain else (None, "unscorable_final")


def normalize(value):
    try:
        return Fraction(value.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def score(final, expected):
    answer, method = extract_answer(final)
    return {
        "correct": answer is not None and normalize(answer) == normalize(expected),
        "extracted_answer": answer,
        "scoring_method": method,
    }
