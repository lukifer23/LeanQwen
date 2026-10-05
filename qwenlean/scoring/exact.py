"""Strict final-answer scoring; never give final accuracy credit for reasoning alone."""

import re
from fractions import Fraction

SCORING_VERSION = "terminal_answers_v4"

NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?"


def extract_answer(final):
    markers = re.findall(r"FINAL\s*:\s*(" + NUMBER + r")(?![\d/]|[.]\d)", final, re.I)
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


def extract_terminal_answer(final):
    # Keep historical extraction separately; planning examples are not answers.
    answer, method = extract_answer(final)
    if method == "conflicting_final_markers":
        return None, method
    if method == "final_marker":
        matches = list(re.finditer(r"FINAL\s*:\s*(" + NUMBER + r")(?![\d/]|[.]\d)", final, re.I))
        if re.fullmatch(r"[\s.!]*", final[matches[-1].end() :]):
            return answer, method
        return None, "nonterminal_final_marker"
    if method == "boxed":
        matches = list(re.finditer(r"\\boxed\{(" + NUMBER + r")\}", final))
        if re.fullmatch(r"[\s.!$]*", final[matches[-1].end() :]):
            return answer, method
        return None, "nonterminal_boxed_answer"
    return answer, method


def normalize(value):
    try:
        return Fraction(value.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def extract_conclusion(final):
    clean = final.replace("**", "").replace("$", "").replace("\\(", "").replace("\\)", "")
    cue = (
        r"(?:final answer|answer|final result|result|integer solution|solution|final state|"
        r"total true expressions|total number of true expressions|true count|total price)"
    )
    pattern = cue + r"\s*(?:is|equals|=|:)\s*(?:[a-z]\s*=\s*)?(" + NUMBER + r")(?![\d/]|[.]\d)"
    matches = []
    for match in re.finditer(pattern, clean, re.I):
        # A conclusion must be terminal, allowing an empty requested FINAL marker.
        tail = clean[match.end() :]
        if re.fullmatch(r"[\s.!]*(?:FINAL\s*:?[\s.!]*)?", tail, re.I):
            matches.append(match.group(1))
    if matches and len({normalize(x) for x in matches}) == 1:
        return matches[-1], "terminal_final_conclusion"
    return None, "unscorable_final"


def score(final, expected):
    target = normalize(expected)
    if target is None:
        raise ValueError("Exact-answer ground truth must be a valid rational number")
    strict_answer, strict_method = extract_answer(final)
    answer, method = extract_terminal_answer(final)
    if answer is None and method == "unscorable_final":
        answer, method = extract_conclusion(final)
    compliant = re.search(r"FINAL\s*:\s*(" + NUMBER + r")\s*$", final, re.I)
    return {
        "correct": answer is not None and normalize(answer) == target,
        "strict_final_correct": strict_answer is not None and normalize(strict_answer) == target,
        "format_compliant": compliant is not None,
        "extracted_answer": answer,
        "scoring_method": method,
        "scoring_version": SCORING_VERSION,
    }
