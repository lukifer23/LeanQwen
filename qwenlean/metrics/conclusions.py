"""Observable conclusion categories; no assertion of internal certainty.

Ground truth is used only for DEV/EVAL annotation, never to select final answers.
A numeric mention alone is not a solved state. Contradiction cues are imperfect.
"""

import re

from qwenlean.scoring.exact import NUMBER, normalize

CUE = re.compile(r'\b(?:answer|result|total|therefore|thus|final state|conclusion)\b', re.I)


def conclusion_states(text, expected, final_correct, termination):
    target = normalize(expected)
    mentions, conclusions = [], []
    for m in re.finditer(r'(?<![\w.])(' + NUMBER + r')(?![\w/]|[.]\d)', text):
        before = text[max(text.rfind('\n', 0, m.start()) + 1, m.start() - 100):m.start()]
        value = normalize(m.group(1))
        cue = bool(CUE.search(before))
        event = {'char_start': m.start(), 'char_end': m.end(), 'value': str(value),
                 'correct_value': value == target, 'conclusion_cue': cue,
                 'context': text[max(0, m.start()-80):min(len(text), m.end()+60)]}
        if value == target:
            mentions.append(event)
        if cue:
            conclusions.append(event)
    correct = [e for e in conclusions if e['correct_value']]
    first = correct[0]['char_end'] if correct else None
    contradictions = [e for e in conclusions if not e['correct_value'] and first is not None
                      and e['char_start'] > first]
    return {'method': 'observable_numeric_conclusion_categories_v1',
            'correct_numeric_intermediate_appears': bool(mentions),
            'correct_after_conclusion_cue': bool(correct),
            'stable_correct_cue_no_later_conflicting_cue': bool(correct) and not contradictions,
            'later_conflicting_numeric_conclusion_cue': bool(contradictions),
            'correct_final_emitted': bool(final_correct),
            'wrong_or_unextractable_final': not final_correct,
            'generation_never_observed_close': termination != 'eos',
            'correct_mentions': mentions, 'conclusion_mentions': conclusions,
            'possible_contradictions': contradictions,
            'limitations': 'Cue matching can mistake quantities, quoted plans, and intermediate totals for conclusions; manual calibration required. Non-EOS means closure unobserved, not infinite looping.'}
