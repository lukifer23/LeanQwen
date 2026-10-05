import pytest

from qwenlean.generation.selection import arithmetic_value, intermediate_checks, select_natural


def candidate(gid, correct, tokens, reason='2 * 3 = 6', eos=True):
    return {'split': 'train', 'task_id': 't', 'generation_id': gid, 'correct': correct,
            'reasoning_tokens': tokens, 'reasoning': reason,
            'termination_reason': 'eos' if eos else 'max_output_tokens',
            'parse_status': 'complete' if eos else 'unclosed_thinking',
            'repetition': {'loop_evidence': None, 'repetition_density': 0,
                           'lexical_redundancy': {'density': 0}},
            'provenance': {'source_id': gid, 'prompt_origin': 'algorithmic',
                           'response_origin': 'qwen_self', 'license': 'Apache-2.0',
                           'training_permitted': True}}


def test_wrong_short_cannot_beat_correct_long_and_invalid_reasoning_rejected():
    rows = [candidate('wrong', False, 10), candidate('rigorous', True, 500),
            candidate('hallucinated', True, 20, '2 * 3 = 7'),
            candidate('truncated', True, 100, eos=False)]
    _, best, report = select_natural(rows)
    assert best[0]['generation_id'] == 'rigorous'
    assert report['tier_counts'] == {'D': 1, 'A': 2, 'C': 1}
    assert intermediate_checks(rows[2])['invalid_claims']
    rows[0]['split'] = 'dev'
    with pytest.raises(ValueError, match='TRAIN'):
        select_natural(rows)


def test_restricted_arithmetic_is_safe_and_exact():
    assert arithmetic_value('(3 + 5) / 2') == 4
    with pytest.raises(ValueError):
        arithmetic_value('__import__("os")')


def test_numeric_checks_handle_latex_without_certifying_symbolic_work():
    r = candidate('r', True, 100, r'* $6 \times 7 = 43$' + '\n' + r'$$6 \times 7 = 42$$' + '\n' + 'x + 2 = 9')
    c = intermediate_checks(r)
    assert len(c['invalid_claims']) == 1
    assert len(c['verified_claims']) == 1
