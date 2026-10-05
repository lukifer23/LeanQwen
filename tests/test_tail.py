import pytest

from qwenlean.evaluation.tail import check_prefix, survival_bounds


def row(task, ids, cap, eos=False):
    return {'task_id': task, 'generation_id': f'{task}-{cap}', 'seed': 4, 'prompt': 'compute',
            'model_identifier': {'revision': 'abc'}, 'token_ids': ids,
            'generation_parameters': {'max_output_tokens': cap, 'sampling': {'temperature': 1}},
            'total_output_tokens': len(ids), 'reasoning_tokens': len(ids) - int(eos),
            'termination_reason': 'eos' if eos else 'max_output_tokens'}


def test_exact_prefix_and_invariant_mismatch():
    a, b = row('a', [1, 2], 2), row('a', [1, 2, 3, 4], 4)
    assert check_prefix(a, b)['exact_prefix']
    b['token_ids'][1] = 5
    assert check_prefix(a, b)['first_mismatch'] == 1
    b['seed'] = 9
    with pytest.raises(ValueError):
        check_prefix(a, b)


def test_survival_does_not_treat_caps_as_completed():
    rows = [row('a', [1, 2], 2, True), row('b', [1, 2, 3, 4], 4)]
    s = survival_bounds(rows, (2, 4, 8))
    assert s['right_censored'] == 1
    assert s['natural_completed_reasoning_lengths'] == [1]
    assert s['horizons'][2]['survival_lower_bound'] == 0
    assert s['horizons'][2]['survival_upper_bound'] == .5
    assert s['horizons'][1]['known_still_generating'] == 1
