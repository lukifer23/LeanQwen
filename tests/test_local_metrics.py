import numpy as np

from qwenlean.metrics.conclusions import conclusion_states
from qwenlean.metrics.guard_replay import replay
from qwenlean.metrics.semantic import chunks, semantic_statistics


def test_semantic_metric_alignment_and_numeric_caveat():
    texts = ['The result is thirteen after all operations.',
             'The result is fourteen after all operations.', 'A distinct unrelated argument.']
    r = semantic_statistics(texts, np.array([[1, 0], [1, 0], [0, 1]], dtype=float))
    assert r['near_duplicate_chunks'] == 1
    assert r['pairs'][0]['cosine'] == 1
    assert len(chunks(' '.join(['word'] * 130))) == 2  # short tail excluded transparently


def test_conclusion_categories_do_not_treat_intermediate_as_cue():
    r = conclusion_states('2 + 3 = 5. Therefore the answer is 5.\nThe answer is 6.', '5', False, 'max_output_tokens')
    assert r['correct_numeric_intermediate_appears']
    assert r['correct_after_conclusion_cue']
    assert r['later_conflicting_numeric_conclusion_cue']
    assert not r['stable_correct_cue_no_later_conflicting_cue']
    assert r['generation_never_observed_close']


def test_guard_replay_respects_polling():
    ids = list(range(24))*3 + [999]*4
    r = replay(ids, {'min_period': 24, 'max_period': 24, 'repeats': 3}, check_every=8)
    assert r['trigger_reasoning_token'] == 72
    assert r['observed_reasoning_tokens_saved'] == 4
    assert replay(list(range(100)), {'min_period': 24, 'max_period': 24, 'repeats': 3}) is None


def test_guard_replay_preserves_control_polling_and_excludes_final():
    block = list(range(24))
    ids = [1000] + block*3 + [1001] + block*3 + [1002]
    # Opening token shifts global polling. A reasoning-only replay would fire
    # at 72, but the actual 8-token cadence misses the exact cycle before close.
    assert replay(ids, {'min_period':24, 'max_period':24, 'repeats':3},
                  check_every=8, opening=1000, closing=1001, eos_ids={1002}) is None
