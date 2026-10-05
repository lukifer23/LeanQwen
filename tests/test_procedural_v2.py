"""Independent exact checks of generated tasks and structural holdouts."""

import itertools
from fractions import Fraction

import pytest

from qwenlean.datasets.procedural_v2 import (
    DIFFICULTIES,
    FAMILIES,
    SPLITS,
    check_structural_contamination,
    evaluate_ast,
    generate,
)
from qwenlean.datasets.prompts import render_task


def independent_expression(node, values=None):
    if isinstance(node, int):
        return Fraction(node)
    op, *args = node
    if op == 'var':
        return values[args[0]]
    if op == 'NOT':
        return not independent_expression(args[0], values)
    a, b = [independent_expression(v, values) for v in args]
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        return a / b
    return {'AND': bool(a and b), 'OR': bool(a or b), 'XOR': bool(a) != bool(b)}[op]


@pytest.mark.parametrize('split', SPLITS)
def test_generated_ground_truth_and_difficulty(split):
    rows = generate(split, 2)
    assert rows == generate(split, 2)
    assert len(rows) == len(FAMILIES) * len(DIFFICULTIES) * 2
    assert all('FINAL:' not in r['body'] and r['body'] == r['prompt'] for r in rows)
    for r in rows:
        f, v = r['facts'], int(r['expected'])
        if r['family'] in {'arithmetic', 'boolean_logic'}:
            assert v == independent_expression(f['ast'], f.get('values'))
        elif r['family'] == 'algebra':
            x, a, b, c = v, f['a'], f['b'], f['c']
            if split == 'train':
                rhs = a * (x + b) - c
            elif split == 'dev':
                rhs = a * x - b * (x - c)
            else:
                rhs = (a + b) * x + c * x + DIFFICULTIES.index(r['difficulty'])
            for step in range(f['wrappers']):
                rhs = 2 * (rhs + step + 1)
            assert rhs == f['rhs']
        elif r['family'] == 'state_machine':
            state = f['initial'] % f['modulus']
            for op, n in f['ops']:
                state = (state + n if op == 'add' else state - n if op == 'subtract'
                         else state * n) % f['modulus']
            assert state == v
        elif r['family'] == 'ordering':
            assert f['order'][f['position']-1] == v
            assert all(i < j for i, j in f['edges'])
        elif r['family'] == 'counting':
            sums = [sum(x) for x in itertools.combinations(range(1, f['n']+1), f['k'])]
            assert v == sum((s % 3 == 0 if '3' in f['restriction'] else
                             s % 2 == int('odd' in f['restriction'])) for s in sums)
        elif r['family'] == 'word_problem':
            v -= sum(f['fees'])
            gross = f['boxes'] * f['items'] * f['price']
            net = gross * (100-f['discount']) // 100
            if split == 'train':
                assert v == net
            elif split == 'dev':
                assert v == net + f['price']//100
            else:
                assert v == gross - f['boxes'] * f['price'] * (100-f['discount'])//100
        else:
            seq = f['input'][:]
            for op in f['ops']:
                seq = (list(reversed(seq)) if op == 'reverse' else sorted(seq) if op == 'sort'
                       else seq[1:] + seq[:1] if op == 'rotate_left' else [-x for x in seq])
            assert seq[f['position']-1] == v
        assert render_task(r, 'P0')['task_id'] == r['task_id']


def test_structural_holdout_and_detected_leak():
    pools = {s: generate(s, 2) for s in SPLITS}
    assert check_structural_contamination(pools)['cross_split_structure_collisions'] == 0
    pools['dev'][0]['structural_signature'] = pools['train'][0]['structural_signature']
    with pytest.raises(ValueError, match='structural'):
        check_structural_contamination(pools)


def test_ast_rejects_unknown_operations():
    with pytest.raises(ValueError):
        evaluate_ast(['exec', 1, 2])
