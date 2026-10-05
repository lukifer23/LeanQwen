"""Conservative natural-trajectory ranking; correctness always precedes length."""

import ast
import operator
import re
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

from qwenlean.datasets.provenance import validate_provenance

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def arithmetic_value(text):
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.operand))
        raise ValueError('Unsupported arithmetic syntax')
    if len(text) > 200:
        raise ValueError('Arithmetic claim too long')
    return visit(ast.parse(text.strip(), mode='eval').body)


def intermediate_checks(row):
    text = row['reasoning']
    verified, invalid = [], []
    # Only complete line-local numeric equations. Symbolic equations, approximations,
    # chained equalities, and unparsed prose remain unverified, never marked valid.
    pattern = re.compile(r'^\s*([()\d\s+*/.\-]+)\s*=\s*(-?\d+(?:\.\d+)?)\s*[.;]?\s*$')
    for line in text.splitlines():
        m = pattern.match(line)
        if not m or not any(x in m.group(1).lstrip('+-') for x in '+-*/'):
            continue
        try:
            actual = arithmetic_value(m.group(1))
            stated = Fraction(m.group(2))
        except (ValueError, SyntaxError, ZeroDivisionError):
            continue
        (verified if actual == stated else invalid).append(line.strip())
    facts = row.get('facts', {})
    if 'history' in facts:
        for m in re.finditer(r'after\s+(?:operation|step)\s+(\d+)[^\n]{0,30}?state\s*(?:is|=|:)\s*(-?\d+)', text, re.I):
            n, v = int(m.group(1)), int(m.group(2))
            if 0 <= n < len(facts['history']):
                (verified if v == facts['history'][n] else invalid).append(m.group(0))
    return {'method': 'restricted_numeric_equations_and_indexed_states_v1',
            'verified_claims': verified, 'invalid_claims': invalid,
            'coverage': 'Only parsed claims checked. Unparsed reasoning is not certified.'}


def classify(row, semantic=None):
    validate_provenance(row, for_training=True)
    checks = intermediate_checks(row)
    rep = row['repetition']
    eos = row['termination_reason'] == 'eos' and row['parse_status'] == 'complete'
    loop = rep.get('loop_evidence') is not None
    if loop or row['termination_reason'] in {'max_output_tokens', 'runtime_loop_guard'}:
        tier = 'E'
    elif row['correct'] and not eos:
        tier = 'C'
    elif not row['correct']:
        tier = 'D'
    else:
        obvious = rep['repetition_density'] >= .15 or rep['lexical_redundancy']['density'] >= .15
        tier = 'B' if obvious else 'A'
    return {'tier': tier, 'intermediate_checks': checks, 'eligible_natural_target':
            tier in {'A', 'B'} and not checks['invalid_claims'],
            'semantic': semantic,
            'semantic_used_to_rank': False,  # Pilot calibration is not yet a validated training filter.
            'rule': 'Correct + closed + no exact loop + no detected invalid arithmetic; obvious lexical/literal redundancy defines A/B. Semantic similarity reported separately, length only breaks quality ties.'}


def select_natural(rows):
    groups = defaultdict(list)
    enriched = []
    for r in rows:
        quality = classify(r)
        enriched.append({**r, 'quality': quality})
        groups[r['task_id']].append(enriched[-1])
    selected = []
    for candidates in groups.values():
        eligible = [r for r in candidates if r['quality']['eligible_natural_target']]
        if eligible:
            # Recognized valid reasoning before length; does not reward the sheer
            # count of repeated verified equations. All selections still need review.
            best = min(eligible, key=lambda r: (
                r['quality']['tier'] != 'A',
                not bool(r['quality']['intermediate_checks']['verified_claims']),
                r['repetition']['repetition_density'] + r['repetition']['lexical_redundancy']['density'],
                r['reasoning_tokens'], r['generation_id']))
            selected.append(best)
    medians = {k: float(np.median([r['reasoning_tokens'] for r in rs])) for k, rs in groups.items()}
    summary = {'candidate_count': len(rows), 'problem_count': len(groups),
               'tier_counts': dict(Counter(r['quality']['tier'] for r in enriched)),
               'tasks_with_correct_candidate': sum(any(r['correct'] for r in rs) for rs in groups.values()),
               'tasks_with_clean_correct_candidate': len(selected),
               'coverage_clean': len(selected)/max(1, len(groups)),
               'selected_count': len(selected),
               'selected_reasoning_tokens': [r['reasoning_tokens'] for r in selected],
               'median_candidate_minus_selected_tokens': [medians[r['task_id']] - r['reasoning_tokens'] for r in selected],
               'natural_only': True, 'compression_applied': False,
               'selection_requires_manual_review': True}
    return enriched, selected, summary
