"""Independent exact proof checks. No natural-language expression is certified.

Only the typed operation schema is accepted. Unknown operations, missing steps,
extra steps and forged states fail closed. Numeric prose heuristics are irrelevant.
"""

import ast
import itertools
from collections import Counter
from fractions import Fraction

from qwenlean.datasets.procedural_v2 import render_ast
from qwenlean.utils.io import digest

VERSION = 'independent-proof-verifier-1.0.0'


def require(condition):
    if not condition:
        raise ValueError('Invalid or incomplete structured derivation')


def arithmetic_expression(expression):
    """Independent Python parser with exact rational semantics; no eval."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return a+b
            if isinstance(node.op, ast.Sub):
                return a-b
            if isinstance(node.op, ast.Mult):
                return a*b
            if isinstance(node.op, ast.Div):
                return a/b
        raise ValueError('Unrecognized arithmetic AST')
    return visit(ast.parse(expression, mode='eval').body)


def logic_mask(node):
    """Truth table for every assignment, computed by 64-bit Boolean operations."""
    full = (1 << 64)-1
    op, *args = node
    if op == 'var':
        require(len(args) == 1 and type(args[0]) is int and 0 <= args[0] < 6)
        return sum(1 << row for row in range(64) if row & (1 << args[0]))
    if op == 'NOT':
        require(len(args) == 1)
        return full ^ logic_mask(args[0])
    require(op in {'AND', 'OR', 'XOR'} and len(args) == 2)
    a, b = logic_mask(args[0]), logic_mask(args[1])
    return a & b if op == 'AND' else a | b if op == 'OR' else a ^ b


def nodes_postorder(node):
    if isinstance(node, int) or node[0] == 'var':
        return []
    return [n for child in node[1:] for n in nodes_postorder(child)] + [node]


def affine(ops):
    a, b = Fraction(1), Fraction(0)
    for op, n in ops:
        if op == '+':
            b += n
        elif op == '-':
            b -= n
        elif op == '*':
            a *= n
            b *= n
        else:
            raise ValueError('Unsupported symbolic operation')
    require(a != 0)
    return a, b


def enumeration_table(n, k, m, start=1):
    table = []
    for size in range(k+1):
        counts = Counter(sum(c) % m for c in itertools.combinations(range(start, start+n), size))
        table.append([counts[r] for r in range(m)])
    return table


def verify_proof(family, spec, proof):
    require(set(proof) in ({'initial_state', 'steps', 'final_answer'},
                          {'initial_state', 'steps', 'final_answer', 'verification'}))
    require(proof['initial_state'] == spec and bool(proof['steps']))
    steps = proof['steps']
    require(all(set(step) == {'operation', 'before', 'after', 'context'} for step in steps))
    final = Fraction(proof['final_answer'])
    require(final.denominator == 1)
    method = ''
    if family in {'arithmetic', 'boolean_logic'}:
        nodes = nodes_postorder(spec['ast'])
        require(len(nodes) == len(steps))
        row = sum(int(v) << i for i, v in enumerate(spec.get('values', [])))

        def value(node):
            if family == 'arithmetic':
                return arithmetic_expression(render_ast(node))
            return Fraction((logic_mask(node) >> row) & 1)

        for node, step in zip(nodes, steps, strict=True):
            require(step['context'] == node and step['operation'] == node[0])
            require([Fraction(v) for v in step['before']] == [value(n) for n in node[1:]])
            require(Fraction(step['after']) == value(node))
        require(final == value(spec['ast']))
        method = 'Python AST / exact Fraction' if family == 'arithmetic' else '64-assignment Boolean truth tables'
    elif family == 'state_machine':
        require(len(steps) == len(spec['ops'])+1)
        m, raw = spec['modulus'], spec['initial']
        require(steps[0] == {'operation': 'reduce', 'before': raw, 'after': raw % m, 'context': m})
        for (op, n), step in zip(spec['ops'], steps[1:], strict=True):
            previous = raw % m
            if op == 'add':
                raw += n
                local = previous+n
            elif op == 'subtract':
                raw -= n
                local = previous-n
            elif op == 'multiply':
                raw *= n
                local = previous*n
            else:
                raise ValueError('Unknown modular operation')
            require(step == {'operation': op, 'before': previous, 'after': raw % m,
                             'context': {'operand': n, 'raw': local, 'modulus': m}})
        require(final == raw % m)
        method = 'Unreduced integer forward simulation and congruence at every prefix'
    elif family == 'sequence':
        require(len(steps) == len(spec['ops'])+1)
        current = spec['input'][:]
        for op, step in zip(spec['ops'], steps[:-1], strict=True):
            require(step['operation'] == op and step['before'] == current and step['context'] is None)
            after = step['after']
            require(isinstance(after, list) and len(after) == len(current))
            if op == 'sort':
                require(Counter(after) == Counter(current) and all(a <= b for a, b in zip(after, after[1:])))
            else:
                for i, v in enumerate(after):
                    if op == 'reverse':
                        require(v == current[len(current)-1-i])
                    elif op == 'rotate_left':
                        require(v == current[(i+1) % len(current)])
                    elif op == 'negate':
                        require(v+current[i] == 0)
                    else:
                        raise ValueError('Unknown sequence operation')
            current = after
        require(steps[-1] == {'operation': 'select', 'before': current, 'after': current[spec['position']], 'context': spec['position']})
        require(final == current[spec['position']])
        method = 'Index mapping and sorted-multiset invariants'
    elif family == 'ordering':
        nodes = spec['nodes']
        require(len(set(nodes)) == len(nodes) and len(steps) == len(nodes)+1)
        # Transitive closure establishes rank independently of Kahn removal.
        reach = {n: set() for n in nodes}
        for a, b in spec['edges']:
            require(a in reach and b in reach)
            reach[a].add(b)
        for mid in nodes:
            for start in nodes:
                if mid in reach[start]:
                    reach[start].update(reach[mid])
        require(all(n not in reach[n] for n in nodes))
        ranks = {n: sum(n in reach[p] for p in nodes) for n in nodes}
        require(set(ranks.values()) == set(range(len(nodes))))
        ordered = sorted(nodes, key=ranks.get)
        require(all(ranks[a] < ranks[b] for a, b in spec['edges']))
        for i, (step, node) in enumerate(zip(steps[:-1], ordered, strict=True)):
            require(step == {'operation': 'place', 'before': sorted(ordered[i:]),
                             'after': sorted(ordered[i+1:]), 'context': {'runner': node, 'position': i+1}})
        require(steps[-1] == {'operation': 'select', 'before': ordered,
                             'after': ordered[spec['position']], 'context': spec['position']})
        require(final == ordered[spec['position']])
        method = 'Transitive closure ranks and independent constraint checks'
    elif family == 'algebra':
        ops = spec['ops']
        a, b = affine(ops)
        x = (Fraction(spec['rhs'])-b)/a
        require(final == x and len(steps) == len(ops)+1)
        for i, step in enumerate(steps[:-1]):
            remaining = len(ops)-i-1
            op, n = ops[remaining]
            before_a, before_b = affine(ops[:remaining+1])
            after_a, after_b = affine(ops[:remaining])
            require(step['operation'] == 'undo_'+op)
            require(Fraction(step['before']) == before_a*x+before_b)
            require(Fraction(step['after']) == after_a*x+after_b)
            require(step['context'] == {'operand': n, 'remaining_ops': ops[:remaining]})
        # Separate explicit substitution through the original operators.
        substituted = x
        for op, n in ops:
            if op == '+':
                substituted += n
            elif op == '-':
                substituted -= n
            else:
                substituted *= n
        require(substituted == spec['rhs'])
        require(steps[-1]['operation'] == 'substitute' and Fraction(steps[-1]['before']) == x
                and Fraction(steps[-1]['after']) == spec['rhs'] and steps[-1]['context'] == ops)
        method = 'Exact affine symbolic coefficients at every remaining expression plus substitution'
    elif family == 'word_problem':
        require(len(steps) == 3*len(spec['baskets'])+len(spec['rates'])+len(spec['fees']))
        total = Fraction(0)
        pos = 0
        for b, i, p in spec['baskets']:
            claims = [(' * ', [b, i], b*i), (' * ', [b*i, p], b*i*p), (' + ', [total, b*i*p], total+b*i*p)]
            for operation, before, after in claims:
                step = steps[pos]
                require(step['operation'] == operation.strip() and step['context'] is None)
                require([Fraction(v) for v in step['before']] == before and Fraction(step['after']) == after)
                # Parsed exact arithmetic verifies each rendered numeric claim too.
                require(arithmetic_expression(operation.join(step['before'])) == Fraction(step['after']))
                pos += 1
            total += b*i*p
        for rate in spec['rates']:
            step = steps[pos]
            require(step['operation'] == 'discount' and step['context'] == rate and Fraction(step['before']) == total)
            # Check discount as removed amount rather than retained-percent solver formula.
            total -= total*Fraction(rate, 100)
            require(Fraction(step['after']) == total)
            pos += 1
        for fee in spec['fees']:
            step = steps[pos]
            require(step['operation'] == '+' and [Fraction(v) for v in step['before']] == [total, fee] and step['context'] is None)
            total += fee
            require(Fraction(step['after']) == total)
            pos += 1
        require(final == total)
        method = 'Exact cents/Fraction, parsed arithmetic and subtract-discount accounting'
    elif family == 'counting':
        n, k, m = (spec[x] for x in ('n', 'k', 'modulus'))
        require(1 <= k <= 4 and k <= n <= 20 and 2 <= m <= 5)
        require(len(steps) == n+1)
        previous = enumeration_table(0, k, m, spec['start'])
        for prefix, step in enumerate(steps[:-1], 1):
            table = enumeration_table(prefix, k, m, spec['start'])
            require(step == {'operation': 'include_or_exclude', 'before': previous,
                             'after': table, 'context': spec['start']+prefix-1})
            previous = table
        expected = previous[k][spec['residue']]
        require(steps[-1] == {'operation': 'select_count', 'before': previous, 'after': expected,
                             'context': {'size': k, 'residue': spec['residue']}})
        require(final == expected)
        method = 'Exhaustive combinations enumeration for every prefix/size/remainder cell'
    else:
        raise ValueError('Unrecognized family; cannot certify')
    return {'verified': True, 'verifier_version': VERSION, 'method': method,
            'step_count': len(steps), 'verified_step_indices': list(range(len(steps))),
            'proof_payload_hash': digest({k: proof[k] for k in ('initial_state', 'steps', 'final_answer')}),
            'independent_expected_answer': str(final.numerator)}
