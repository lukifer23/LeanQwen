"""Read intact external reasoning and check arithmetic; never author solutions.

Annotation consistency is deliberately not a certificate of prose semantics.
Unrecognized or approximate claims remain unverified and require independent review.
"""

import ast
import operator
import re
from collections import Counter
from fractions import Fraction

from qwenlean.utils.io import digest

VERSION = 'external-reasoning-v1'
VERIFIER_VERSION = 'external-exact-annotations-v1'
GSM8K_REVISION = '3101c7d5072418e28b9008a6636bde82a006892c'
GSM8K_TRAIN_SHA256 = '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465'
GSM8K_TEST_SHA256 = '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'
GSM8K_LICENSE_SHA256 = '86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b'
GSM8K_SOURCE = 'gsm8k-original-human-train'
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def exact_value(expression):
    if len(expression) > 200 or not re.fullmatch(r'[0-9.,+*/()\s-]+', expression):
        raise ValueError('Unsupported numeric claim')
    # Only thousands-group commas may be removed; reject arbitrary comma expressions.
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', expression.strip())
    if ',' in text:
        raise ValueError('Unrecognized comma syntax')
    parsed = ast.parse(text, mode='eval')

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            # Source lexeme, not binary float, preserves exact decimals.
            return Fraction(ast.get_source_segment(text, node))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError('Unrecognized AST node')
    return evaluate(parsed.body)


def source_parts(record):
    if set(record) != {'question', 'answer'} or not all(isinstance(v, str) for v in record.values()):
        raise ValueError('Original GSM8K question/answer schema required')
    answer = record['answer']
    match = re.fullmatch(r'(?s)(.+)\n####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*', answer)
    if not match or answer.count('####') != 1:
        raise ValueError('Incomplete or malformed original final delimiter')
    reasoning, final = match.groups()
    expected = exact_value(final)
    return reasoning, final, expected


def verify_annotations(record):
    reasoning, final, expected = source_parts(record)
    found = list(re.finditer(r'<<([^<>]*)>>', reasoning))
    if reasoning.count('<<') != len(found) or reasoning.count('>>') != len(found):
        raise ValueError('Unclosed calculation annotation')
    verified, invalid, unverified = [], [], []
    for match in found:
        claim = match.group(1)
        parts = claim.split('=')
        try:
            if len(parts) < 2 or any(not p.strip() for p in parts):
                raise ValueError('Unfinished equation')
            values = [exact_value(p) for p in parts]
            entry = {'claim': claim, 'span': [match.start(), match.end()],
                     'values': [str(v) for v in values]}
            (verified if len(set(values)) == 1 else invalid).append(entry)
        except (ValueError, SyntaxError, ZeroDivisionError) as exc:
            unverified.append({'claim': claim, 'span': [match.start(), match.end()], 'reason': str(exc)})
    last_matches = bool(verified) and Fraction(verified[-1]['values'][-1]) == expected
    return {'verifier_version': VERIFIER_VERSION, 'method': 'Whitelisted Python AST with exact decimal Fraction; complete annotated equation chains',
            'verified_annotations': verified, 'invalid_annotations': invalid,
            'unverified_annotations': unverified, 'final_label': final,
            'last_verified_annotation_matches_final': last_matches,
            'annotation_consistent': bool(verified) and not invalid and not unverified and last_matches,
            'complete': True, 'prose_semantics_verified': False,
            'scope': 'Checks annotated arithmetic and final agreement only. Unannotated/contextual prose, relevance and productive effort require independent review.'}


def fingerprints(record):
    reasoning, _, _ = source_parts(record)
    def normalize(text):
        return ' '.join(text.casefold().split())
    # Number-free wording plus operation structure detects common superficial variants.
    numeric_free = re.sub(r'[-+]?\d+(?:[.,]\d+)*', '#', normalize(record['question']))
    operations = []
    for claim in re.findall(r'<<([^<>]*)>>', reasoning):
        expression = claim.split('=')[0]
        try:
            tree = ast.parse(expression, mode='eval')
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant):
                    node.value = 1
            operations.append(ast.dump(tree, include_attributes=False))
        except SyntaxError:
            operations.append('unrecognized')
    return {'source_record_hash': digest(record), 'task_identity_hash': digest(normalize(record['question'])),
            'semantic_structure_fingerprint': digest({'numeric_free_problem': numeric_free, 'operation_shapes': operations}),
            'numeric_free_prompt_hash': digest(numeric_free)}


def efficiency_flags(record):
    reasoning, _, _ = source_parts(record)
    sentences = [re.sub(r'\s+', ' ', line.strip()).casefold()
                 for line in re.split(r'\n|(?<=[.!?])\s+', reasoning) if line.strip()]
    repeated = [s for s, count in Counter(sentences).items() if count > 1 and len(s.split()) >= 5]
    flags = []
    if repeated:
        flags.append('repeated_complete_sentence_or_line')
    if re.search(r'\b(?:wait|reconsider|rethink|start over|formatting|final line|boxed answer)\b', reasoning, re.I):
        flags.append('reconsideration_or_format_cue_needs_review')
    if ' '.join(record['question'].casefold().split()) in ' '.join(reasoning.casefold().split()):
        flags.append('complete_problem_restatement')
    return {'flags': flags, 'repeated_units': repeated,
            'method': 'Conservative deterministic flags only; absence is not a semantic-quality certificate'}


def candidate_record(raw, index):
    reasoning, final, expected = source_parts(raw)
    identity = f'gsm8k-human-train-{index:05d}-{digest(raw)[:16]}'
    return {'task_id': identity, 'sample_id': identity, 'dataset': VERSION, 'split': 'train',
            'source_split': 'train', 'source_index': index, 'source_record': raw,
            'body': raw['question'], 'prompt': raw['question'], 'source_reasoning': reasoning,
            'source_final': final, 'expected': str(expected.numerator) if expected.denominator == 1 else str(expected),
            'family': 'external_math_word_problem', 'difficulty': 'unrated_human_source',
            'structural_complexity': len(re.findall(r'<<', reasoning)),
            **fingerprints(raw), 'verification': verify_annotations(raw),
            'efficiency_screen': efficiency_flags(raw),
            'provenance': {'source_id': identity, 'dataset_source': GSM8K_SOURCE,
                'dataset_revision': GSM8K_REVISION, 'source_url': f'https://github.com/openai/grade-school-math/tree/{GSM8K_REVISION}',
                'publisher': 'OpenAI / human writers; Surge AI data collection',
                'originating_author': 'human_problem_writer', 'teacher_model': None,
                'prompt_origin': 'licensed_dataset', 'response_origin': 'licensed_external_reasoning',
                'license': 'MIT', 'training_permitted': True,
                'upstream_model_output_terms': 'Not applicable: original contractor-written answers; Socratic and model-solution files excluded'}}
