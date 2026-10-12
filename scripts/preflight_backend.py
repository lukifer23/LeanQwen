"""Leased short backend/cache/profile checks and exact historical P2 prefix replay."""

import argparse
import gc
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import psutil
import yaml
from mlx.utils import tree_flatten

from qwenlean.inference.mlx_backend import MLXBackend
from qwenlean.training.integrity import tensor_fingerprint
from qwenlean.utils.io import digest, read_jsonl, write_json
from qwenlean.utils.process_lock import model_process_lock


def cleanup_observation():
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    return {'mlx_active_bytes': mx.get_active_memory(),
            'available_ram_bytes': psutil.virtual_memory().available,
            'rss_bytes': psutil.Process().memory_info().rss}


def validate_telemetry(row):
    types = Counter(row['cache']['cache_types'])
    if types != {'KVCache': 6, 'ArraysCache': 18}:
        raise RuntimeError(f'Unexpected real cache layout: {types}')
    values = [row['cache']['attention_kv_bytes'], row['cache']['linear_state_bytes']]
    if any(v < 0 for v in values) or sum(values) > row['mlx_active_end_bytes']:
        raise RuntimeError('Cache byte accounting is invalid')
    if len(set(row['cache']['attention_offsets'])) != 1:
        raise RuntimeError('Full-attention cache offsets disagree')
    if not any(s['phase'] == 'prefill' for s in row['memory_samples']):
        raise RuntimeError('No prompt progress observations')
    if any(s['attention_kv_bytes'] < 0 or s['linear_state_bytes'] < 0 for s in row['memory_samples']):
        raise RuntimeError('Negative observed cache bytes')
    if row['backend_prefill_duration_s'] is None or row['backend_decode_duration_s'] is None:
        raise RuntimeError('Supported backend timing fields are missing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefix-tokens', type=int, default=128)
    args = parser.parse_args()
    if Path(args.output).exists() or args.prefix_tokens < 1:
        raise ValueError('New output and positive prefix token count required')
    cfg = yaml.safe_load(Path('configs/baseline.yaml').read_text())
    report = {'status': 'initializing', 'available_ram_before_load_bytes': psutil.virtual_memory().available,
              'historical_prefixes': [], 'short_profiles': [], 'test_used': False}
    write_json(args.output, report)
    try:
        with model_process_lock(), MLXBackend(cfg) as backend:
            before = tensor_fingerprint(tree_flatten(backend.model.parameters()))
            args_before = digest(asdict(backend.model.args))
            model_info_before = digest(backend.model_info)
            baseline = cleanup_observation()
            prompt = 'Compute 2 + 3. End your response with a final line containing only the integer answer.'
            for profile in ('mac-128k', 'mac-100k', 'native-262k'):
                short = {**cfg, 'enable_thinking': False, 'context_profile': profile,
                         'max_output_tokens': 64, 'sampling': {'temperature': 0, 'presence_penalty': 0, 'repetition_penalty': 1}}
                row = backend.generate(prompt, cfg['seed'], short)
                validate_telemetry(row)
                if row['termination_reason'] != 'eos':
                    raise RuntimeError('Short preflight failed to naturally terminate')
                clean = cleanup_observation()
                if clean['mlx_active_bytes'] > baseline['mlx_active_bytes'] + 16 * 2**20:
                    raise RuntimeError('Request-local cache cleanup left unexpected active memory')
                report['short_profiles'].append({'generation': row, 'after_cleanup': clean})
                write_json(args.output, report)
            first = report['short_profiles'][0]['generation']
            cap = backend.generate(prompt, cfg['seed'], {**short, 'context_profile': 'mac-128k', 'max_output_tokens': 1})
            validate_telemetry(cap)
            if cap['termination_reason'] != 'max_output_tokens' or len(cap['token_ids']) != 1:
                raise RuntimeError('One-token cap termination is incompatible')
            report['one_token_cap'] = cap
            archive = read_jsonl('reports/prompt_policy_records.jsonl')
            for family in ('arithmetic', 'algebra'):
                source = next(r for r in archive if r['family'] == family and r['prompt_policy'] == 'P2' and r['replicate'] == 0)
                if digest(source['model_identifier']) != model_info_before:
                    raise RuntimeError('Historical model identity differs')
                formatted = backend.tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': source['prompt']}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True)
                if backend.tokenizer.encode(formatted, add_special_tokens=False) != source['prompt_token_ids']:
                    raise RuntimeError('Historical template/tokenization differs')
                config = {**source['generation_parameters'], 'max_output_tokens': args.prefix_tokens}
                replay = backend.generate_tokens(source['prompt_token_ids'], source['seed'], config,
                                                 formatted_prompt=formatted, thinking=True)
                validate_telemetry(replay)
                match = replay['token_ids'] == source['token_ids'][:len(replay['token_ids'])]
                report['historical_prefixes'].append({'source_generation_id': source['generation_id'],
                    'source_archive_hash': digest(source), 'family': family, 'matched_prefix': match,
                    'checked_tokens': len(replay['token_ids']), 'generation': replay})
                write_json(args.output, report)
                if not match or replay['termination_reason'] not in {'eos', 'max_output_tokens'}:
                    raise RuntimeError('Historical prefix incompatible; no tail escalation allowed')
            report.update(all_loaded_tensors_unchanged=before == tensor_fingerprint(tree_flatten(backend.model.parameters())),
                live_model_args_unchanged=args_before == digest(asdict(backend.model.args)),
                tokenizer_model_metadata_unchanged=model_info_before == digest(backend.model_info),
                short_profile_ids_identical=all(r['generation']['token_ids'] == first['token_ids'] for r in report['short_profiles']),
                initial_loaded_fingerprint=before, baseline_memory=baseline,
                final_memory_after_request_cleanup=cleanup_observation())
            if not all(report[k] for k in ('all_loaded_tensors_unchanged', 'live_model_args_unchanged',
                                           'tokenizer_model_metadata_unchanged', 'short_profile_ids_identical')):
                raise RuntimeError('Profile invariance check failed')
        report.update(status='passed', full_262k_inference_verified=False)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        write_json(args.output, report)
        raise
    write_json(args.output, report)
    print('Short EOS/cap, 6 KV + 18 state caches, cleanup, all three profiles and two historical P2 prefixes passed.')


if __name__ == '__main__':
    main()
