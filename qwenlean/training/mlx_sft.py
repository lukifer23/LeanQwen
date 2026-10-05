"""Real masked LoRA SFT, optimizer steps, adapter save/reload, bounded memory."""

import gc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import psutil
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters

from qwenlean.datasets.prompts import render_task
from qwenlean.inference.mlx_backend import MLXBackend
from qwenlean.inference.parsing import parse_tokens
from qwenlean.scoring.exact import score
from qwenlean.training.quality import (
    supervision_range,
    validate_parent_sources,
    validate_training_pool,
)
from qwenlean.utils.environment import environment_manifest
from qwenlean.utils.io import digest, read_jsonl, write_json, write_jsonl
from qwenlean.utils.process_lock import model_process_lock


def parameter_hash(parameters):
    result = hashlib.sha256()
    for name, value in tree_flatten(parameters):
        result.update(name.encode())
        result.update(np.asarray(value.astype(mx.float32)).tobytes())
    return result.hexdigest()


def attach_adapters(model, cfg):
    model.freeze()
    expected = []
    first = len(model.layers)-cfg['num_layers']
    for i, layer in enumerate(model.layers[first:], start=first):
        keys = ['linear_attn.in_proj_qkv'] if layer.is_linear else ['self_attn.q_proj', 'self_attn.v_proj']
        expected += [(i, key) for key in keys]
    linear_to_lora_layers(model, cfg['num_layers'], cfg['lora_parameters'])
    leaves = tree_flatten(model.trainable_parameters())
    if len(leaves) != 2*len(expected) or any(not n.endswith(('lora_a', 'lora_b')) for n, _ in leaves):
        raise RuntimeError('Adapter targeting mismatch; refuse generic fallback')
    return {'target_layer_modules': expected, 'parameter_names': [n for n, _ in leaves],
            'trainable_parameters': sum(p.size for _, p in leaves),
            'rationale': 'Full-attention q/v and gated-delta combined qkv; MLP, embeddings, recurrence gates and base tensors frozen.'}


def train(config, dataset, quality_manifest, output):
    rows = read_jsonl(dataset)
    quality = json.loads(Path(quality_manifest).read_text())
    validate_training_pool(rows, quality, config)
    validate_parent_sources(rows, quality)
    output = Path(output)
    if output.exists():
        raise ValueError('Refuse existing adapter output; optimizer resume is not implemented')
    with model_process_lock():
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'config.json', config)
        write_json(output/'quality_manifest.json', quality)
        write_json(output/'environment.json', environment_manifest())
        write_json(output/'completion.json', {'status': 'initializing'})
        try:
            return _train(config, rows, output)
        except BaseException as exc:
            write_json(output/'completion.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
            raise


def _train(cfg, rows, output):
    mx.random.seed(cfg['seed'])
    rng = random.Random(cfg['seed'])
    backend = MLXBackend(cfg)
    for row in rows:
        formatted = backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        if backend.tokenizer.encode(formatted, add_special_tokens=False) != row["prompt_token_ids"]:
            raise ValueError("Saved prompt IDs differ from official thinking template")
        if row["target_token_ids"][-1] not in backend.eos_ids:
            raise ValueError("Training target lacks a preserved Qwen EOS token")
    model = backend.model
    targets = attach_adapters(model, cfg)
    model.train()  # Native differentiable gated-delta path; inference kernels have no VJP.
    mx.eval(model.parameters())
    before_hash = parameter_hash(model.trainable_parameters())
    initial = {n: mx.array(p) for n, p in tree_flatten(model.trainable_parameters())}
    mx.eval(initial)
    base = {n: p for n, p in tree_flatten(model.parameters()) if '.linear.weight' in n}
    base_hash = parameter_hash(base)
    optimizer = optim.AdamW(learning_rate=cfg['learning_rate'], weight_decay=cfg.get('weight_decay', .01))
    mx.reset_peak_memory()
    process, rss_peak = psutil.Process(), psutil.Process().memory_info().rss
    order, position = list(range(len(rows))), 0
    rng.shuffle(order)

    def loss_fn(model, tokens, begin, end):
        logits = model(tokens[:, :-1])[:, begin:end, :].astype(mx.float32)
        return nn.losses.cross_entropy(logits, tokens[:, 1:][:, begin:end], reduction='mean')

    value_grad = nn.value_and_grad(model, loss_fn)
    logs, training_start = [], time.perf_counter()
    with (output/'loss.jsonl').open('w') as handle:
        for step in range(1, cfg['steps']+1):
            start = time.perf_counter()
            accumulator, losses, supervised = None, [], 0
            for _ in range(cfg['gradient_accumulation']):
                r = rows[order[position]]
                position += 1
                if position == len(order):
                    position = 0
                    rng.shuffle(order)
                tokens = mx.array([r['prompt_token_ids'] + r['target_token_ids']], dtype=mx.int32)
                begin, end = supervision_range(r['prompt_token_ids'], r['target_token_ids'])
                loss, gradients = value_grad(model, tokens, begin, end)
                mx.eval(loss, gradients)
                if not bool(mx.isfinite(loss).item()) or not all(bool(mx.all(mx.isfinite(v)).item()) for _, v in tree_flatten(gradients)):
                    raise RuntimeError('Non-finite training loss/gradients')
                accumulator = gradients if accumulator is None else tree_map(lambda a, b: a+b, accumulator, gradients)
                losses.append(float(loss.item()))
                supervised += len(r['target_token_ids'])
                rss_peak = max(rss_peak, process.memory_info().rss)
            averaged = tree_map(lambda g: g/cfg['gradient_accumulation'], accumulator)
            optimizer.update(model, averaged)
            mx.eval(model.trainable_parameters(), optimizer.state)
            mx.synchronize()
            elapsed = time.perf_counter()-start
            log = {'step': step, 'loss': sum(losses)/len(losses), 'optimizer_steps': step,
                   'step_s': elapsed, 'supervised_tokens': supervised,
                   'supervised_tokens_per_second': supervised/elapsed,
                   'mlx_peak_bytes': mx.get_peak_memory(), 'rss_peak_observed_bytes': rss_peak}
            logs.append(log)
            handle.write(json.dumps(log)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
            print(f"SFT step {step}/{cfg['steps']} loss={log['loss']:.5f} {elapsed:.2f}s peak={log['mlx_peak_bytes']/1e9:.2f}GB", flush=True)
            if step % cfg['checkpoint_every'] == 0 or step == cfg['steps']:
                mx.save_safetensors(str(output/f'step-{step:06d}.safetensors'), dict(tree_flatten(model.trainable_parameters())))
            del gradients, accumulator, averaged, loss, tokens
            mx.clear_cache()
    training_time = time.perf_counter()-training_start
    leaves = dict(tree_flatten(model.trainable_parameters()))
    changed = [n for n, p in leaves.items() if bool(mx.any(p != initial[n]).item())]
    after_hash = parameter_hash(model.trainable_parameters())
    if not changed or before_hash == after_hash:
        raise RuntimeError('Optimizer did not change adapter weights')
    if parameter_hash(base) != base_hash:
        raise RuntimeError('Selected frozen base weights changed')
    mx.save_safetensors(str(output/'adapters.safetensors'), leaves)
    write_json(output/'adapter_config.json', {'fine_tune_type': 'lora', 'num_layers': cfg['num_layers'],
               'lora_parameters': cfg['lora_parameters'], 'base_model': cfg['model'], 'revision': cfg['revision']})
    info = backend.model_info
    del leaves, initial, base, value_grad, optimizer, model, backend
    gc.collect()
    mx.clear_cache()
    reloaded = MLXBackend(cfg)
    load_adapters(reloaded.model, output)
    reloaded.model.eval()
    mx.eval(reloaded.model.parameters())
    reload_leaves = {n: p for n, p in tree_flatten(reloaded.model.parameters()) if n.endswith(('lora_a', 'lora_b'))}
    reload_hash = parameter_hash(reload_leaves)
    adapter_file_hash = hashlib.sha256((output/'adapters.safetensors').read_bytes()).hexdigest()
    reloaded.model_info.update(adapter_sha256=adapter_file_hash,
        weights_identifier=info['weights_identifier'] + ':adapter:' + adapter_file_hash)
    if reload_hash != after_hash:
        raise RuntimeError('Reloaded adapter tensor hash differs')
    smoke_tasks, smoke_rows = read_jsonl('data/splits/procedural-v2/dev.jsonl'), []
    for family in ('arithmetic', 'state_machine'):
        source = next(t for t in smoke_tasks if t['family'] == family and t['difficulty'] == 'trivial')
        t = render_task(source, cfg['prompt_policy'])
        gcfg = {**cfg, 'enable_thinking': True, 'max_output_tokens': 512, 'loop_guard': {'enabled': False}}
        g = reloaded.generate(t['prompt'], cfg['seed'], gcfg)
        p = parse_tokens(g['token_ids'], reloaded.tokenizer, True)
        fields = asdict(p)
        fields.pop('reasoning_ids')
        smoke_rows.append({**t, **g, **fields, **score(p.final, t['expected'], cfg['prompt_policy']),
                           'pipeline_sanity_only': True, 'adapter_hash': after_hash,
                           'provenance': {**t['provenance'], 'response_origin': 'qwen_self', 'training_permitted': False}})
    write_jsonl(output/'dev-smoke.jsonl', smoke_rows)
    report = {'status': 'optimizer_save_reload_generate_ok', 'optimizer_steps': cfg['steps'],
              'config': cfg, **targets, 'examples': len(rows), 'dataset_hash': digest(rows),
              'sequence_lengths': [len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in rows],
              'prompt_masked': True, 'loss_history': logs, 'training_s': training_time,
              'mlx_peak_bytes': max(r['mlx_peak_bytes'] for r in logs), 'rss_peak_observed_bytes': rss_peak,
              'adapter_hash_before': before_hash, 'adapter_hash_after': after_hash,
              'adapter_hash_reloaded': reload_hash, 'changed_parameter_names': changed,
              'selected_frozen_base_hash_unchanged': True, 'base_model_context': info['context_window_tokens'],
              'output': str(output), 'test_evaluated': False,
              'dev_smoke_correct': sum(r['correct'] for r in smoke_rows),
              'dev_smoke_count': len(smoke_rows), 'improvement_claim': False}
    write_json(output/'smoke_report.json', report)
    write_json(output/'completion.json', {'status': 'complete', 'optimizer_steps': cfg['steps']})
    return report
