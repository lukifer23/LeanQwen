"""Real unquantized upstream-weight inference, preserving token IDs and template."""
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import psutil
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

meta = json.loads(Path('reports/environment.json').read_text())
path = snapshot_download(meta['model'], revision=meta['revision'],
    allow_patterns=['*.json', '*.jinja', '*.safetensors', 'tokenizer*', '*.txt'])
print('Snapshot:', path, flush=True)
start = time.perf_counter()
model, tokenizer = load(path)
mx.eval(model.parameters())
print('Loaded in', time.perf_counter()-start, flush=True)
records = []
for thinking in [False, True]:
    mx.random.seed(42)
    template = tokenizer.apply_chat_template(
        [{'role':'user','content':'Compute 17 * 23. End your response with FINAL: followed by just the integer.'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    print('Template ending:', repr(template[-200:]), flush=True)
    responses = []
    start = time.perf_counter()
    for r in stream_generate(model, tokenizer, template, max_tokens=512,
            sampler=make_sampler(temp=1.0, top_p=.95, top_k=20)):
        responses.append(r)
    elapsed = time.perf_counter()-start
    raw = ''.join(r.text for r in responses)
    record = {'enable_thinking':thinking, 'prompt_template':template,
        'token_ids':[r.token for r in responses], 'raw_output':raw, 'latency_s':elapsed,
        'last_response':{k:v for k,v in vars(responses[-1]).items() if k!='logprobs'}, 'eos_token_ids':list(tokenizer.eos_token_ids),
        'mlx_peak_bytes':mx.get_peak_memory(), 'rss_bytes':psutil.Process().memory_info().rss,
        'model_revision':meta['revision'],
        'template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
    records.append(record)
    print(raw, flush=True)
    print('Elapsed', elapsed, 'Tokens', len(responses), flush=True)
Path('reports/smoke_generations.json').write_text(json.dumps(records, indent=2)+'\n')
