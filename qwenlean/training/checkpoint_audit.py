"""CPU byte-level audit of small float32 LoRA safetensors, without MLX/model loading."""

import hashlib
import json
import math
import struct
from pathlib import Path


def audit_lora_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 8*1024**2:
        raise ValueError('Missing/oversized small LoRA checkpoint')
    content = path.read_bytes()
    if len(content) < 8:
        raise ValueError('Incomplete safetensors header')
    header_size = struct.unpack('<Q', content[:8])[0]
    if not 2 <= header_size <= min(1024**2, len(content)-8):
        raise ValueError('Invalid safetensors header extent')
    header = json.loads(content[8:8+header_size])
    payload, fingerprint, extents, parameters, tensors = content[8+header_size:], hashlib.sha256(), [], 0, []
    for name, item in sorted(header.items()):
        if name == '__metadata__':
            continue
        if not name.endswith(('lora_a', 'lora_b')) or item['dtype'] != 'F32':
            raise ValueError('Expected adapter-only float32 tensors')
        shape = item['shape']
        if not shape or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError('Invalid adapter shape')
        begin, end = item['data_offsets']
        size = math.prod(shape)
        if not 0 <= begin < end <= len(payload) or end-begin != size*4:
            raise ValueError('Incomplete or invalid tensor data extent')
        raw = payload[begin:end]
        if not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)):
            raise ValueError('Non-finite adapter parameter')
        # Same sorted name/shape/dtype/raw-byte contract used by the live MLX
        # fingerprint, independently reconstructed from the file's byte format.
        fingerprint.update(json.dumps([name, shape, 'mlx.core.float32'], separators=(',', ':')).encode()+b'\0')
        fingerprint.update(raw)
        extents.append((begin, end))
        parameters += size
        tensors.append({'name': name, 'shape': shape, 'dtype': item['dtype'], 'data_bytes': len(raw)})
    cursor = 0
    for begin, end in sorted(extents):
        if begin != cursor:
            raise ValueError('Overlapping or missing safetensors payload bytes')
        cursor = end
    if not tensors or cursor != len(payload):
        raise ValueError('Empty or trailing/missing checkpoint data')
    return {'path': str(path), 'file_bytes': len(content), 'file_sha256': hashlib.sha256(content).hexdigest(),
            'header_bytes': header_size, 'tensor_count': len(tensors), 'parameters': parameters,
            'tensor_fingerprint': fingerprint.hexdigest(), 'all_parameters_finite': True,
            'tensors': tensors, 'method': 'cpu-safetensors-f32-byte-audit-v1', 'gpu_used': False}


def require_approved_adapter(path, config):
    """Reject quarantine/partial artifacts before even loading the base model."""
    path = Path(path)
    if (path/'quarantine.json').exists():
        raise ValueError('Quarantined unsafe checkpoint cannot be loaded as a LeanQwen adapter')
    try:
        completion = json.loads((path/'completion.json').read_text())
        report = json.loads((path/'smoke_report.json').read_text())
        meta = json.loads((path/'adapter_config.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError('Complete approved adapter/integrity/reload evidence required') from exc
    if (completion.get('status') != 'complete'
        or type(completion.get('optimizer_steps')) is not int or completion['optimizer_steps'] < 1
        or completion.get('optimizer_steps') != report.get('optimizer_steps')
        or report.get('status') != 'optimizer_save_reload_generate_ok'
        or report.get('all_frozen_base_unchanged') is not True or report.get('reload_logits_exact') is not True
        or meta.get('base_model') != config['model'] or meta.get('revision') != config['revision']):
        raise ValueError('Adapter training did not pass complete technical acceptance')
    from qwenlean.training.external_contract import authorize_training_manifest
    quality = json.loads((path/'quality_manifest.json').read_text())
    authorize_training_manifest(quality, config)
    if (quality['dataset_hash'] != report.get('dataset_hash')
        or quality['model_revision'] != config['revision']
        or not report.get('frozen_base_before')
        or report['frozen_base_before'] != report.get('frozen_base_after')
        or report['frozen_base_before'] != report.get('frozen_base_reloaded')):
        raise ValueError('Adapter source/base integrity certificate differs')
    audited = audit_lora_file(path/'adapters.safetensors')
    if audited['tensor_fingerprint'] != report.get('adapter_hash_after') or audited['tensor_fingerprint'] != report.get('adapter_hash_reloaded'):
        raise ValueError('Saved adapter differs from the integrity/reload-approved tensors')
    return audited
