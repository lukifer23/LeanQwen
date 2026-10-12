"""Small actual Metal tensors only; never load Qwen to test the hash primitive."""

import argparse
import traceback
from pathlib import Path

import mlx.core as mx
import psutil

from qwenlean.training.integrity import tensor_fingerprint
from qwenlean.utils.io import write_json
from qwenlean.utils.process_lock import model_process_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('Refuse existing preflight artifact')
    result = {'purpose': 'small_Metal_hash_primitive_only', 'full_model_loaded': False,
              'available_ram_start_bytes': psutil.virtual_memory().available}
    with model_process_lock():
        try:
            if not mx.metal.is_available():
                raise RuntimeError('Metal unavailable')
            copies = []
            with mx.stream(mx.gpu):
                a = mx.arange(64, dtype=mx.float32).reshape(8, 8).astype(mx.bfloat16)
                mx.eval(a)
                mx.synchronize()
                before = tensor_fingerprint([('metal_bf16', a)], chunk_elements=3,
                                            on_host_chunk=copies.append)
                repeated = tensor_fingerprint([('metal_bf16', a)], chunk_elements=7)
                changed = tensor_fingerprint([('metal_bf16', a + 1)], chunk_elements=3)
            checks = {'stable_hash': before == repeated, 'changed_value_changes_hash': before != changed,
                      'bounded_host_copy': max(copies) <= 3 * a.dtype.size,
                      'copied_all_bytes': sum(copies) == a.nbytes}
            result.update(status='passed' if all(checks.values()) else 'failed', checks=checks,
                          input_dtype=str(a.dtype), input_elements=a.size,
                          initial_fingerprint=before, repeated_fingerprint=repeated,
                          changed_fingerprint=changed, host_chunk_bytes=copies,
                          max_host_chunk_bytes=max(copies), metal_peak_bytes=mx.get_peak_memory())
            if not all(checks.values()):
                raise RuntimeError('Metal integrity preflight check failed')
            del a
            mx.synchronize()
            mx.clear_cache()
        except BaseException as exc:
            result.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc())
            write_json(args.output, result)
            raise
    write_json(args.output, result)
    print('Small Metal bfloat16 fingerprint preflight passed; no full model loaded.')


if __name__ == '__main__':
    main()
