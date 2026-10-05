"""Record non-identifying machine/dependency information and inspect model metadata."""
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path

import psutil
import torch
from huggingface_hub import HfApi, hf_hub_download

out = Path('reports/environment.json')
model_id = 'Qwen/Qwen3.5-0.8B'
info = HfApi().model_info(model_id)
config_path = hf_hub_download(model_id, 'config.json', revision=info.sha)
config = json.loads(Path(config_path).read_text())
data = {
    'platform': platform.platform(), 'macos': platform.mac_ver()[0],
    'python': platform.python_version(), 'machine': platform.machine(),
    'chip': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
    'ram_bytes': psutil.virtual_memory().total,
    'available_ram_bytes_at_probe': psutil.virtual_memory().available,
    'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available(),
    'packages': {p: md.version(p) for p in ['torch', 'transformers', 'peft', 'trl', 'mlx', 'mlx-lm']},
    'model': model_id, 'revision': info.sha, 'model_config': config,
    'license': getattr(info.card_data, 'license', None),
}
out.write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps({k:v for k,v in data.items() if k!='model_config'}, indent=2))
