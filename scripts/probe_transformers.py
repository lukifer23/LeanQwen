"""Real MPS inference and PEFT adapter attachment, without optimizer updates."""

import json
import time
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwenlean.utils.process_lock import model_process_lock

_process_lock = model_process_lock()
_process_lock.__enter__()

meta = json.loads(Path("reports/environment.json").read_text())
report = {"model": meta["model"], "revision": meta["revision"]}
start = time.perf_counter()
try:
    model = AutoModelForCausalLM.from_pretrained(
        meta["model"], revision=meta["revision"], dtype=torch.bfloat16, attn_implementation="eager"
    ).to("mps")
    tokenizer = AutoTokenizer.from_pretrained(meta["model"], revision=meta["revision"])
    report["load_s"] = time.perf_counter() - start
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Compute 17 * 23."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    torch.mps.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    torch.mps.synchronize()
    ids = output[0, inputs.input_ids.shape[1] :].tolist()
    report.update(
        latency_s=time.perf_counter() - start,
        token_ids=ids,
        raw_output=tokenizer.decode(ids, skip_special_tokens=False),
        rss_bytes=psutil.Process().memory_info().rss,
        mps_allocated_bytes=torch.mps.current_allocated_memory(),
        mps_driver_bytes=torch.mps.driver_allocated_memory(),
    )
    model = get_peft_model(
        model,
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    report["peft_trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    report["status"] = "inference_and_adapter_attachment_ok; backward_not_tested"
except Exception as exc:
    import traceback

    report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
Path("reports/transformers_probe.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

_process_lock.__exit__(None, None, None)
