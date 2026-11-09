"""sft_train.py: LoRA SFT on train/sft.jsonl (inside WSL, GPU free, vLLM stopped).

    /opt/vllm/bin/python tools/sft_train.py --out /root/adapters/sft

Loss is on the assistant turn only; the prompt is the served chat template with the generation
prompt, so training and vLLM see the same tokens. Every epoch is saved (OUT/epoch-N) because
the epoch is chosen on the dev split, never on train loss. Writes train/adapter-manifest.json
beside the data: dataset hash, base model, hyperparameters, per-epoch loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "train" / "sft.jsonl"
BASE = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
HP = {"rank": 16, "alpha": 32, "dropout": 0.05, "lr": 2e-4, "epochs": 4, "grad_accum": 4, "seed": 0,
      "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def encode(tok, messages: list[dict], eos: bool = True) -> tuple[list[int], list[int]]:
    """Token ids and labels: -100 over the prompt, the assistant turn (plus eos) as the target."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(messages[-1]["content"], add_special_tokens=False)["input_ids"] + ([tok.eos_token_id] if eos else [])
    return p + c, [-100] * len(p) + c


def load(adapter: str | None = None, trainable: bool = True):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if adapter:
        from peft import PeftModel
        return tok, PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return tok, get_peft_model(model, LoraConfig(r=HP["rank"], lora_alpha=HP["alpha"], lora_dropout=HP["dropout"],
                                                 target_modules=HP["targets"], task_type="CAUSAL_LM"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    random.seed(HP["seed"])
    torch.manual_seed(HP["seed"])
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    tok, model = load()
    data = [encode(tok, r["messages"]) for r in rows]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=HP["lr"])
    model.train()
    losses = []
    for epoch in range(HP["epochs"]):
        order = random.sample(range(len(data)), len(data))
        total = 0.0
        for i, j in enumerate(order):
            ids, labels = (torch.tensor([x], device="cuda") for x in data[j])
            loss = model(input_ids=ids, labels=labels).loss
            (loss / HP["grad_accum"]).backward()
            total += loss.item()
            if (i + 1) % HP["grad_accum"] == 0 or i == len(order) - 1:
                opt.step()
                opt.zero_grad()
        losses.append(round(total / len(data), 4))
        model.save_pretrained(f"{args.out}/epoch-{epoch + 1}")
        print(f"epoch {epoch + 1}: mean loss {losses[-1]}")
    manifest = {"base_model": BASE, "data": "train/sft.jsonl", "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
                "examples": len(rows), "max_tokens": max(len(d[0]) for d in data), "hyperparameters": HP,
                "epoch_loss": losses, "adapter_dir": args.out, "selected_epoch": None}
    (ROOT / "train" / "adapter-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
