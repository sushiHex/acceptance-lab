"""grpo_step.py: one GRPO policy update from a judged batch (inside WSL, GPU free, vLLM stopped).

    /opt/vllm/bin/python tools/grpo_step.py --adapter-in DIR --batch FILE --out DIR

The batch is what tools/grpo.py wrote: each row a train prompt, the policy's own sampled reply,
and its group-relative advantage (reward minus the group mean, the reward being the judge's
scene.model_success). Groups the judge scored all alike carry no signal and are not in it.
The objective is the clipped surrogate over reply tokens. The old log-probabilities are taken
from the adapter that sampled the batch, over the reply text retokenized (vLLM's sampled token
ids are not kept, so a reply that tokenizes differently is scored as its canonical tokens, and
a "stop" reply is closed with the tokenizer's eos). No KL term: the dev split, which picks the
iteration that ships, is the guard against drift.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_train import encode, load  # noqa: E402

HP = {"lr": 2e-5, "clip": 0.2, "minibatch": 16, "epochs": 1, "max_grad_norm": 1.0, "seed": 0}


def reply_logprobs(model, ids: list[int], labels: list[int]) -> torch.Tensor:
    n = sum(1 for x in labels if x != -100)
    x = torch.tensor([ids], device="cuda")
    logits = model(input_ids=x, logits_to_keep=n + 1).logits[0, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, x[0, -n:, None])[:, 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-in", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    random.seed(HP["seed"])
    torch.manual_seed(HP["seed"])
    rows = [json.loads(l) for l in Path(args.batch).read_text(encoding="utf-8").splitlines() if l.strip()]
    tok, model = load(args.adapter_in)
    data = [encode(tok, r["messages"] + [{"role": "assistant", "content": r["completion"]}], eos=r["finished"])
            for r in rows]
    model.train()                    # train mode keeps gradient checkpointing; dropout off so that
    for m in model.modules():        # old and new log-probabilities agree exactly at step 0
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    with torch.no_grad():
        old = [reply_logprobs(model, *d) for d in data]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=HP["lr"])
    state = Path(args.adapter_in) / "optimizer.pt"
    if state.exists():
        opt.load_state_dict(torch.load(state))
    stats = {"rows": len(rows), "steps": 0, "clipped": 0, "tokens": 0}
    for _ in range(HP["epochs"]):
        order = random.sample(range(len(rows)), len(rows))
        for s in range(0, len(order), HP["minibatch"]):
            mb = order[s:s + HP["minibatch"]]
            for j in mb:
                new = reply_logprobs(model, *data[j])
                ratio = torch.exp(new - old[j])
                a = rows[j]["advantage"]
                obj = torch.minimum(ratio * a, ratio.clamp(1 - HP["clip"], 1 + HP["clip"]) * a).mean()
                (-obj / len(mb)).backward()
                stats["clipped"] += int(((ratio - 1).abs() > HP["clip"]).sum())
                stats["tokens"] += ratio.numel()
            torch.nn.utils.clip_grad_norm_(params, HP["max_grad_norm"])
            opt.step()
            opt.zero_grad()
            stats["steps"] += 1
    model.save_pretrained(args.out)
    torch.save(opt.state_dict(), Path(args.out) / "optimizer.pt")
    print(json.dumps({"hyperparameters": HP, **stats}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
