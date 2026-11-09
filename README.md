# acceptance-lab

**A partner writes acceptance as data. The same data grades the agent, guards it, and trains it.**

An agent repairs OpenUSD scenes against a machine-readable contract. A request like "make this asset
Y-up in meters with /World as its default prim" arrives with its requirements, what must be preserved,
and which operations are allowed. The model proposes; an executor applies; a judge decides. Nothing
the model writes is trusted on its own word.

A companion to [stagewright](https://github.com/sushiHex/stagewright), which builds OpenUSD scenes
from prompts. This project asks the next question: how do you know the agent did it right?

![One acceptance contract feeds three jobs: an A2A judge grades, one shared preflight guards, and the verdict is the training reward. Each task runs Nemotron Nano 4B on vLLM, a NeMo Guardrails rail, an MCP executor and an A2A judge.](docs/assets/acceptance-loop.svg)

[Results](#results) · [Technical guide](docs/guide.md) · [Quickstart](#quickstart)

## Results

On the held-out test split (4 tasks: a composite repair, a no-op, an orientation ambiguity that must
be refused, and an axis conversion the contract forbids):

| Row | Accepted | Model success |
|---|---|---|
| always-abstain | 2/4 | |
| scripted heuristic | 2/4 | |
| Nemotron Nano 4B on vLLM | 3/4 | 2/4 |
| + structured outputs | 3/4 | 2/4 |
| + SFT adapter (judge-filtered samples, epoch chosen on dev) | **4/4** | **4/4** |
| + GRPO adapter (judge as reward, iteration chosen on dev) | 3/4 | 3/4 |

*Accepted* is the judge's grade. *Model success* is the training reward: it gives no credit when the
executor had to block an operation the model proposed.

- **SFT** is rejection sampling: the model's own samples, kept only where the judge accepts them.
  On dev (17 tasks), the SFT adapter is accepted on 16.
- **GRPO** reached 17 of 17 on dev, but it saturated the train split within five iterations:
  almost every sampled group was judged all alike, so the updates had no signal left. On test it
  proposed a malformed operation on the composite task, and the executor blocked it.

**The contract drives the behaviour.** `demo-01` asks only for `up_axis: Z`. The scene already meets
it, so the right answer is a no-op. The SFT adapter abstains, and the judge fails it. Adding
`default_prim: /World` to the contract turns the task into a repair, and the same adapter then sets
the default prim and passes all 12 checks. The model, executor and judge are unchanged; only the
data moved.

## The stack

| Stage | Tools |
|---|---|
| Model | Nemotron Nano 4B v1.1, served by vLLM with LoRA adapters |
| Proposer | LangChain `ChatOpenAI`, optional structured outputs, fastembed rule retrieval |
| Guard | NeMo Guardrails output rail running the shared preflight |
| Execute | MCP server, one process per attempt, OpenUSD (`pxr` from the OpenUSD Exchange SDK package) |
| Judge | A2A agent, read-only over the frozen task and the attempt |
| Train | PEFT LoRA SFT, and GRPO with the judge as the reward |

## Checked, with clear boundaries

- 12 boundary tests. Each names the mutation that breaks it, and `tools/mutate.py` confirms all
  11 mutations turn a test red.
- 60 of 60 fixtures behave: every task ships a pass control the judge accepts and a fail control it
  rejects.
- Train, dev and test are disjoint. The test families (composite, orientation ambiguity, axis
  refusal) never appear in training. Dev chooses what ships, and the table reads test runs only.

The boundaries:

- The test split is 4 tasks, run once at temperature 0, so the table shows direction, not
  statistical significance.
- The minimal-repair check is coarse (see the [guide](docs/guide.md#how-a-task-runs)).
- The no-op demo is still failed by every adapter.

## Quickstart

Windows, Python 3.13. The baselines need no GPU.

```powershell
git clone https://github.com/sushiHex/acceptance-lab.git
cd acceptance-lab
python -m venv .venv
.venv\Scripts\pip install -r requirements-windows.lock

.venv\Scripts\python run.py baseline            # both baselines, then runs/table.html
.venv\Scripts\python -m unittest test_lab -v    # boundary tests
.venv\Scripts\python tools/controls.py          # every fixture's pass and fail control
```

For the model rows, serving runs vLLM in WSL Ubuntu on a 24 GB card. See the
[technical guide](docs/guide.md#setup) for setup, evaluation, training and the full layout.

[MIT](LICENSE) for original code and fixtures. Dependencies retain their own terms.
