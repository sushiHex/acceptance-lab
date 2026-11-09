# acceptance-lab

An agent that repairs OpenUSD scenes, and a judge that grades each repair against a written
contract.

This is a companion project to [stagewright](https://github.com/sushiHex/stagewright).
stagewright builds OpenUSD scenes from prompts. This project checks an agent's changes to them.

## The problem

A partner sends a scene and a request: "make this asset Y-up in meters, with /World as its default
prim." An agent can do that work. The hard part is to know that it did. A language model reports
success in the same confident words whether the scene is fixed, half-fixed, or damaged. And the
definition of "done" is usually in someone's head, so a reviewer must check each result by hand.

## The goal

Write "done" down once, as data, and use that same data for every decision about the agent:

1. **Grading.** A judge checks the repaired scene against each item in the contract.
2. **Guarding.** One preflight function decides which operations are permitted. The executor, the
   NeMo Guardrails rail and the judge all call it, so there is one definition of "allowed".
3. **Training.** The judge's verdict is the only label for fine-tuning and the only reward for
   reinforcement learning. The agent never grades itself.

## What I built

![The contract feeds the judge, the shared preflight and the training reward. Each task runs Nemotron Nano 4B on vLLM, a NeMo Guardrails rail, an MCP executor and an A2A judge.](docs/assets/acceptance-loop.svg)

- **The contract** is a JSON file per task. It lists the requirements, the properties that must not
  change, and the operations the agent may use.
- **The model** (Nemotron Nano 4B on vLLM) does not touch files. It sends one JSON proposal.
- **The executor** is an MCP server. It applies the proposal or refuses it, and writes a receipt.
- **The judge** is an A2A agent. It reads the result and returns "accepted" or "failed", with the
  name and status of each check.
- **61 tasks** in train, dev and test splits: axis, units, default prim, broken references,
  composite repairs, scenes that need no change, ambiguous requests the agent must refuse, forbidden
  operations, and a prompt injection. The test task types do not occur in training.
- **Training** used only the judge: SFT on the model's own samples that the judge accepted, then
  GRPO with the judge's verdict as the reward.

## Result

| Configuration | Accepted | Model success |
|---|---|---|
| Always abstain | 2/4 | |
| Scripted heuristic | 2/4 | |
| Nemotron Nano 4B on vLLM | 3/4 | 2/4 |
| + structured outputs | 3/4 | 2/4 |
| + SFT adapter | **4/4** | **4/4** |
| + GRPO adapter | 3/4 | 3/4 |

On the held-out test split, fine-tuning on judge-accepted samples took the model from 2 of 4 to 4 of
4. "Accepted" is the judge's grade. "Model success" is the training reward. It is lower when the
executor had to block an operation the model proposed.

## Three things the judge taught me

**The repair that did too much.** The first judge accepted a repair when every requirement and
preservation check passed. A repair that also changed a field the contract did not mention still
passed, so the training reward would pay the agent for edits nobody asked for. I added a check:
every operation must serve a requirement that the original scene fails. Then I added 15 tasks where
an allowed operation could "tidy" a field, and the correct answer is to leave that field alone.

**More training, less signal.** GRPO passed 17 of 17 dev tasks, one more than SFT. On test it did
worse: it proposed a malformed operation on the composite task, and the executor blocked it. After
five iterations almost every sampled group got the same score, so the updates had nothing left to
learn from. A reward only teaches while the model still fails some of the time.

**Change the contract, change the answer.** Task `demo-01` requires only `up_axis: Z`. The scene is
already Z-up, so the correct answer is no change. The SFT adapter refuses the task instead, and the
judge fails it. When `default_prim: /World` is added to the contract, the correct answer becomes a
repair. The same adapter sets the default prim, and all 12 checks pass. No code changed between the
two runs. Only the contract did.

## Stack

| Stage | Tools |
|---|---|
| Model | Nemotron Nano 4B v1.1, served by vLLM with LoRA adapters |
| Proposer | LangChain `ChatOpenAI`, optional structured outputs, fastembed rule retrieval |
| Guard | NeMo Guardrails output rail that runs the shared preflight |
| Execute | MCP server, one process per attempt, OpenUSD (`pxr` from the OpenUSD Exchange SDK package) |
| Judge | A2A agent with read-only access to the task and the attempt |
| Train | LoRA SFT with PEFT, and GRPO with the judge as the reward |

## Verification

- 12 tests. Each test names the code change that must make it fail. `tools/mutate.py` makes those
  11 changes and confirms that a test fails each time.
- All 60 fixtures have a pass control and a fail control. The judge accepts every pass control and
  rejects every fail control.
- The train, dev and test splits do not overlap. Dev selects the adapter. The results table uses
  test runs only.

Limits:

- The test split has 4 tasks, run once at temperature 0. The table shows a direction, not a
  statistically significant result.
- The check for unnecessary changes is approximate. The [guide](docs/guide.md#how-a-task-runs)
  gives the known gaps.
- All adapters still fail `demo-01` as shipped.

## Quickstart

Windows, Python 3.13. The baselines do not need a GPU.

```powershell
git clone https://github.com/sushiHex/acceptance-lab.git
cd acceptance-lab
python -m venv .venv
.venv\Scripts\pip install -r requirements-windows.lock

.venv\Scripts\python run.py baseline            # both baselines, then runs/table.html
.venv\Scripts\python -m unittest test_lab -v    # tests
.venv\Scripts\python tools/controls.py          # pass and fail control for each fixture
```

The model configurations need vLLM in WSL Ubuntu and a 24 GB GPU. The
[technical guide](docs/guide.md#setup) covers setup, evaluation, training and the file layout.

[MIT](LICENSE) for the original code and fixtures. Dependencies keep their own licenses.
