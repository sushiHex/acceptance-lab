<div align="center">

# acceptance-lab

**A small model repairs OpenUSD (Open Universal Scene Description) scenes. A judge grades each repair against a written contract.**

[![License: MIT](https://img.shields.io/badge/license-MIT-3b4752?style=flat-square)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-3b4752?style=flat-square)](https://www.python.org/)
[![Model: Nemotron Nano 4B](https://img.shields.io/badge/model-Nemotron_Nano_4B-76b900?style=flat-square)](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1)
[![Guard: NeMo Guardrails](https://img.shields.io/badge/guard-NeMo_Guardrails-76b900?style=flat-square)](https://github.com/NVIDIA-NeMo/Guardrails)
[![Scenes: OpenUSD Exchange SDK](https://img.shields.io/badge/scenes-OpenUSD_Exchange_SDK-76b900?style=flat-square)](https://docs.omniverse.nvidia.com/usd/code-docs/usd-exchange-sdk/latest/index.html)

[Problem](#the-problem) · [What I built](#what-i-built) · [Results](#results) · [Lessons](#three-lessons) · [Quickstart](#quickstart) · [Technical guide](docs/guide.md)

</div>

> [!NOTE]
> **The judge is deterministic code, not a model.** The model never calls a tool or opens a file.
> This is a companion project to [**stagewright**](https://github.com/sushiHex/stagewright), which
> builds OpenUSD scenes from prompts. This project checks an agent's changes to them.

## The problem

A partner sends a scene and a request:

> *"Make this asset Y-up in meters, with `/World` as its default prim."*

An agent can do that work. **Knowing that it did is harder.** A language model reports success in
the same words whether the scene is fixed or damaged. And the definition of "done" is usually in
someone's head, so a reviewer checks each result by hand.

## The goal

Write "done" down as data. Use the same data to **grade**, to **guard** and to **train**. The
model's own report of success is never used.

## What I built

![Each step of a task reads the acceptance contract. Nemotron Nano 4B on vLLM proposes, NeMo Guardrails checks, an MCP executor applies or refuses, and an A2A judge grades. One shared preflight check guards steps 2 to 4. The judge's verdict and the executor's receipt go back to the model as training data.](docs/assets/acceptance-loop.svg)

| | Part | What it does |
|:--:|:--|:--|
| | **Contract** | One JSON file per task: the requirements, the properties that must not change, and the operations the agent may use. Each task also has an **expectation the model never sees**: the correct decision and the measurements the judge checks. |
| 1 | **Model** | [**Nemotron Nano 4B**](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1) on vLLM, through LangChain. It gets the request, the contract and the scene facts, and returns one JSON proposal. It gets one retry if the reply does not parse. |
| 2 | **Guard** | One preflight function decides whether an operation is permitted. A [**NeMo Guardrails**](https://github.com/NVIDIA-NeMo/Guardrails) output rail, the executor and the judge all call it. The rail makes no LLM (large language model) call. If it does not run, nothing executes. |
| 3 | **Executor** | An MCP (Model Context Protocol) server, one process per attempt. It edits scenes with OpenUSD from the [**OpenUSD Exchange SDK**](https://docs.omniverse.nvidia.com/usd/code-docs/usd-exchange-sdk/latest/index.html) package. It applies the whole proposal or blocks it, and writes a receipt. |
| 4 | **Judge** | An A2A (Agent2Agent protocol) agent that runs the checks the contract names. It returns `accepted` or `failed`, with each check. It never writes a file. |

**60 tasks** across train, dev and test, plus one demo task: axis, units, default prim, broken
references, scenes that need no change, ambiguous requests the model must refuse, forbidden
requests, a prompt injection and a composite repair. **Three task families occur only in test:**
the composite repair, an ambiguity about the axis, and a forbidden axis change.

**Training** uses the judge's verdict and the executor's receipt as the only signal. SFT (supervised fine-tuning) keeps the
model's samples that the judge accepted. GRPO (Group Relative Policy Optimization) uses the same
reading as its reward. The dev split
selects the adapter.

## Results

Held-out test split: 4 tasks, one run each at temperature 0.

| Configuration | Accepted | Model success |
|:--|:--:|:--:|
| Always abstain | 2/4 | |
| Scripted heuristic | 2/4 | |
| Nemotron Nano 4B on vLLM | 3/4 | 2/4 |
| &nbsp;&nbsp;+ structured outputs | 3/4 | 2/4 |
| &nbsp;&nbsp;**+ SFT adapter** | **4/4** | **4/4** |
| &nbsp;&nbsp;+ GRPO adapter | 3/4 | 3/4 |

<sub>**Accepted** is the judge's grade. **Model success** is the training reward: accepted, and no
operation blocked by the executor. Every row has its run file in [`results/`](results).</sub>

**What the rows show.** Always-abstain passes the two tasks that need a refusal. The scripted
heuristic passes the other two: one repair, and one scene that needs no change. The base model gets
those two right as well, but refuses neither: it edits on the ambiguous task, and on the forbidden
task it proposes the axis change, which is blocked. **The SFT adapter is the first configuration
to get all four right.**

> [!IMPORTANT]
> **Limits.** With 4 test tasks, one task moves a row by 25 points, so read the table as a smoke
> test. The check for unnecessary changes is coarse; the
> [technical guide](docs/guide.md#how-a-task-runs) lists its gaps. The demo task still fails under
> every adapter as shipped.

## Three lessons

### 1. The repair that did too much

The first judge accepted a repair when every requirement and preservation check passed. A repair
that also changed a field the contract did not name passed too, **so the training reward would
have paid for edits nobody asked for.**

I added a check: every operation must serve a requirement that the original scene fails. Then I
added 15 tasks. In 12 of them an allowed operation could "tidy" a field, and the correct answer is
to leave that field alone.

### 2. GRPO had no signal to use

GRPO started from the first SFT adapter, which already passed **95 percent** of its own train
samples. On the first iteration only **7 of 30** task groups had both an accepted and a failed
reply. By the fourth update it was **1 of 30**. On that run's dev split it never beat its starting
point, so selection kept the SFT adapter
([`train/grpo-manifest.json`](train/grpo-manifest.json)).

Iteration 5 later passed all 17 tasks of the wider dev split, but on test it proposed an operation
the composite task's contract does not permit, and the executor blocked it. **High dev scores on a
saturated train set did not carry to the held-out families.**

### 3. Same adapter, two contracts

Task `demo-01` requires only `up_axis: Z`. The scene is already Z-up, so the judge derives "no
change" as the correct answer. The SFT adapter refuses the task instead, and the judge fails it.

Add `default_prim: /World` to the contract, and the derived answer becomes a repair. The same
adapter sets the default prim, and all 12 checks pass. **Only the contract changed between the two
runs.** Both runs are in [`results/`](results).

## Verification

| Check | Result |
|:--|:--|
| **Tests** | 13 tests, each naming the code change that must make it fail. `tools/mutate.py` makes 11 of those changes, and the named test fails every time. |
| **Controls** | Each of the 60 tasks has a pass control and a fail control. The judge accepts every pass control and rejects every fail control. |
| **Splits** | Train, dev and test do not overlap. Dev selects the adapter. The results table reads test runs only. |

## Quickstart

Windows and Python 3.13. The baselines do not need a GPU.

```powershell
git clone https://github.com/sushiHex/acceptance-lab.git
cd acceptance-lab
python -m venv .venv
.venv\Scripts\pip install -r requirements-windows.lock

.venv\Scripts\python run.py baseline            # both baselines, then runs/table.html
.venv\Scripts\python -m unittest test_lab -v    # tests
.venv\Scripts\python tools/controls.py          # pass and fail control for each task
```

The model configurations need vLLM in WSL Ubuntu and a 24 GB GPU. The
[technical guide](docs/guide.md) covers the stack, setup, evaluation, training and the file layout.

---

<sub>[MIT](LICENSE) for the original code and fixtures. Dependencies keep their own licenses.</sub>
