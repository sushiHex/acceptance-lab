# Technical guide

How Acceptance Lab works, how to run and train it, and how it is verified. The [README](../README.md) has the overview and results.

## Stack

| Stage | Tools |
|---|---|
| Model | [Nemotron Nano 4B v1.1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1), served by vLLM with LoRA adapters |
| Proposer | LangChain `ChatOpenAI`, optional structured outputs, fastembed rule retrieval |
| Guard | [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) output rail that runs the shared preflight |
| Execute | MCP server, one process per attempt, OpenUSD from the [OpenUSD Exchange SDK](https://docs.omniverse.nvidia.com/usd/code-docs/usd-exchange-sdk/latest/index.html) package |
| Judge | A2A agent; deterministic checks, read-only access to the task and the attempt |
| Train | LoRA SFT with PEFT, and GRPO with the judge as the reward |

## How a task runs

```
tasks.jsonl ──► coordinator (run.py) ──visible projection──► proposer (baseline or model)
                    │                                              │ proposal JSON
                    │                                              ▼
                    │                              Guardrails output rail (preflight)
                    │ MCP (stdio)                                  │
                    ├──────────► executor: inspect, finalize ◄─────┘
                    │                 writes the repaired bundle + manifest.json
                    │ A2A (HTTP)
                    └──────────► judge: reads the frozen task and the attempt dir, returns a verdict
```

- **The model never sees a tool or a file.** It receives one allowlisted projection of the task
  (`run.visible`: id, request, contract, approved asset ids, scene facts) and returns one JSON
  proposal. The coordinator is the only MCP client and the only A2A client.
- **The executor** (`protocols.py mcp`) is one MCP server process per attempt, bound to its attempt
  directory at spawn. Two tools: `inspect` and `finalize(proposal, execution_allowed)`.
- **The judge** (`protocols.py judge`) is an A2A agent. It gets only an attempt id, resolves it
  through the run registry, reads the frozen `tasks.jsonl` and fixtures read-only, and walks the
  result with the same checks the contract names. It never mutates anything. A repair passes only
  if every operation serves a requirement the original fails (`expectation.minimal`), so tidying a
  field the contract does not name is a failure, in grading and in the training reward alike.
  The check maps each operation to one requirement (`scene.SERVES`), so it is coarse: it does not
  catch a duplicated operation, and it would fail a prerequisite step or a reference repair made
  to satisfy a default-prim check. No shipped task needs either.
- **One preflight** (`scene.preflight`) decides whether an operation is permitted. The executor,
  the judge and the Guardrails rail all call it, so there is one definition of "allowed".

## The model lane (`propose.py`)

- **Serving:** any OpenAI-compatible endpoint. The reference row is Nemotron Nano 4B v1.1 on vLLM.
- **Client:** LangChain `ChatOpenAI`, temperature 0, no retries, system turn `detailed thinking off`.
- **Strict parsing:** the reply must be exactly one valid proposal. One repair turn is allowed;
  after that the attempt is `proposal_unparseable`. A leaked `<think>` block counts as a failure.
- **Structured outputs (`--structured`):** the server constrains decoding to `proposal_json_schema()`
  (OpenAI `response_format`). Same prompt, so it is its own row and differs by the constraint alone.
- **Retrieval:** fastembed (`BAAI/bge-small-en-v1.5`) over `rules/rules.jsonl`. A task sees general
  rules plus rules from its own split, top 3 above a cosine floor of 0.60. A chunk pinned to a task
  always reaches that task, which is how the planted injection note in `injection-01` is delivered.
- **Guardrails:** a NeMo Guardrails output rail (`guardrails/`) runs `scene.preflight` over the
  proposal. It makes no LLM call. A rail that did not run counts as blocked, and a block on a
  proposal preflight would permit is recorded as `rail_false_positive`.

## Tasks

61 bundles in `fixtures/`, authored deterministically by `tools/build_fixtures.py` and frozen.
Splits: train 39, dev 17, test 4, demo 1. Kinds: up-axis and units (metadata and geometry),
default prim, broken reference, composite, no-op, ambiguous (must abstain), forbidden operations,
and a prompt injection. Train and dev are widened by `VARIANTS` (other values, prims, permissions
and requests per kind, each with its own controls); the test families stay held out: no
composite, no orientation ambiguity and no axis refusal appear outside test.

## Commands

```
python run.py baseline                                  # always-abstain + scripted heuristic, then the table
python run.py eval --base-url URL --model ID [--label L] [--split test] [--structured]
python run.py demo --base-url URL --model ID            # demo-01, prints each verdict check
python run.py report                                    # rebuild runs/table.html from every run
```

Training. The judge is the only label source and the only reward, and dev picks what ships:

```
python tools/sft_data.py sample --base-url URL --model ID [--structured]   # K sampled runs over train, executed and judged
python tools/sft_data.py build RUN_DIR ...      # keep what scene.model_success accepts -> train/sft.jsonl
wsl -d Ubuntu-24.04 -u root -- /opt/vllm/bin/python tools/sft_train.py --out /root/adapters/sft   # one adapter per epoch
wsl -d Ubuntu-24.04 -u root -- env LORAS="e1=/root/adapters/sft/epoch-1 e2=..." bash tools/serve.sh
python run.py eval --base-url URL --model e2 --split dev --label "sft epoch 2"   # choose the epoch on dev
python tools/grpo.py --start /root/adapters/sft/epoch-N      # GRPO: sample, judge, update; dev picks the iteration
```

- **SFT** is rejection sampling: the model's own samples, kept only where the judge accepts them.
- **GRPO** samples a group of single-turn replies per train task, rewards each with
  `scene.model_success`, and updates on the group-relative advantage (`tools/grpo_step.py`,
  clipped surrogate, no KL term; the dev split is the guard). Groups the judge scored all alike
  are dropped, since they carry no signal.
- `train/adapter-manifest.json` and `train/grpo-manifest.json` record data hashes,
  hyperparameters, losses, per-iteration reward and dev success, and the adapter selected.
- Dev runs never enter the table; it reads test-split runs only.

The table compares one cohort: runs over the same `tasks.jsonl` hash, scored on the test split.
`configurations.json` declares every planned row, and a row that did not run says why.

## Verification

```
python -m unittest test_lab -v        # boundary tests, each naming the mutation that breaks it
python tools/mutate.py                # breaks the code on purpose; every guarding test must go red
python tools/controls.py              # every fixture: a pass control accepted, a fail control rejected
python tools/stub_model.py PORT MODE  # OpenAI-compatible test double: scripted | fence | junk | obey-notes
```

The stub exercises the real model lane (LangChain, the repair turn, retrieval, Guardrails) with no
GPU. It is a test double and never a table row.

## Setup

Windows (baselines and the whole lane except serving):

```
python -m venv .venv
.venv\Scripts\pip install -r requirements-windows.lock
```

`requirements-baseline.lock` is enough for the baselines alone. Serving runs in WSL Ubuntu:

```
wsl -d Ubuntu-24.04 -u root -- bash tools/serve_setup.sh
wsl -d Ubuntu-24.04 -u root -- bash tools/serve.sh     # bf16, 16K context, 60% of a 24 GB card
python run.py demo --base-url http://127.0.0.1:8000/v1 --model nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1
```

## Layout

| Path | What it is |
|---|---|
| `scene.py` | Everything done to an OpenUSD bundle: inspect, preflight, apply, walk, checks |
| `protocols.py` | The MCP executor and the A2A judge, thin over `scene.py` |
| `run.py` | Coordinator and CLI, baseline proposers |
| `propose.py` | The model lane: retrieval, prompt, parsing, the rail, the proposer |
| `report.py` | Run log line and `runs/table.html` |
| `rules/`, `guardrails/` | Retrieval corpus and the rail config |
| `probes/` | Version probes for pxr, A2A and Guardrails, with their log |
| `tools/` | Fixture builder, controls, mutation runner, leak scan, stub model, WSL serve setup and launch |
