# Results

The run files behind the numbers in the [README](../README.md). Each folder holds one run's
`results.jsonl`: a header line (configuration, split, task hash, versions), then one line per task
with the proposal, the executor's receipt and the judge's verdict.

| Folder | What it is |
|---|---|
| `baseline-*` | Both baselines over all 60 tasks. The README table reads their 4 test tasks. |
| `test-*` | The four model configurations on the test split. |
| `dev-sft-epoch-*` | The SFT adapter after each epoch, on dev. Epoch 3 was selected. |
| `dev-grpo` | GRPO iteration 5 on dev. |
| `demo-*` | `demo-01` as shipped, and with `default_prim: /World` added to its contract. |

To print each run's counts:

```
python tools/snapshot_results.py
```
