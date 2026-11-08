#!/usr/bin/env bash
# serve.sh: serve the reference model on 127.0.0.1:8000 (run inside WSL after serve_setup.sh).
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/.../tools/serve.sh
set -euo pipefail
MODEL="${MODEL:-nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1}"

# The FlashInfer sampler JIT-compiles on first use and needs nvcc, which WSL does not ship.
# The lab decodes at temperature 0, so the sampler choice cannot change a result.
export VLLM_USE_FLASHINFER_SAMPLER=0

# LORA=/root/adapters/sft also serves that adapter as model id "sft"; the base id stays served.
exec /opt/vllm/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.6 \
  ${LORA:+--enable-lora --max-lora-rank 16 --lora-modules "sft=$LORA"}
