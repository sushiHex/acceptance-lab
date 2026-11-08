#!/usr/bin/env bash
# serve_setup.sh: one-time vLLM install inside WSL Ubuntu (run as root). Idempotent.
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/.../tools/serve_setup.sh
set -euo pipefail
VLLM_VERSION="${VLLM_VERSION:-0.33.*}"

# python3-dev: Triton compiles a C helper against Python.h on the first engine start.
if ! command -v uv >/dev/null || [ ! -f /usr/include/python3.12/Python.h ]; then
  apt-get update -qq
  apt-get install -y -qq curl ca-certificates build-essential python3-dev >/dev/null
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

[ -x /opt/vllm/bin/python ] || uv venv /opt/vllm --python 3.12
uv pip install --python /opt/vllm/bin/python "vllm==${VLLM_VERSION}"
/opt/vllm/bin/python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())'
