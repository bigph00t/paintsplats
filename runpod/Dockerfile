# Paintsplats — RunPod Serverless image.
#
# Bakes SAM 3, SAM 3D Objects, and SDXL weights into the image so cold starts
# don't pay 1–2 minutes streaming 60+ GB off a network volume. Pass HF_TOKEN
# at build time as a BuildKit secret:
#
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=hf,env=HF_TOKEN \
#     -t paintsplats:0.1.0 -f runpod/Dockerfile .

# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /workspace

# Torch first so the rest of the stack resolves against it.
RUN pip3 install --upgrade pip \
 && pip3 install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# Paintsplats source.
COPY pyproject.toml README.md /workspace/
COPY paintsplats /workspace/paintsplats
RUN pip3 install -e ".[gpu,runpod]"

# SAM 3D Objects — install from source, repo doesn't publish a wheel.
RUN git clone --depth 1 https://github.com/facebookresearch/sam-3d-objects.git /opt/sam-3d-objects \
 && pip3 install -e /opt/sam-3d-objects

# SAM 3 — same story.
RUN git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3 \
 && pip3 install -e /opt/sam3

# Pre-download checkpoints using the HF_TOKEN secret so the image is fully
# self-contained. The token is NOT baked into the layer.
RUN --mount=type=secret,id=hf,target=/run/secrets/hf \
    HF_TOKEN="$(cat /run/secrets/hf)" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
token = os.environ["HF_TOKEN"]
for repo_id in (
    "facebook/sam-3d-objects",
    "facebook/sam3",
    "stabilityai/stable-diffusion-xl-base-1.0",
):
    snapshot_download(repo_id, token=token, local_dir_use_symlinks=False)
PY

COPY runpod/handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
