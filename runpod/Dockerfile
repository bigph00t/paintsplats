# Paintsplats — RunPod Serverless image (lean variant).
#
# This image ships CODE + DEPS only; model weights are downloaded lazily on
# first inference and cached in /runpod-volume (or ~/.cache/huggingface for
# workers without a volume). Trades a 5–10 min first-cold-start for a small,
# CI-buildable image and a simpler deploy story.
#
# Build:
#   docker build -t ghcr.io/bigph00t/paintsplats:dev -f runpod/Dockerfile .
#
# Or use the GitHub Actions workflow (`.github/workflows/docker.yml`) which
# pushes ghcr.io/bigph00t/paintsplats:latest + :v* on every push / tag.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /workspace

# Torch first — pins the rest of the resolver.
# sam-3d-objects pins torch*==2.5.1+cu121. Match that exactly here so its
# install resolves; switching this also requires changing the FROM base.
RUN pip3 install --upgrade pip \
 && pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Paintsplats source + Python deps.
COPY pyproject.toml README.md /workspace/
COPY paintsplats /workspace/paintsplats
RUN pip3 install -e ".[gpu,runpod]" hf_transfer

# SAM 3D Objects and SAM 3 don't publish wheels; install from source.
RUN git clone --depth 1 https://github.com/facebookresearch/sam-3d-objects.git /opt/sam-3d-objects \
 && pip3 install -e /opt/sam-3d-objects \
 && git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3 \
 && pip3 install -e /opt/sam3

COPY runpod/handler.py /workspace/handler.py

# Weights are NOT baked. Set HF_TOKEN in the RunPod endpoint env so
# huggingface_hub can pull facebook/sam-3d-objects, facebook/sam3, and
# stabilityai/stable-diffusion-xl-base-1.0 on first inference.
CMD ["python3", "-u", "/workspace/handler.py"]
