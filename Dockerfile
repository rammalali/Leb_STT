FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# CUDA-matched torch wheels first (largest layer, cached unless these change)
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

# --extra-index-url cu126 ensures any transitive upgrade still pulls CUDA wheels
COPY requirements.txt ./
RUN python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python3", "run.py"]
