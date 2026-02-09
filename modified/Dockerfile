# syntax=docker/dockerfile:1.6
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    HUGGINGFACE_HUB_CACHE=/models/hf \
    QWEN_TTS_DEVICE=cpu \
    QWEN_TTS_DTYPE=float32 \
    QWEN_TTS_ATTN=eager

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgomp1 \
    libsndfile1 \
    libsox-fmt-all \
    sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url ${TORCH_INDEX_URL} torch torchaudio \
    && pip install -e .

EXPOSE 8000

# Default API server via uvicorn. Args can be overridden at runtime.
ENTRYPOINT ["python", "-m", "uvicorn", "qwen_tts.api.server:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
