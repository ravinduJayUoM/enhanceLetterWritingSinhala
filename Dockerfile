FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/cache/huggingface/sentence-transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

# Install CPU-only torch first from PyTorch's wheel index. This avoids pulling
# ~3GB of CUDA libraries that the deployment VM (no GPU) will never use.
# `requirements.txt` then sees torch already satisfied and skips it.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY rag/ ./rag/
COPY data/ ./data/

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /cache/huggingface /app/rag/faiss_index \
    && chown -R app:app /app /cache

USER app

WORKDIR /app/rag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:8000/ >/dev/null || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
