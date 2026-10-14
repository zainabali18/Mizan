# Mizan API: CPU-only, runs fully offline once built (no customer data leaves the container).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Dependencies (CPU-only PyTorch keeps the image much smaller than the CUDA build)
COPY requirements-serve.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-serve.txt

# 2. Run as a non-root user
RUN useradd --create-home mizan
ENV HF_HOME=/home/mizan/.cache/huggingface

# 3. Application code + model config
COPY --chown=mizan:mizan mizan/ mizan/
COPY --chown=mizan:mizan mizan_config.json .
USER mizan

# 4. Download the model at BUILD time, then forbid network access to the Hub at runtime
RUN python -c "from mizan.service import MizanClassifier; MizanClassifier().warm_up(); print('model cached')"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "mizan.api:app", "--host", "0.0.0.0", "--port", "8000"]