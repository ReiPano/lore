FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY config.yaml ./config.yaml

RUN pip install --upgrade pip \
 && pip install ".[parsers]"

# Model cache for FastEmbed; mount a volume here in production to avoid re-downloads.
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed

EXPOSE 8765

CMD ["hybrid-search", "serve-api", "--host", "0.0.0.0"]
