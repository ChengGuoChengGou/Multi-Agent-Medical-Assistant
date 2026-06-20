# ==============================================================================
# Multi-stage Dockerfile for Multi-Agent Medical Assistant
# Phase 31: Optimized for production — non-root user, layer caching, healthcheck
# ==============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system deps (removed in runtime stage)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libpng-dev \
    libjpeg-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first → leverage Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── Stage 2: Runtime (minimal) ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="medical-assistant" \
      version="3.1" \
      description="Multi-Agent Medical Chatbot – production image"

# Install only runtime system deps (no build-essential)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN groupadd -r medical && useradd -r -g medical -d /app -s /sbin/nologin medical

WORKDIR /app

# Copy application code
COPY --chown=medical:medical . .

# Create necessary directories with proper ownership
RUN mkdir -p uploads/backend uploads/frontend uploads/skin_lesion_output \
             uploads/speech data log && \
    chown -R medical:medical /app

# Switch to non-root user
USER medical

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Expose port (app reads PORT env or defaults to 8000)
EXPOSE 8000

# Healthcheck (matches docker-compose + K8s readiness)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint: gunicorn with uvicorn workers
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
