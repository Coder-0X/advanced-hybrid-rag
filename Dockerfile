# ==============================================================================
# Production Multi-Stage Dockerfile for Advanced Hybrid RAG Engine
# ==============================================================================

# Stage 1: Build & Dependencies
FROM python:3.10-slim-bookworm AS builder

WORKDIR /build

# Install compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install wheels to clean prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Minimal Runtime Environment
FROM python:3.10-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0

# Install runtime curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user
RUN groupadd -g 10001 raggroup && \
    useradd -u 10001 -g raggroup -s /bin/bash -m appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source code
COPY --chown=appuser:raggroup app/ /app/app/
COPY --chown=appuser:raggroup sample_data/ /app/sample_data/
COPY --chown=appuser:raggroup run_benchmark.py /app/run_benchmark.py
COPY --chown=appuser:raggroup pyproject.toml /app/pyproject.toml

# Create data directories with appropriate permissions
RUN mkdir -p /app/data/benchmarks /app/data/qdrant_db && \
    chown -R appuser:raggroup /app/data

# Switch to non-privileged user
USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
