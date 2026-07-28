# --- Build stage ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Build toolchain - pre-built wheels exist for scipy/numpy/psycopg[binary]
# on common platforms, but gcc + libpq-dev are kept as a fallback for
# any transitive dependency that requires compilation from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install ".[telemetry,llm]"

# --- Runtime stage ---
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Runtime library fallback - psycopg[binary] bundles its own libpq,
# but libpq5 is kept in case a pure-Python psycopg fallback is used.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root runtime user
RUN groupadd --system appuser && useradd --system --gid appuser appuser
USER appuser

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "hazard_assessment.app:app", "--host", "0.0.0.0", "--port", "8000"]
