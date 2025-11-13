# ---- STAGE 1: Build deps ----
FROM python:3.10-slim AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build essentials for any native deps during install
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY wheels/pandas_ta-0.3.14b0-py3-none-any.whl ./wheels/
RUN pip install --no-cache-dir ./wheels/pandas_ta-0.3.14b0-py3-none-any.whl
RUN pip install --no-cache-dir poetry

WORKDIR /app

# Copy only dep files first for cache-friendly installs
COPY pyproject.toml poetry.lock* ./

# If lock is out-of-sync, refresh it in-image (no updates beyond pinned versions)
# If there is no lock, this will create one.
RUN poetry lock --no-update --no-ansi || true

# Install deps (no app yet)
RUN poetry install --no-root --no-ansi

# ---- STAGE 2: Runtime ----
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# If you use psycopg2 v2, you may need libpq at runtime; uncomment if you get ImportError:
# RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*

# Bring in installed deps from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your app
COPY . .

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
