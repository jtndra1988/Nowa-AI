# ---- STAGE 1: Build deps ----
FROM python:3.10-slim AS builder
ENV POETRY_NO_INTERACTION=1 POETRY_VIRTUALENVS_CREATE=false
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir poetry

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
# Install your deps once (no-root so it doesn't try to install your app)
RUN poetry install --no-root --no-ansi
# Install local pandas-ta wheel
COPY wheels/pandas_ta-0.3.14b0-py3-none-any.whl ./wheels/
RUN pip install ./wheels/pandas_ta-0.3.14b0-py3-none-any.whl

# ---- STAGE 2: Runtime ----
FROM python:3.10-slim
WORKDIR /app

# Bring in installed deps
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your app
COPY . .

# Default command is only used by backend service; celery overrides it in compose
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]