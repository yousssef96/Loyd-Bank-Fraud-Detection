# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# System dependency needed by XGBoost / some sklearn ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (better layer caching — only re-installs if these change)
COPY pyproject.toml uv.lock* ./

# Install uv, then sync locked dependencies (no dev deps in production image)
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy project source code
COPY src ./src

# Copy the downloaded registered model artifacts
# (must run `mlflow artifacts download` locally before building this image)
COPY src/serving/model /app/model

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]