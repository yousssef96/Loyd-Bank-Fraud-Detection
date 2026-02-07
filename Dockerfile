# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app


# We combine update, install, and cleanup into one RUN to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

    
# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock* ./

# Install uv (dependency manager)
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy project files
COPY . .

# Explicitly copy model (in case .dockerignore excluded mlruns)
# NOTE: destination changed to /app/src/serving/model to match inference.py's path
COPY src/serving/model /app/src/serving/model

# Copy MLflow run (artifacts + metadata) to the flat /app/model convenience path
COPY src/serving/model/49b0b2861e544aa98a64014b37c12022/artifacts/model /app/model
COPY src/serving/model/49b0b2861e544aa98a64014b37c12022/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/49b0b2861e544aa98a64014b37c12022/artifacts/preprocessing.pkl /app/model/preprocessing.pkl


# make "serving" and "app" importable without the "src." prefix
# ensures logs are shown in real-time (no buffering).
# lets you import modules using from app... instead of from src.app....
ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app/src

    
# Expose FastAPI default port
EXPOSE 8000

# Command to run API with Uvicorn
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]