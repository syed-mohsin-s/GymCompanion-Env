# ── GymCompanion-Env Docker Image ──────────────────────────────────────────────
# Lightweight Python 3.10 image for Hugging Face Spaces (8 GB RAM limit)

FROM python:3.10-slim

WORKDIR /app

# Prevent .pyc files and enable unbuffered stdout/stderr for clean logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies (leverages Docker layer cache)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir openenv-core uvicorn fastapi websockets

# Copy application files
COPY openenv.yaml /app/
COPY README.md /app/
COPY models.py /app/
COPY client.py /app/
COPY inference.py /app/
COPY server/ /app/server/

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Start the FastAPI/OpenEnv server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
