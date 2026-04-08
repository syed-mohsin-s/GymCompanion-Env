# Use a lightweight Python image to stay well within the 8GB RAM limit
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Prevent Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies first (leverages Docker cache)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install openenv-core uvicorn fastapi  # Ensuring core OpenEnv dependencies are present

# Copy the rest of the application
# We need the models, the YAML, the README, and the server logic
COPY openenv.yaml /app/
COPY README.md /app/
COPY models.py /app/
COPY server/ /app/server/

# Hugging Face Spaces route web traffic to port 7860 by default
EXPOSE 7860

# Command to run the FastAPI/OpenEnv wrapper
# Adjust 'server.app:app' if your FastAPI instance is named or located differently
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
