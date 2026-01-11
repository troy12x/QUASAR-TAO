FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY validator_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy challenge code
COPY validator_api/ ./validator_api/

# Create data directory for docmath dataset
RUN mkdir -p /data

# Environment variables
ENV DATASET_PATH=/data/docmath.jsonl
ENV CHALLENGE_HOST=0.0.0.0
ENV CHALLENGE_PORT=8080
ENV SAMPLES_PER_EVALUATION=50
ENV TIMEOUT_SECS=300

# Expose port
EXPOSE 8080

# Run the challenge server
CMD ["uvicorn", "validator_api.app:app", "--host", "0.0.0.0", "--port", "8080"]
