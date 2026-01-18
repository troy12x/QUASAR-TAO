# Use official Python runtime as a parent image
# Cache bust: 2025-01-18-02
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for compiling some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY necessary files first to cache dependencies
COPY validator_api/requirements.txt /app/requirements.txt

# Install dependencies (no-cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
# We copy the root 'quasar' package because the API imports it
COPY quasar /app/quasar
COPY validator_api /app/validator_api

# Expose the API port
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "rm -f /app/quasar.db && uvicorn validator_api.app:app --host 0.0.0.0 --port 8000"]
