FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy miner code
COPY miner/ ./miner/

# Environment variables
ENV MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MAX_NEW_TOKENS=100

# Expose port
EXPOSE 8000

# Run the miner server
CMD ["uvicorn", "miner.server:app", "--host", "0.0.0.0", "--port", "8000"]
