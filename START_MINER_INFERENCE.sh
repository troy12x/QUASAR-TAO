#!/bin/bash
# Script to start the Miner Inference Server (for inference verification testing)
# Uses port 8001 by default to avoid conflict with Validator API (8000)
# Run from project root

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting Miner Inference Server"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Activated virtual environment"
fi

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✅ Loaded .env file"
else
    echo "⚠️  No .env file found. Using defaults."
fi

# Use 8001 to avoid conflict with Validator API (8000)
export MINER_INFERENCE_PORT=${MINER_INFERENCE_PORT:-8001}
export PORT=$MINER_INFERENCE_PORT
export HOST=${HOST:-"0.0.0.0"}
export MODEL_NAME=${REFERENCE_MODEL:-"Qwen/Qwen2.5-0.5B-Instruct"}
export DEVICE=${DEVICE:-"cuda"}

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT (avoids Validator API on 8000)"
echo "  Model: $MODEL_NAME"
echo "  Device: $DEVICE"
echo ""

# Check CUDA
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ CUDA is available"
else
    echo "⚠️  CUDA not available (will use CPU - slower)"
fi

echo ""
echo "Inference server will be at: http://$HOST:$PORT"
echo "  POST /inference - Run inference with logit capture"
echo "  GET  /health    - Health check"
echo ""
echo "Press CTRL+C to stop"
echo ""

python miner/inference_server.py
