#!/bin/bash
# Script to start the FastAPI Validator API Server
# Run from project root

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting QUASAR-SUBNET Validator API"
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

# Set defaults if not in .env
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8000}
export DATABASE_URL=${DATABASE_URL:-"sqlite:///./quasar_validator.db"}

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Database: ${DATABASE_URL:0:50}..."
echo ""

echo "Starting FastAPI server..."
echo "Server will be available at: http://$HOST:$PORT"
echo "API docs at: http://localhost:$PORT/docs"
echo ""
echo "Press CTRL+C to stop"
echo ""

uvicorn validator_api.app:app --host "$HOST" --port "$PORT" --reload
