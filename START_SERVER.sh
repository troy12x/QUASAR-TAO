#!/bin/bash
# Quick script to start the FastAPI server
# Run from project root

cd "$(dirname "$0")"

echo "Starting FastAPI server..."
echo "Server will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop"
echo ""

uvicorn validator_api.app:app --host 0.0.0.0 --port 8000 --reload
