#!/bin/bash
# Script to start the Miner
# Run from project root

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting QUASAR-SUBNET Miner"
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
export VALIDATOR_API_URL=${VALIDATOR_API_URL:-"http://localhost:8000"}
export NETUID=${NETUID:-383}
export SUBTENSOR_NETWORK=${SUBTENSOR_NETWORK:-"test"}
export WALLET_MINER_NAME=${WALLET_MINER_NAME:-"quasar_miner"}
export WALLET_HOTKEY=${WALLET_HOTKEY:-"default"}
export TARGET_SEQUENCE_LENGTH=${TARGET_SEQUENCE_LENGTH:-100000}
export AGENT_ITERATIONS=${AGENT_ITERATIONS:-100}
export OPTIMIZATION_INTERVAL=${OPTIMIZATION_INTERVAL:-300}

# Model configuration
export MINER_MODEL_NAME=${MINER_MODEL_NAME:-"Qwen/Qwen3-4B-Instruct-2507"}

# Context builder configuration (Phase 2: Full Repository Context)
export USE_FULL_CONTEXT=${USE_FULL_CONTEXT:-"true"}
export CONTEXT_MAX_FILES=${CONTEXT_MAX_FILES:-50}
export CONTEXT_MAX_SIZE=${CONTEXT_MAX_SIZE:-200000}
# Optional: BYOC mode (commented by default)
# export REPO_PATH=${REPO_PATH:-""}
# export BYOC_FILE_PATH=${BYOC_FILE_PATH:-""}

# Check required environment variables
if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GITHUB_TOKEN is not set!"
    echo "   Please set it in .env file or export it:"
    echo "   export GITHUB_TOKEN=your_token_here"
    exit 1
fi

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ GITHUB_USERNAME is not set!"
    echo "   Please set it in .env file or export it:"
    echo "   export GITHUB_USERNAME=your_username"
    exit 1
fi

echo "Configuration:"
echo "  API URL: $VALIDATOR_API_URL"
echo "  NetUID: $NETUID"
echo "  Network: $SUBTENSOR_NETWORK"
echo "  Wallet: $WALLET_MINER_NAME/$WALLET_HOTKEY"
echo "  GitHub User: $GITHUB_USERNAME"
echo "  Target Seq Length: $TARGET_SEQUENCE_LENGTH"
echo "  Agent Iterations: $AGENT_ITERATIONS"
echo "  Optimization Interval: $OPTIMIZATION_INTERVAL seconds"
echo ""
echo "Model Configuration:"
echo "  Model: $MINER_MODEL_NAME"
echo ""
echo "Context Builder (Phase 2: Full Repository Context):"
echo "  Use Full Context: $USE_FULL_CONTEXT"
echo "  Max Files: $CONTEXT_MAX_FILES"
echo "  Max Size: $CONTEXT_MAX_SIZE chars (~$((CONTEXT_MAX_SIZE / 4))K tokens)"
if [ -n "$REPO_PATH" ]; then
    echo "  Repo Path: $REPO_PATH (BYOC mode)"
fi
if [ -n "$BYOC_FILE_PATH" ]; then
    echo "  BYOC File: $BYOC_FILE_PATH"
fi
echo ""

# Check if API is running
echo "Checking validator API..."
if curl -s "$VALIDATOR_API_URL/health" > /dev/null 2>&1; then
    echo "✅ Validator API is running"
else
    echo "⚠️  Validator API is not running at $VALIDATOR_API_URL"
    echo "   You may want to start it first: ./START_SERVER.sh"
    echo "   Continuing anyway..."
fi

# Check CUDA
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ CUDA is available"
else
    echo "⚠️  CUDA is not available (miner will run on CPU - slower)"
fi

echo ""
echo "Starting miner..."
echo "Press CTRL+C to stop"
echo ""

# Build command arguments
MINER_ARGS=(
    --netuid "$NETUID"
    --wallet.name "$WALLET_MINER_NAME"
    --wallet.hotkey "$WALLET_HOTKEY"
    --subtensor.network "$SUBTENSOR_NETWORK"
    --agent-iterations "$AGENT_ITERATIONS"
    --target-seq-len "$TARGET_SEQUENCE_LENGTH"
    --optimization-interval "$OPTIMIZATION_INTERVAL"
    --model-name "$MINER_MODEL_NAME"
    --logging.debug
)

# Add context builder arguments if configured
if [ "$USE_FULL_CONTEXT" = "true" ]; then
    MINER_ARGS+=(--use-full-context)
fi
MINER_ARGS+=(--context-max-files "$CONTEXT_MAX_FILES")
MINER_ARGS+=(--context-max-size "$CONTEXT_MAX_SIZE")

# Add BYOC mode arguments if configured
if [ -n "$REPO_PATH" ]; then
    MINER_ARGS+=(--repo-path "$REPO_PATH")
fi
if [ -n "$BYOC_FILE_PATH" ]; then
    MINER_ARGS+=(--byoc-file "$BYOC_FILE_PATH")
fi

python -m neurons.miner "${MINER_ARGS[@]}"
