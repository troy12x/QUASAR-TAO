#!/bin/bash
# Script to start the Validator Neuron
# Run from project root

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting QUASAR-SUBNET Validator Neuron"
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
export WALLET_VALIDATOR_NAME=${WALLET_VALIDATOR_NAME:-"quasar_validator"}
export WALLET_HOTKEY=${WALLET_HOTKEY:-"default"}
export POLLING_INTERVAL=${POLLING_INTERVAL:-300}

# Inference verification settings
export ENABLE_LOGIT_VERIFICATION=${ENABLE_LOGIT_VERIFICATION:-"true"}
export REFERENCE_MODEL=${REFERENCE_MODEL:-"Qwen/Qwen2.5-0.5B-Instruct"}
export COSINE_SIM_THRESHOLD=${COSINE_SIM_THRESHOLD:-0.99}
export MAX_ABS_DIFF_THRESHOLD=${MAX_ABS_DIFF_THRESHOLD:-0.1}

# Commit-reveal settings
export BLOCKS_UNTIL_REVEAL=${BLOCKS_UNTIL_REVEAL:-100}
export BLOCK_TIME_SECONDS=${BLOCK_TIME_SECONDS:-12}

echo "Configuration:"
echo "  API URL: $VALIDATOR_API_URL"
echo "  NetUID: $NETUID"
echo "  Network: $SUBTENSOR_NETWORK"
echo "  Wallet: $WALLET_VALIDATOR_NAME/$WALLET_HOTKEY"
echo "  Polling Interval: $POLLING_INTERVAL seconds"
echo ""
echo "Inference Verification:"
echo "  Enabled: $ENABLE_LOGIT_VERIFICATION"
echo "  Reference Model: $REFERENCE_MODEL"
echo "  Cosine Similarity Threshold: $COSINE_SIM_THRESHOLD"
echo ""
echo "Commit-Reveal:"
echo "  Blocks Until Reveal: $BLOCKS_UNTIL_REVEAL (~$((BLOCKS_UNTIL_REVEAL * BLOCK_TIME_SECONDS / 60)) minutes)"
echo ""

# Check if API is running
echo "Checking validator API..."
if curl -s "$VALIDATOR_API_URL/health" > /dev/null 2>&1; then
    echo "✅ Validator API is running"
else
    echo "❌ Validator API is not running!"
    echo "   Please start it first: ./START_SERVER.sh"
    exit 1
fi

# Check CUDA for inference verification
if [ "$ENABLE_LOGIT_VERIFICATION" = "true" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "✅ CUDA is available for inference verification"
    else
        echo "⚠️  CUDA is not available (inference verification will be slower)"
    fi
fi

echo ""
echo "Starting validator neuron..."
echo "Press CTRL+C to stop"
echo ""

python -m neurons.validator \
    --netuid "$NETUID" \
    --wallet.name "$WALLET_VALIDATOR_NAME" \
    --wallet.hotkey "$WALLET_HOTKEY" \
    --subtensor.network "$SUBTENSOR_NETWORK" \
    --neuron.polling_interval "$POLLING_INTERVAL" \
    --logging.debug
