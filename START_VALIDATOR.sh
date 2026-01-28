#!/bin/bash
# Script to start the Validator
# Run from project root

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting QUASAR-SUBNET Validator"
echo "=========================================="
echo ""

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Loaded .env file"
else
    echo "⚠️  No .env file found. Using defaults."
fi

# Set defaults if not in .env
export VALIDATOR_API_URL=${VALIDATOR_API_URL:-"http://localhost:8000"}
export NETUID=${NETUID:-24}
export SUBTENSOR_NETWORK=${SUBTENSOR_NETWORK:-"test"}
# Force quasar_validator if not explicitly set (override .env if it says "validator")
if [ -z "$WALLET_NAME" ] || [ "$WALLET_NAME" = "validator" ]; then
    export WALLET_NAME="quasar_validator"
fi
export WALLET_HOTKEY=${WALLET_HOTKEY:-"default"}

echo "Configuration:"
echo "  API URL: $VALIDATOR_API_URL"
echo "  NetUID: $NETUID"
echo "  Network: $SUBTENSOR_NETWORK"
echo "  Wallet: $WALLET_NAME/$WALLET_HOTKEY"
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

echo ""
echo "Starting validator..."
echo "Press CTRL+C to stop"
echo ""

python neurons/validator.py \
    --netuid "$NETUID" \
    --wallet.name "$WALLET_NAME" \
    --wallet.hotkey "$WALLET_HOTKEY" \
    --subtensor.network "$SUBTENSOR_NETWORK" \
    --neuron.polling_interval 300 \
    --logging.debug
