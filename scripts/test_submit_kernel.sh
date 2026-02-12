#!/bin/bash
# Helper script to test /submit_kernel endpoint with proper authentication
# Usage: ./scripts/test_submit_kernel.sh [wallet_name] [hotkey_name]

set -e

WALLET_NAME=${1:-"quasar_miner"}
HOTKEY_NAME=${2:-"default"}
API_URL=${VALIDATOR_API_URL:-"http://localhost:8000"}

echo "=========================================="
echo "Testing /submit_kernel endpoint"
echo "=========================================="
echo "Wallet: $WALLET_NAME"
echo "Hotkey: $HOTKEY_NAME"
echo "API URL: $API_URL"
echo ""

ENDPOINT="/submit_kernel"
echo "Generating authentication headers..."

AUTH_OUTPUT=$(python3 << EOF
import bittensor as bt
import sys

try:
    wallet = bt.wallet(name="$WALLET_NAME", hotkey="$HOTKEY_NAME")
    hotkey_ss58 = wallet.hotkey.ss58_address
    
    # Sign the hotkey address
    message = hotkey_ss58.encode()
    signature = wallet.hotkey.sign(message)
    signature_hex = signature.hex()
    
    print(f"HOTKEY={hotkey_ss58}")
    print(f"SIGNATURE={signature_hex}")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate authentication headers"
    exit 1
fi

# Extract values
HOTKEY=$(echo "$AUTH_OUTPUT" | grep "HOTKEY=" | cut -d'=' -f2)
SIGNATURE=$(echo "$AUTH_OUTPUT" | grep "SIGNATURE=" | cut -d'=' -f2)

echo "Hotkey: ${HOTKEY:0:20}..."
echo ""

curl -X POST "$API_URL$ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Hotkey: $HOTKEY" \
  -H "Signature: $SIGNATURE" \
  -d "{
    \"miner_hotkey\": \"$HOTKEY\",
    \"fork_url\": \"https://github.com/test/flash-linear-attention\",
    \"commit_hash\": \"abc123\",
    \"repo_hash\": \"test_hash_123\",
    \"target_sequence_length\": 100000,
    \"tokens_per_sec\": 1000.0,
    \"signature\": \"test_signature\"
  }"

echo ""
echo "=========================================="
echo "Test completed"
echo "=========================================="
