#!/bin/bash

# Miner Registration Script (Simplified)
# Registers a miner with the validator API
#
# Usage:
#   ./scripts/register_miner.sh --hotkey <hotkey_ss58> [options]
#
# Options:
#   --hotkey HOTKEY         Hotkey SS58 address (REQUIRED)
#   --wallet-name NAME      Wallet name (default: quasar_miner)
#   --hotkey-name NAME      Hotkey name (default: default)
#   --model MODEL           Model name (default: Qwen/Qwen2.5-0.5B-Instruct)
#   --league LEAGUE         League: 100k, 200k, ..., 1M (default: 100k)
#   --api-url URL           API URL (overrides env detection)
#   --env ENV               Environment: local or production (default: auto-detect)
#   --help                  Show this help message

set -e

# Default values
HOTKEY=""
WALLET_NAME="quasar_miner"
HOTKEY_NAME="default"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
LEAGUE="100k"
API_URL=""
ENV_MODE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hotkey)
            HOTKEY="$2"
            shift 2
            ;;
        --wallet-name)
            WALLET_NAME="$2"
            shift 2
            ;;
        --hotkey-name)
            HOTKEY_NAME="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --league)
            LEAGUE="$2"
            shift 2
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --env)
            ENV_MODE="$2"
            shift 2
            ;;
        --help)
            echo "Miner Registration Script"
            echo ""
            echo "Usage: $0 --hotkey <hotkey_ss58> [options]"
            echo ""
            echo "Required:"
            echo "  --hotkey HOTKEY         Hotkey SS58 address"
            echo ""
            echo "Options:"
            echo "  --wallet-name NAME      Wallet name (default: quasar_miner)"
            echo "  --hotkey-name NAME      Hotkey name (default: default)"
            echo "  --model MODEL           Model name (default: Qwen/Qwen2.5-0.5B-Instruct)"
            echo "  --league LEAGUE         League: 100k, 200k, ..., 1M (default: 100k)"
            echo "  --api-url URL           API URL (overrides env detection)"
            echo "  --env ENV               Environment: local or production (default: auto-detect)"
            echo ""
            echo "Examples:"
            echo "  $0 --hotkey 5HN7ZJXeoHUHskh4pA2ZmW68Pdq5XUE1ak9W5pqwELhgRiFQ --env local"
            echo "  $0 --hotkey 5HN7ZJXeoHUHskh4pA2ZmW68Pdq5XUE1ak9W5pqwELhgRiFQ --env production"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate hotkey is provided
if [ -z "$HOTKEY" ]; then
    echo -e "${RED}❌ Error: --hotkey is required${NC}"
    echo ""
    echo "Usage: $0 --hotkey <hotkey_ss58> [options]"
    echo "Use --help for more information"
    exit 1
fi

# Validate hotkey format (SS58 addresses start with 5 and are ~48 chars)
if ! echo "$HOTKEY" | grep -qE '^5[1-9A-HJ-NP-Za-km-z]{47}$'; then
    echo -e "${YELLOW}⚠️  Warning: Hotkey format may be invalid${NC}"
    echo "Expected SS58 address format (starts with 5, ~48 characters)"
fi

# Validate league
VALID_LEAGUES=("100k" "200k" "300k" "400k" "500k" "600k" "700k" "800k" "900k" "1M")
LEAGUE_VALID=false
for valid_league in "${VALID_LEAGUES[@]}"; do
    if [ "$LEAGUE" == "$valid_league" ]; then
        LEAGUE_VALID=true
        break
    fi
done

if [ "$LEAGUE_VALID" == false ]; then
    echo -e "${RED}❌ Invalid league: $LEAGUE${NC}"
    echo "Valid leagues: ${VALID_LEAGUES[*]}"
    exit 1
fi

# Detect environment if not specified
if [ -z "$API_URL" ]; then
    if [ -z "$ENV_MODE" ]; then
        # Auto-detect: check for .env file or environment variable
        if [ -f ".env" ]; then
            if grep -q "VALIDATOR_API_URL" .env 2>/dev/null; then
                ENV_API_URL=$(grep "^VALIDATOR_API_URL=" .env | cut -d '=' -f2 | tr -d '"' | tr -d "'" | xargs)
                if [ ! -z "$ENV_API_URL" ] && [ "$ENV_API_URL" != "http://localhost:8000" ]; then
                    API_URL="$ENV_API_URL"
                    ENV_MODE="production"
                else
                    API_URL="http://localhost:8000"
                    ENV_MODE="local"
                fi
            else
                API_URL="http://localhost:8000"
                ENV_MODE="local"
            fi
        else
            if [ ! -z "${VALIDATOR_API_URL}" ] && [ "${VALIDATOR_API_URL}" != "http://localhost:8000" ]; then
                API_URL="${VALIDATOR_API_URL}"
                ENV_MODE="production"
            else
                API_URL="http://localhost:8000"
                ENV_MODE="local"
            fi
        fi
    else
        case "$ENV_MODE" in
            local)
                API_URL="http://localhost:8000"
                ;;
            production)
                if [ -f ".env" ] && grep -q "VALIDATOR_API_URL" .env 2>/dev/null; then
                    API_URL=$(grep "^VALIDATOR_API_URL=" .env | cut -d '=' -f2 | tr -d '"' | tr -d "'" | xargs)
                elif [ ! -z "${VALIDATOR_API_URL}" ]; then
                    API_URL="${VALIDATOR_API_URL}"
                else
                    echo -e "${RED}❌ Production API URL not found${NC}"
                    echo "Set VALIDATOR_API_URL in .env or environment variable"
                    exit 1
                fi
                ;;
            *)
                echo -e "${RED}❌ Invalid environment: $ENV_MODE${NC}"
                echo "Use 'local' or 'production'"
                exit 1
                ;;
        esac
    fi
fi

# Print configuration
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Miner Registration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Environment: ${YELLOW}$ENV_MODE${NC}"
echo -e "Hotkey:      ${YELLOW}$HOTKEY${NC}"
echo -e "Model:       ${YELLOW}$MODEL_NAME${NC}"
echo -e "League:      ${YELLOW}$LEAGUE${NC}"
echo -e "API URL:     ${YELLOW}$API_URL${NC}"
echo ""

# Check if Python is available and find the right one
PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
elif ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 not found${NC}"
    exit 1
fi

# Generate signature using Python
echo -e "${BLUE}🔐 Generating signature...${NC}"
SIGNATURE=$($PYTHON_CMD << EOF
import bittensor as bt
import sys

wallet_name = "$WALLET_NAME"
hotkey_name = "$HOTKEY_NAME"
hotkey_ss58 = "$HOTKEY"

try:
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    keypair = wallet.hotkey
    
    # Verify the hotkey matches
    if wallet.hotkey.ss58_address != hotkey_ss58:
        print(f"Error: Hotkey mismatch. Expected {hotkey_ss58}, got {wallet.hotkey.ss58_address}", file=sys.stderr)
        sys.exit(1)
    
    # Sign the hotkey SS58 address
    message = hotkey_ss58.encode('utf-8')
    signature = keypair.sign(message)
    signature_hex = signature.hex()
    
    print(signature_hex)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error generating signature${NC}"
    echo "$SIGNATURE" >&2
    echo ""
    echo "Make sure:"
    echo "  1. Wallet '$WALLET_NAME' with hotkey '$HOTKEY_NAME' exists"
    echo "  2. The hotkey SS58 address matches: $HOTKEY"
    echo "  3. Bittensor is installed: pip install bittensor"
    exit 1
fi

echo -e "${GREEN}✅ Signature generated${NC}"
echo ""

# Prepare request body
REQUEST_BODY=$(cat << EOF
{
    "hotkey": "$HOTKEY",
    "model_name": "$MODEL_NAME",
    "league": "$LEAGUE"
}
EOF
)

# Make API request
echo -e "${BLUE}📤 Sending registration request...${NC}"
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/register_miner" \
    -H "Content-Type: application/json" \
    -H "Hotkey: $HOTKEY" \
    -H "Signature: $SIGNATURE" \
    -d "$REQUEST_BODY" 2>&1)

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

# Check if curl failed
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Request failed${NC}"
    echo "$RESPONSE"
    exit 1
fi

# Parse and display response
echo -e "${BLUE}Response (HTTP $HTTP_CODE):${NC}"

# Try to format as JSON, fallback to raw output
FORMATTED=$(echo "$BODY" | $PYTHON_CMD -m json.tool 2>/dev/null || echo "$BODY")
echo "$FORMATTED"
echo ""

# Check response
if [ "$HTTP_CODE" -eq 200 ] || [ "$HTTP_CODE" -eq 201 ]; then
    STATUS=$(echo "$BODY" | $PYTHON_CMD -c "import sys, json; print(json.load(sys.stdin).get('status', ''))" 2>/dev/null || echo "")
    
    if [ "$STATUS" == "registered" ]; then
        echo -e "${GREEN}✅ Miner registered successfully!${NC}"
        echo ""
        echo "You can now submit kernels via /submit_kernel endpoint"
        exit 0
    elif [ "$STATUS" == "already_registered" ]; then
        echo -e "${YELLOW}ℹ️  Miner already registered${NC}"
        echo ""
        echo "Registration is active. You can submit kernels."
        exit 0
    else
        echo -e "${YELLOW}⚠️  Unexpected response status: $STATUS${NC}"
        exit 0
    fi
else
    echo -e "${RED}❌ Registration failed (HTTP $HTTP_CODE)${NC}"
    echo ""
    echo "Common issues:"
    echo "  - Check that API server is running"
    echo "  - Verify API URL is correct: $API_URL"
    echo "  - Check network connectivity"
    exit 1
fi
