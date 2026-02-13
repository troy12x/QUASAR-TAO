#!/usr/bin/env python3
"""
Register a miner with the validator API (Simplified).

Usage:
    python scripts/register_miner.py --hotkey <hotkey_ss58> [options]
"""

import argparse
import bittensor as bt
import requests
import json
import sys

def register_miner(hotkey_ss58, wallet_name, hotkey_name, model_name, league, api_url):
    """Register a miner with the validator API."""
    
    print("=" * 63)
    print("  Miner Registration")
    print("=" * 63)
    print()
    print(f"Hotkey:      {hotkey_ss58}")
    print(f"Model:       {model_name}")
    print(f"League:      {league}")
    print(f"API URL:     {api_url}")
    print()
    
    # Load wallet
    try:
        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
        wallet_hotkey = wallet.hotkey.ss58_address
        
        # Verify hotkey matches
        if wallet_hotkey != hotkey_ss58:
            print(f"❌ Error: Hotkey mismatch")
            print(f"   Expected: {hotkey_ss58}")
            print(f"   Got:      {wallet_hotkey}")
            return False
    except Exception as e:
        print(f"❌ Error loading wallet: {e}")
        print(f"   Make sure wallet '{wallet_name}' with hotkey '{hotkey_name}' exists")
        return False
    
    print("✅ Wallet loaded")
    
    # Generate signature
    try:
        message = hotkey_ss58.encode('utf-8')
        signature = wallet.hotkey.sign(message)
        signature_hex = signature.hex()
    except Exception as e:
        print(f"❌ Error generating signature: {e}")
        return False
    
    print("✅ Signature generated")
    print()
    
    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "Hotkey": hotkey_ss58,
        "Signature": signature_hex
    }
    
    body = {
        "hotkey": hotkey_ss58,
        "model_name": model_name,
        "league": league
    }
    
    print("📤 Sending registration request...")
    print()
    
    # Make request
    try:
        response = requests.post(
            f"{api_url}/register_miner",
            headers=headers,
            json=body,
            timeout=30
        )
        
        print(f"📥 Response (HTTP {response.status_code}):")
        
        if response.status_code in [200, 201]:
            result = response.json()
            print(json.dumps(result, indent=2))
            print()
            
            status = result.get("status", "")
            
            if status == "registered":
                print("✅ Miner registered successfully!")
                print()
                print("You can now submit kernels via /submit_kernel endpoint")
                return True
            elif status == "already_registered":
                print("ℹ️  Miner already registered")
                print()
                print("Registration is active. You can submit kernels.")
                return True
            else:
                print(f"⚠️  Unexpected response status: {status}")
                return True
        else:
            print(response.text)
            print()
            print(f"❌ Registration failed (HTTP {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Register a miner with the validator API")
    parser.add_argument("--hotkey", required=True, help="Hotkey SS58 address (REQUIRED)")
    parser.add_argument("--wallet-name", default="quasar_miner", help="Wallet name (default: quasar_miner)")
    parser.add_argument("--hotkey-name", default="default", help="Hotkey name (default: default)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name (default: Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--league", default="100k", 
                       choices=["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "1M"],
                       help="League (default: 100k)")
    parser.add_argument("--api-url", default=None, help="API URL (default: auto-detect from .env)")
    
    args = parser.parse_args()
    
    # Auto-detect API URL if not provided
    api_url = args.api_url
    if not api_url:
        import os
        # Try .env file
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("VALIDATOR_API_URL="):
                        api_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        
        # Try environment variable
        if not api_url:
            api_url = os.environ.get("VALIDATOR_API_URL")
        
        # Default to local
        if not api_url or api_url == "http://localhost:8000":
            api_url = "http://localhost:8000"
    
    success = register_miner(
        hotkey_ss58=args.hotkey,
        wallet_name=args.wallet_name,
        hotkey_name=args.hotkey_name,
        model_name=args.model,
        league=args.league,
        api_url=api_url
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
