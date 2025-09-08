#!/usr/bin/env python3
"""
HFA Infinite Context Subnet - Testnet Deployment Script

This script automates the deployment of the HFA subnet on Bittensor testnet.
It handles wallet creation, subnet registration, and initial setup.

Usage:
    python scripts/deploy_testnet.py --wallet-name <wallet_name> --hotkey-name <hotkey_name>
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command and return result"""
    print(f"🔄 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"❌ Error details: {e.stderr.strip()}")
        if check:
            sys.exit(1)
        return e

def check_bittensor_installation():
    """Check if Bittensor is properly installed"""
    print("🔍 Checking Bittensor installation...")
    
    try:
        result = run_command("btcli --version", check=False)
        if result.returncode == 0:
            print("✅ Bittensor CLI is installed")
            return True
        else:
            print("❌ Bittensor CLI not found")
            return False
    except Exception as e:
        print(f"❌ Error checking Bittensor: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing HFA subnet dependencies...")
    
    # Install requirements
    run_command("pip install -r requirements.txt")
    
    # Install Bittensor if not present
    if not check_bittensor_installation():
        print("📦 Installing Bittensor...")
        run_command("pip install bittensor")

def create_wallet(wallet_name, hotkey_name):
    """Create Bittensor wallet for subnet operations"""
    print(f"🔑 Creating wallet: {wallet_name} with hotkey: {hotkey_name}")
    
    # Create coldkey
    print("Creating coldkey...")
    run_command(f"btcli wallet new_coldkey --wallet.name {wallet_name}")
    
    # Create hotkey
    print("Creating hotkey...")
    run_command(f"btcli wallet new_hotkey --wallet.name {wallet_name} --wallet.hotkey {hotkey_name}")
    
    print("✅ Wallet created successfully")

def get_test_tao(wallet_name, hotkey_name):
    """Get test TAO from faucet"""
    print("💰 Getting test TAO from faucet...")
    
    # Check current balance
    run_command(f"btcli wallet balance --wallet.name {wallet_name} --network test")
    
    print("📝 To get test TAO:")
    print("1. Join Bittensor Discord: https://discord.gg/bittensor")
    print("2. Go to #testnet-faucet channel")
    print("3. Request test TAO with your wallet address")
    print("4. Wait for confirmation")
    
    input("Press Enter after you've received test TAO...")

def check_subnet_burn_cost():
    """Check current subnet creation burn cost"""
    print("💸 Checking subnet burn cost...")
    
    result = run_command("btcli subnet burn-cost --network test")
    return result

def create_subnet(wallet_name, hotkey_name):
    """Create new subnet on testnet"""
    print("🏗️ Creating HFA Infinite Context subnet on testnet...")
    
    # Check burn cost first
    check_subnet_burn_cost()
    
    # Create subnet
    create_cmd = f"btcli subnet create --wallet.name {wallet_name} --wallet.hotkey {hotkey_name} --network test"
    result = run_command(create_cmd)
    
    if result.returncode == 0:
        print("✅ Subnet created successfully!")
        
        # Extract netuid from output (this is simplified - actual parsing may vary)
        print("📝 Note the netuid from the output above")
        netuid = input("Enter the netuid of your created subnet: ")
        return netuid
    else:
        print("❌ Subnet creation failed")
        return None

def register_miner(wallet_name, hotkey_name, netuid):
    """Register as miner on the subnet"""
    print(f"⛏️ Registering miner on subnet {netuid}...")
    
    register_cmd = f"btcli subnet register --wallet.name {wallet_name} --wallet.hotkey {hotkey_name} --netuid {netuid} --network test"
    run_command(register_cmd)
    
    print("✅ Miner registered successfully")

def register_validator(wallet_name, hotkey_name, netuid):
    """Register as validator on the subnet"""
    print(f"🔍 Registering validator on subnet {netuid}...")
    
    register_cmd = f"btcli subnet register --wallet.name {wallet_name} --wallet.hotkey {hotkey_name} --netuid {netuid} --network test"
    run_command(register_cmd)
    
    print("✅ Validator registered successfully")

def create_run_scripts(wallet_name, hotkey_name, netuid):
    """Create convenience scripts for running miner and validator"""
    
    # Miner run script
    miner_script = f"""#!/bin/bash
# HFA Miner Run Script
echo "🚀 Starting HFA Infinite Context Miner..."

python neurons/miner.py \\
    --netuid {netuid} \\
    --wallet.name {wallet_name} \\
    --wallet.hotkey {hotkey_name} \\
    --network test \\
    --logging.debug \\
    --axon.port 8091
"""
    
    with open("run_miner.sh", "w") as f:
        f.write(miner_script)
    
    # Validator run script
    validator_script = f"""#!/bin/bash
# HFA Validator Run Script
echo "🔍 Starting HFA Infinite Context Validator..."

python neurons/validator.py \\
    --netuid {netuid} \\
    --wallet.name {wallet_name} \\
    --wallet.hotkey {hotkey_name} \\
    --network test \\
    --logging.debug \\
    --neuron.sample_size 8
"""
    
    with open("run_validator.sh", "w") as f:
        f.write(validator_script)
    
    # Make scripts executable
    os.chmod("run_miner.sh", 0o755)
    os.chmod("run_validator.sh", 0o755)
    
    print("✅ Run scripts created: run_miner.sh, run_validator.sh")

def main():
    parser = argparse.ArgumentParser(description="Deploy HFA Infinite Context Subnet on Testnet")
    parser.add_argument("--wallet-name", required=True, help="Name for the wallet")
    parser.add_argument("--hotkey-name", required=True, help="Name for the hotkey")
    parser.add_argument("--skip-wallet", action="store_true", help="Skip wallet creation")
    parser.add_argument("--skip-tao", action="store_true", help="Skip test TAO acquisition")
    parser.add_argument("--netuid", help="Existing netuid if subnet already created")
    
    args = parser.parse_args()
    
    print("🚀 HFA Infinite Context Subnet - Testnet Deployment")
    print("=" * 60)
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Create wallet (if needed)
    if not args.skip_wallet:
        create_wallet(args.wallet_name, args.hotkey_name)
    
    # Step 3: Get test TAO (if needed)
    if not args.skip_tao:
        get_test_tao(args.wallet_name, args.hotkey_name)
    
    # Step 4: Create subnet (if netuid not provided)
    if args.netuid:
        netuid = args.netuid
        print(f"📝 Using existing subnet: {netuid}")
    else:
        netuid = create_subnet(args.wallet_name, args.hotkey_name)
        if not netuid:
            print("❌ Deployment failed - could not create subnet")
            sys.exit(1)
    
    # Step 5: Register on subnet
    print("\n🔧 Registration Options:")
    print("1. Register as miner")
    print("2. Register as validator") 
    print("3. Register as both (requires separate hotkeys)")
    
    choice = input("Choose registration type (1/2/3): ")
    
    if choice in ["1", "3"]:
        register_miner(args.wallet_name, args.hotkey_name, netuid)
    
    if choice in ["2", "3"]:
        if choice == "3":
            # For both, we need a separate validator hotkey
            validator_hotkey = f"{args.hotkey_name}_validator"
            print(f"Creating separate validator hotkey: {validator_hotkey}")
            run_command(f"btcli wallet new_hotkey --wallet.name {args.wallet_name} --wallet.hotkey {validator_hotkey}")
            register_validator(args.wallet_name, validator_hotkey, netuid)
        else:
            register_validator(args.wallet_name, args.hotkey_name, netuid)
    
    # Step 6: Create run scripts
    create_run_scripts(args.wallet_name, args.hotkey_name, netuid)
    
    # Step 7: Final instructions
    print("\n🎉 HFA Subnet Deployment Complete!")
    print("=" * 60)
    print(f"📊 Subnet NetUID: {netuid}")
    print(f"🔑 Wallet: {args.wallet_name}")
    print(f"🔥 Hotkey: {args.hotkey_name}")
    print(f"🌐 Network: Testnet")
    print("\n📋 Next Steps:")
    print("1. Run miner: ./run_miner.sh")
    print("2. Run validator: ./run_validator.sh")
    print("3. Monitor on Taostats: https://taostats.io/")
    print("4. Check logs for performance metrics")
    print("\n🔍 Monitoring Commands:")
    print(f"• Check balance: btcli wallet balance --wallet.name {args.wallet_name} --network test")
    print(f"• View subnet: btcli subnet list --network test")
    print(f"• Check registration: btcli subnet metagraph --netuid {netuid} --network test")

if __name__ == "__main__":
    main()
