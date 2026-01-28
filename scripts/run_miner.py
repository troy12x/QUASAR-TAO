#!/usr/bin/env python3
"""
Run HFA-SimpleMind Subnet Miner

This script starts a miner node for the unified HFA-SimpleMind subnet.
The miner can run HFA, SimpleMind, or hybrid models based on configuration.

Usage:
    python scripts/run_miner.py --netuid 1 --subtensor.network finney --wallet.name my_wallet --wallet.hotkey my_hotkey
    python scripts/run_miner.py --netuid 1 --subtensor.network test --wallet.name test_wallet --wallet.hotkey test_hotkey --model.architecture simplemind
    python scripts/run_miner.py --help  # Show all options
"""

import argparse
import sys
import os

# Add the subnet directory to Python path
subnet_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, subnet_dir)

import bittensor as bt
from neurons.miner import HFAMiner


def get_config():
    """Get configuration from command line arguments"""
    
    parser = argparse.ArgumentParser(description="HFA-SimpleMind Subnet Miner")
    
    # Add Bittensor arguments
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    
    # Add subnet-specific arguments
    parser.add_argument("--netuid", type=int, default=1, help="Subnet netuid")
    
    # Model configuration
    parser.add_argument("--model.architecture", type=str, default="hfa", 
                       choices=["hfa", "simplemind", "hybrid"],
                       help="Model architecture to use")
    parser.add_argument("--model.name", type=str, default="hfa_model",
                       help="Model name/path")
    parser.add_argument("--model.max_context_length", type=int, default=32768,
                       help="Maximum context length")
    parser.add_argument("--model.skip_loading", action="store_true",
                       help="Skip model loading for testing (uses mock responses)")
    
    # Performance settings
    parser.add_argument("--performance.batch_size", type=int, default=1,
                       help="Inference batch size")
    parser.add_argument("--performance.max_workers", type=int, default=4,
                       help="Maximum worker threads")
    
    # Monitoring settings
    parser.add_argument("--monitoring.enabled", action="store_true", default=True,
                       help="Enable monitoring and telemetry")
    parser.add_argument("--monitoring.port", type=int, default=8080,
                       help="Monitoring dashboard port")
    
    config = bt.config(parser)
    
    return config


def main():
    """Main miner entry point"""
    
    # Get configuration
    config = get_config()
    
    # Set up logging
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info("Starting HFA-SimpleMind Subnet Miner")
    bt.logging.info(f"Configuration: {config}")
    
    try:
        # Create and run miner
        miner = HFAMiner(config=config)
        
        bt.logging.info(f"🚀 Starting miner with {config.model.architecture} architecture")
        bt.logging.info(f"📡 Network: {config.subtensor.network}")
        bt.logging.info(f"🔑 Wallet: {config.wallet.name}.{config.wallet.hotkey}")
        bt.logging.info(f"🌐 Netuid: {config.netuid}")
        
        # Start the miner
        miner.run()
        
    except KeyboardInterrupt:
        bt.logging.info("⏹️ Miner stopped by user")
    except Exception as e:
        bt.logging.error(f"💥 Miner failed: {e}")
        raise
    finally:
        bt.logging.info("🔚 Miner shutdown complete")


if __name__ == "__main__":
    main()