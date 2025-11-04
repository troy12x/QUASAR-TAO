#!/usr/bin/env python3
"""
Run HFA-SimpleMind Subnet Validator

This script starts a validator node for the unified HFA-SimpleMind subnet.
The validator evaluates miners using real-world benchmarks and diversity tracking.

Usage:
    python scripts/run_validator.py --netuid 1 --subtensor.network finney --wallet.name my_wallet --wallet.hotkey my_hotkey
    python scripts/run_validator.py --netuid 1 --subtensor.network test --wallet.name test_wallet --wallet.hotkey test_hotkey --benchmarks.enabled
    python scripts/run_validator.py --help  # Show all options
"""

import argparse
import sys
import os

# Add the subnet directory to Python path
subnet_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, subnet_dir)

import bittensor as bt
from neurons.validator import HFAValidator


def get_config():
    """Get configuration from command line arguments"""
    
    parser = argparse.ArgumentParser(description="HFA-SimpleMind Subnet Validator")
    
    # Add Bittensor arguments
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    
    # Add subnet-specific arguments
    parser.add_argument("--netuid", type=int, default=1, help="Subnet netuid")
    
    # Evaluation settings
    parser.add_argument("--evaluation.tasks_per_cycle", type=int, default=8,
                       help="Number of tasks per evaluation cycle")
    parser.add_argument("--evaluation.cycle_interval", type=int, default=60,
                       help="Seconds between evaluation cycles")
    
    # Benchmark settings
    parser.add_argument("--benchmarks.enabled", action="store_true", default=True,
                       help="Enable real-world benchmark evaluation")
    parser.add_argument("--benchmarks.data_path", type=str, default="data/benchmarks",
                       help="Path to benchmark data")
    parser.add_argument("--benchmarks.task_ratio", type=float, default=0.6,
                       help="Ratio of benchmark tasks vs synthetic tasks")
    
    # Diversity tracking settings
    parser.add_argument("--diversity.enabled", action="store_true", default=True,
                       help="Enable diversity tracking and incentives")
    parser.add_argument("--diversity.penalty_factor", type=float, default=0.2,
                       help="Penalty factor for low diversity")
    parser.add_argument("--diversity.bonus_factor", type=float, default=0.1,
                       help="Bonus factor for high diversity")
    
    # Monitoring settings
    parser.add_argument("--monitoring.enabled", action="store_true", default=True,
                       help="Enable monitoring and telemetry")
    parser.add_argument("--monitoring.port", type=int, default=8081,
                       help="Monitoring dashboard port")
    
    # Scoring settings
    parser.add_argument("--scoring.sealed_harness", action="store_true", default=True,
                       help="Use sealed scoring harness")
    parser.add_argument("--scoring.audit_enabled", action="store_true", default=True,
                       help="Enable audit trail logging")
    
    config = bt.config(parser)
    
    return config


def main():
    """Main validator entry point"""
    
    # Get configuration
    config = get_config()
    
    # Set up logging
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info("Starting HFA-SimpleMind Subnet Validator")
    bt.logging.info(f"Configuration: {config}")
    
    try:
        # Create and run validator
        validator = HFAValidator(config=config)
        
        bt.logging.info("🔍 Starting validator with comprehensive evaluation")
        bt.logging.info(f"📡 Network: {config.subtensor.network}")
        bt.logging.info(f"🔑 Wallet: {config.wallet.name}.{config.wallet.hotkey}")
        bt.logging.info(f"🌐 Netuid: {config.netuid}")
        bt.logging.info(f"📊 Benchmarks: {'Enabled' if config.benchmarks.enabled else 'Disabled'}")
        bt.logging.info(f"🎯 Diversity tracking: {'Enabled' if config.diversity.enabled else 'Disabled'}")
        bt.logging.info(f"🔒 Sealed scoring: {'Enabled' if config.scoring.sealed_harness else 'Disabled'}")
        
        # Start the validator
        validator.run()
        
    except KeyboardInterrupt:
        bt.logging.info("⏹️ Validator stopped by user")
    except Exception as e:
        bt.logging.error(f"💥 Validator failed: {e}")
        raise
    finally:
        bt.logging.info("🔚 Validator shutdown complete")


if __name__ == "__main__":
    main()