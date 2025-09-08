#!/usr/bin/env python3
"""
HFA Infinite Context Subnet - Environment Setup Script

This script sets up the development environment for the HFA subnet,
including Python path configuration and dependency verification.

Usage:
    python scripts/setup_environment.py
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_python_path():
    """Setup Python path to include quasar directory"""
    print("🔧 Setting up Python path...")
    
    # Get the quasar directory path
    quasar_path = Path(__file__).parent.parent.parent / "quasar"
    
    if quasar_path.exists():
        # Add to Python path
        sys.path.insert(0, str(quasar_path))
        
        # Create .env file for persistent path setup
        env_content = f"""# HFA Subnet Environment Configuration
PYTHONPATH={quasar_path}:{os.environ.get('PYTHONPATH', '')}
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        print(f"✅ Added {quasar_path} to Python path")
        print("✅ Created .env file for persistent configuration")
    else:
        print(f"⚠️ Warning: Quasar directory not found at {quasar_path}")
        print("   Make sure the HFA model files are available")

def verify_hfa_imports():
    """Verify that HFA components can be imported"""
    print("🔍 Verifying HFA component imports...")
    
    try:
        # Test import of hierarchical flow anchoring
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "quasar"))
        
        import hierarchical_flow_anchoring
        print("✅ hierarchical_flow_anchoring imported successfully")
        
        # Skip small_scale_pretraining import - contains heavy ML dependencies not needed for subnet
        print("✅ HFA core components available for subnet deployment")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure HFA model files are in the quasar directory")
        return False

def check_gpu_availability():
    """Check if GPU is available for training"""
    print("🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            
            # Check VRAM
            total_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"📊 GPU Memory: {total_memory / 1024**3:.1f} GB")
            
            return True
        else:
            print("⚠️ No GPU available - will use CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def create_config_files():
    """Create configuration files for the subnet"""
    print("📝 Creating configuration files...")
    
    # HFA model configuration
    hfa_config = {
        "model_name": "hfa_infinite_context",
        "max_context_length": 100000,
        "memory_retention_target": 1.0,
        "position_understanding_boost": 2.24,
        "scaling_efficiency": "linear",
        "checkpoint_interval": 1000,
        "evaluation_metrics": [
            "memory_retention_score",
            "position_understanding_score", 
            "coherence_score",
            "tokens_per_second",
            "scaling_efficiency"
        ]
    }
    
    import json
    with open("hfa_config.json", "w") as f:
        json.dump(hfa_config, f, indent=2)
    
    print("✅ Created hfa_config.json")
    
    # Subnet configuration
    subnet_config = {
        "subnet_name": "HFA Infinite Context",
        "description": "Breakthrough infinite context processing with Hierarchical Flow Anchoring",
        "version": "1.0.0",
        "evaluation_cycle_seconds": 90,
        "max_miners_per_cycle": 16,
        "context_length_tests": [1000, 5000, 15000, 50000, 100000],
        "scoring_weights": {
            "memory_retention_score": 0.35,
            "position_understanding_score": 0.25,
            "coherence_score": 0.20,
            "tokens_per_second": 0.10,
            "scaling_efficiency": 0.10
        }
    }
    
    with open("subnet_config.json", "w") as f:
        json.dump(subnet_config, f, indent=2)
    
    print("✅ Created subnet_config.json")

def setup_logging():
    """Setup logging configuration"""
    print("📋 Setting up logging configuration...")
    
    logging_config = """
[loggers]
keys=root,bittensor,hfa

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter,detailedFormatter

[logger_root]
level=INFO
handlers=consoleHandler,fileHandler

[logger_bittensor]
level=DEBUG
handlers=consoleHandler,fileHandler
qualname=bittensor
propagate=0

[logger_hfa]
level=DEBUG
handlers=consoleHandler,fileHandler
qualname=hfa
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=detailedFormatter
args=('hfa_subnet.log',)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

[formatter_detailedFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s
"""
    
    with open("logging.conf", "w") as f:
        f.write(logging_config)
    
    print("✅ Created logging.conf")

def main():
    print("🚀 HFA Infinite Context Subnet - Environment Setup")
    print("=" * 60)
    
    # Change to subnet directory
    script_dir = Path(__file__).parent
    subnet_dir = script_dir.parent
    os.chdir(subnet_dir)
    
    print(f"📁 Working directory: {subnet_dir}")
    
    # Setup steps
    setup_python_path()
    
    # Verify dependencies
    print("\n🔍 Verifying dependencies...")
    gpu_available = check_gpu_availability()
    hfa_available = verify_hfa_imports()
    
    # Create configuration files
    print("\n📝 Creating configuration files...")
    create_config_files()
    setup_logging()
    
    # Final status
    print("\n📊 Environment Setup Summary:")
    print("=" * 40)
    print(f"✅ Python path configured")
    print(f"{'✅' if gpu_available else '⚠️'} GPU: {'Available' if gpu_available else 'Not available (CPU mode)'}")
    print(f"{'✅' if hfa_available else '❌'} HFA imports: {'Working' if hfa_available else 'Failed'}")
    print(f"✅ Configuration files created")
    print(f"✅ Logging configured")
    
    if hfa_available:
        print("\n🎉 Environment setup complete!")
        print("Ready to deploy HFA Infinite Context Subnet")
    else:
        print("\n⚠️ Environment setup completed with warnings")
        print("Please ensure HFA model files are available before deployment")
    
    print("\n📋 Next Steps:")
    print("1. Run: python scripts/deploy_testnet.py --wallet-name <name> --hotkey-name <name>")
    print("2. Or run miner: python neurons/miner.py")
    print("3. Or run validator: python neurons/validator.py")

if __name__ == "__main__":
    main()
