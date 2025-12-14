#!/usr/bin/env python3
"""
Test script for the enhanced configuration system.

This script demonstrates how to use the new configuration validation
and loading system for the unified HFA-SimpleMind subnet.
"""

import json
import os
import sys
from typing import Dict, Any

# Add the template directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'template'))

def test_configuration_system():
    """Test the enhanced configuration system."""
    print("Testing Enhanced Configuration System")
    print("=" * 50)
    
    config_dir = os.path.dirname(__file__)
    
    # Test 1: JSON syntax validation
    print("\n1. Testing JSON syntax validation...")
    config_files = ['hfa_config.json', 'subnet_config.json']
    
    for config_file in config_files:
        file_path = os.path.join(config_dir, config_file)
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            print(f"   ✓ {config_file} - Valid JSON syntax")
        except json.JSONDecodeError as e:
            print(f"   ✗ {config_file} - Invalid JSON: {e}")
        except FileNotFoundError:
            print(f"   ✗ {config_file} - File not found")
    
    # Test 2: Configuration structure validation
    print("\n2. Testing configuration structure...")
    
    # Load HFA config
    try:
        with open(os.path.join(config_dir, 'hfa_config.json'), 'r') as f:
            hfa_config = json.load(f)
        
        # Check required fields
        required_hfa_fields = ['model_name', 'supported_architectures', 'model_selection']
        missing_fields = [field for field in required_hfa_fields if field not in hfa_config]
        
        if not missing_fields:
            print("   ✓ HFA config - All required fields present")
        else:
            print(f"   ✗ HFA config - Missing fields: {missing_fields}")
        
        # Check supported architectures
        supported_archs = hfa_config.get('supported_architectures', {})
        expected_archs = ['hfa', 'simplemind', 'hybrid', 'standard']
        
        for arch in expected_archs:
            if arch in supported_archs:
                arch_config = supported_archs[arch]
                if 'enabled' in arch_config and 'default_config' in arch_config:
                    print(f"   ✓ {arch} architecture - Properly configured")
                else:
                    print(f"   ✗ {arch} architecture - Missing required fields")
            else:
                print(f"   ✗ {arch} architecture - Not found in config")
        
    except Exception as e:
        print(f"   ✗ HFA config validation failed: {e}")
    
    # Load subnet config
    try:
        with open(os.path.join(config_dir, 'subnet_config.json'), 'r') as f:
            subnet_config = json.load(f)
        
        # Check required fields
        required_subnet_fields = ['subnet_name', 'version', 'scoring_weights']
        missing_fields = [field for field in required_subnet_fields if field not in subnet_config]
        
        if not missing_fields:
            print("   ✓ Subnet config - All required fields present")
        else:
            print(f"   ✗ Subnet config - Missing fields: {missing_fields}")
        
        # Check scoring weights sum to 1.0
        scoring_weights = subnet_config.get('scoring_weights', {})
        total_weight = sum(scoring_weights.values())
        
        if abs(total_weight - 1.0) < 0.01:
            print(f"   ✓ Scoring weights - Sum to {total_weight:.3f} (valid)")
        else:
            print(f"   ✗ Scoring weights - Sum to {total_weight:.3f} (should be 1.0)")
        
    except Exception as e:
        print(f"   ✗ Subnet config validation failed: {e}")
    
    # Test 3: Architecture-specific configurations
    print("\n3. Testing architecture-specific configurations...")
    
    try:
        architectures = ['hfa', 'simplemind', 'hybrid', 'standard']
        
        for arch in architectures:
            arch_config = hfa_config['supported_architectures'].get(arch, {})
            default_config = arch_config.get('default_config', {})
            
            # Check common fields
            common_fields = ['vocab_size', 'd_model', 'max_seq_len']
            missing_common = [field for field in common_fields if field not in default_config]
            
            if not missing_common:
                print(f"   ✓ {arch} - Common fields present")
            else:
                print(f"   ✗ {arch} - Missing common fields: {missing_common}")
            
            # Check architecture-specific fields
            if arch == 'hfa':
                hfa_fields = ['num_layers', 'num_heads', 'd_ff']
                missing_hfa = [field for field in hfa_fields if field not in default_config]
                if not missing_hfa:
                    print(f"   ✓ {arch} - Architecture-specific fields present")
                else:
                    print(f"   ✗ {arch} - Missing fields: {missing_hfa}")
            
            elif arch == 'simplemind':
                sm_fields = ['num_layers', 'num_channels', 'router_type', 'aggregation_type']
                missing_sm = [field for field in sm_fields if field not in default_config]
                if not missing_sm:
                    print(f"   ✓ {arch} - Architecture-specific fields present")
                else:
                    print(f"   ✗ {arch} - Missing fields: {missing_sm}")
            
            elif arch == 'hybrid':
                hybrid_fields = ['mixing_strategy', 'hfa_config', 'simplemind_config']
                missing_hybrid = [field for field in hybrid_fields if field not in default_config]
                if not missing_hybrid:
                    print(f"   ✓ {arch} - Architecture-specific fields present")
                else:
                    print(f"   ✗ {arch} - Missing fields: {missing_hybrid}")
    
    except Exception as e:
        print(f"   ✗ Architecture config testing failed: {e}")
    
    # Test 4: Enhanced evaluation features
    print("\n4. Testing enhanced evaluation features...")
    
    try:
        # Check benchmark evaluation
        if 'benchmark_evaluation' in subnet_config:
            benchmark_config = subnet_config['benchmark_evaluation']
            if 'enabled_benchmarks' in benchmark_config:
                benchmarks = benchmark_config['enabled_benchmarks']
                print(f"   ✓ Benchmark evaluation - {len(benchmarks)} benchmarks configured")
            else:
                print("   ✗ Benchmark evaluation - No benchmarks configured")
        else:
            print("   ✗ Benchmark evaluation - Not configured")
        
        # Check perturbation testing
        if 'perturbation_testing' in subnet_config:
            perturbation_config = subnet_config['perturbation_testing']
            if perturbation_config.get('enabled', False):
                print("   ✓ Perturbation testing - Enabled")
            else:
                print("   ✗ Perturbation testing - Disabled")
        else:
            print("   ✗ Perturbation testing - Not configured")
        
        # Check diversity incentives
        if 'diversity_incentives' in subnet_config:
            diversity_config = subnet_config['diversity_incentives']
            if 'cosine_similarity_threshold' in diversity_config:
                threshold = diversity_config['cosine_similarity_threshold']
                print(f"   ✓ Diversity incentives - Threshold: {threshold}")
            else:
                print("   ✗ Diversity incentives - No threshold configured")
        else:
            print("   ✗ Diversity incentives - Not configured")
    
    except Exception as e:
        print(f"   ✗ Enhanced evaluation testing failed: {e}")
    
    print("\n" + "=" * 50)
    print("Configuration system test completed!")
    print("\nTo use the enhanced configuration system:")
    print("1. Import ConfigValidator to validate configurations")
    print("2. Import ConfigLoader to load and merge configurations")
    print("3. Use ModelArchitectureFactory.create_model_from_config_files()")
    print("4. Set environment variables or CLI args for overrides")


if __name__ == "__main__":
    test_configuration_system()