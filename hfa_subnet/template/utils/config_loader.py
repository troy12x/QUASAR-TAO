# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 HFA Research Team

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Configuration Loader for Unified HFA-SimpleMind Subnet

This module provides utilities for loading and merging configuration files
with command-line arguments and environment variables.
"""

import json
import os
from typing import Dict, Any, Optional
import bittensor as bt
from .config_validator import ConfigValidator


class ConfigLoader:
    """
    Loads and merges configuration from multiple sources.
    
    Priority order (highest to lowest):
    1. Command-line arguments
    2. Environment variables
    3. Configuration files
    4. Default values
    """
    
    @classmethod
    def load_subnet_config(cls, config_dir: str, bt_config: "bt.Config") -> Dict[str, Any]:
        """
        Load and merge subnet configuration from all sources.
        
        Args:
            config_dir: Directory containing configuration files
            bt_config: Bittensor configuration object
            
        Returns:
            Merged configuration dictionary
        """
        # Load and validate configuration files
        validated_configs = ConfigValidator.validate_all_configs(config_dir)
        
        # Start with file-based configurations
        merged_config = {
            'hfa_config': validated_configs.get('hfa', {}),
            'subnet_config': validated_configs.get('subnet', {}),
        }
        
        # Apply command-line overrides
        cls._apply_cli_overrides(merged_config, bt_config)
        
        # Apply environment variable overrides
        cls._apply_env_overrides(merged_config)
        
        bt.logging.info("Configuration loaded and merged successfully")
        return merged_config
    
    @classmethod
    def get_model_config(
        cls, 
        merged_config: Dict[str, Any], 
        architecture_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model configuration for a specific architecture.
        
        Args:
            merged_config: Merged configuration from load_subnet_config
            architecture_type: Architecture type to get config for
            
        Returns:
            Model configuration dictionary
        """
        hfa_config = merged_config.get('hfa_config', {})
        
        # Determine architecture type
        if not architecture_type:
            model_selection = hfa_config.get('model_selection', {})
            architecture_type = model_selection.get('default_architecture', 'hfa')
        
        # Try to get base model config from HFA config
        try:
            if hfa_config and 'architectures' in hfa_config:
                model_config = ConfigValidator.get_model_config(hfa_config, architecture_type)
            else:
                # If hfa_config is empty or invalid, return a basic default config
                bt.logging.warning(f"HFA config empty or invalid, using basic default for {architecture_type}")
                model_config = cls._get_default_model_config(architecture_type)
        except Exception as e:
            bt.logging.warning(f"Failed to get model config from HFA config: {e}, using default")
            model_config = cls._get_default_model_config(architecture_type)
        
        # Apply any architecture-specific overrides from merged config
        arch_overrides = merged_config.get(f'{architecture_type}_overrides', {})
        model_config.update(arch_overrides)
        
        return model_config
    
    @classmethod
    def _get_default_model_config(cls, architecture_type: str) -> Dict[str, Any]:
        """
        Get a basic default configuration for an architecture.
        
        Args:
            architecture_type: Architecture type
            
        Returns:
            Default model configuration
        """
        common_config = {
            'vocab_size': 50257,
            'd_model': 512,
            'max_seq_len': 100000,
            'dropout': 0.1
        }
        
        if architecture_type == 'hfa':
            return {
                **common_config,
                'num_layers': 6,
                'num_heads': 8,
                'd_ff': 2048
            }
        elif architecture_type == 'simplemind':
            return {
                **common_config,
                'num_layers': 6,
                'num_channels': 64,
                'router_type': 'dynamic',
                'aggregation_type': 'learnable'
            }
        elif architecture_type == 'hybrid':
            return {
                **common_config,
                'mixing_strategy': 'alternating',
                'hfa_config': {
                    'num_layers': 3,
                    'num_heads': 8,
                    'd_ff': 2048
                },
                'simplemind_config': {
                    'num_layers': 3,
                    'num_channels': 64,
                    'router_type': 'dynamic',
                    'aggregation_type': 'learnable'
                }
            }
        else:  # standard
            return {
                **common_config,
                'num_layers': 6,
                'num_heads': 8,
                'd_ff': 2048
            }
    
    @classmethod
    def _apply_cli_overrides(cls, config: Dict[str, Any], bt_config: "bt.Config"):
        """Apply command-line argument overrides to configuration."""
        # Architecture selection override
        if hasattr(bt_config, 'neuron') and bt_config.neuron and hasattr(bt_config.neuron, 'model_architecture'):
            if 'model_selection' not in config['hfa_config']:
                config['hfa_config']['model_selection'] = {}
            config['hfa_config']['model_selection']['default_architecture'] = bt_config.neuron.model_architecture
        
        # Max context length override
        if hasattr(bt_config, 'neuron') and bt_config.neuron and hasattr(bt_config.neuron, 'max_context_length'):
            config['hfa_config']['max_context_length'] = bt_config.neuron.max_context_length
        
        # Hybrid mixing strategy override
        if hasattr(bt_config, 'neuron') and bt_config.neuron and hasattr(bt_config.neuron, 'hybrid_mixing_strategy'):
            hybrid_overrides = config.setdefault('hybrid_overrides', {})
            hybrid_overrides['mixing_strategy'] = bt_config.neuron.hybrid_mixing_strategy
        
        # Benchmark evaluation settings
        if hasattr(bt_config, 'neuron') and bt_config.neuron and hasattr(bt_config.neuron, 'enable_benchmark_evaluation'):
            if 'benchmark_evaluation' not in config['subnet_config']:
                config['subnet_config']['benchmark_evaluation'] = {}
            config['subnet_config']['benchmark_evaluation']['enabled'] = bt_config.neuron.enable_benchmark_evaluation
        
        if hasattr(bt_config, 'neuron') and bt_config.neuron and hasattr(bt_config.neuron, 'benchmark_types'):
            if 'benchmark_evaluation' not in config['subnet_config']:
                config['subnet_config']['benchmark_evaluation'] = {}
            config['subnet_config']['benchmark_evaluation']['enabled_benchmarks'] = bt_config.neuron.benchmark_types
        
        # Validator-specific overrides
        if hasattr(bt_config, 'validator') and bt_config.validator:
            validator_config = config['subnet_config'].setdefault('validator_overrides', {})
            
            if hasattr(bt_config.validator, 'perturbation_testing_frequency'):
                perturbation_config = validator_config.setdefault('perturbation_testing', {})
                perturbation_config['perturbation_frequency'] = bt_config.validator.perturbation_testing_frequency
            
            if hasattr(bt_config.validator, 'consensus_threshold'):
                perturbation_config = validator_config.setdefault('perturbation_testing', {})
                perturbation_config['consensus_threshold'] = bt_config.validator.consensus_threshold
            
            if hasattr(bt_config.validator, 'diversity_bonus_weight'):
                scoring_weights = validator_config.setdefault('scoring_weights', {})
                scoring_weights['diversity_bonus'] = bt_config.validator.diversity_bonus_weight
        
        # Miner-specific overrides
        if hasattr(bt_config, 'miner') and bt_config.miner:
            miner_config = config.setdefault('miner_overrides', {})
            
            if hasattr(bt_config.miner, 'preferred_architecture'):
                miner_config['preferred_architecture'] = bt_config.miner.preferred_architecture
            
            if hasattr(bt_config.miner, 'enable_model_switching'):
                miner_config['enable_model_switching'] = bt_config.miner.enable_model_switching
            
            if hasattr(bt_config.miner, 'model_checkpoint_path'):
                miner_config['model_checkpoint_path'] = bt_config.miner.model_checkpoint_path
        
        # Architecture config override (JSON string)
        if hasattr(bt_config, 'neuron') and bt_config.neuron and hasattr(bt_config.neuron, 'architecture_config_override') and bt_config.neuron.architecture_config_override:
            try:
                override_config = json.loads(bt_config.neuron.architecture_config_override)
                arch_type = config['hfa_config']['model_selection']['default_architecture']
                arch_overrides = config.setdefault(f'{arch_type}_overrides', {})
                arch_overrides.update(override_config)
            except json.JSONDecodeError as e:
                bt.logging.error(f"Invalid JSON in architecture_config_override: {e}")
    
    @classmethod
    def _apply_env_overrides(cls, config: Dict[str, Any]):
        """Apply environment variable overrides to configuration."""
        # Architecture selection
        env_architecture = os.getenv('SUBNET_MODEL_ARCHITECTURE')
        if env_architecture:
            if 'model_selection' not in config['hfa_config']:
                config['hfa_config']['model_selection'] = {}
            config['hfa_config']['model_selection']['default_architecture'] = env_architecture
        
        # Max context length
        env_max_context = os.getenv('SUBNET_MAX_CONTEXT_LENGTH')
        if env_max_context:
            try:
                config['hfa_config']['max_context_length'] = int(env_max_context)
            except ValueError:
                bt.logging.warning(f"Invalid SUBNET_MAX_CONTEXT_LENGTH: {env_max_context}")
        
        # Evaluation cycle seconds
        env_eval_cycle = os.getenv('SUBNET_EVALUATION_CYCLE_SECONDS')
        if env_eval_cycle:
            try:
                config['subnet_config']['evaluation_cycle_seconds'] = float(env_eval_cycle)
            except ValueError:
                bt.logging.warning(f"Invalid SUBNET_EVALUATION_CYCLE_SECONDS: {env_eval_cycle}")
        
        # Enable benchmark evaluation
        env_benchmark_eval = os.getenv('SUBNET_ENABLE_BENCHMARK_EVALUATION')
        if env_benchmark_eval:
            if 'benchmark_evaluation' not in config['subnet_config']:
                config['subnet_config']['benchmark_evaluation'] = {}
            config['subnet_config']['benchmark_evaluation']['enabled'] = env_benchmark_eval.lower() in ('true', '1', 'yes')
        
        # Enabled benchmarks
        env_benchmarks = os.getenv('SUBNET_ENABLED_BENCHMARKS')
        if env_benchmarks:
            if 'benchmark_evaluation' not in config['subnet_config']:
                config['subnet_config']['benchmark_evaluation'] = {}
            config['subnet_config']['benchmark_evaluation']['enabled_benchmarks'] = env_benchmarks.split(',')
        
        # Perturbation testing frequency
        env_perturbation_freq = os.getenv('SUBNET_PERTURBATION_FREQUENCY')
        if env_perturbation_freq:
            try:
                if 'perturbation_testing' not in config['subnet_config']:
                    config['subnet_config']['perturbation_testing'] = {}
                config['subnet_config']['perturbation_testing']['perturbation_frequency'] = float(env_perturbation_freq)
            except ValueError:
                bt.logging.warning(f"Invalid SUBNET_PERTURBATION_FREQUENCY: {env_perturbation_freq}")
    
    @classmethod
    def create_runtime_config(
        cls, 
        config_dir: str, 
        bt_config: "bt.Config",
        architecture_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a complete runtime configuration for the subnet.
        
        Args:
            config_dir: Directory containing configuration files
            bt_config: Bittensor configuration object
            architecture_type: Optional architecture type override
            
        Returns:
            Complete runtime configuration dictionary or None if configuration fails
        """
        try:
            # Load merged configuration
            merged_config = cls.load_subnet_config(config_dir, bt_config)
            
            # Check if we have valid hfa_config
            if not merged_config.get('hfa_config'):
                bt.logging.error("HFA configuration is missing or empty")
                return None
            
            # Determine architecture type with fallback
            if architecture_type:
                final_arch_type = architecture_type
            elif 'model_selection' in merged_config['hfa_config']:
                final_arch_type = merged_config['hfa_config']['model_selection'].get('default_architecture', 'hfa')
            else:
                final_arch_type = 'hfa'
            
            # Get enabled architectures from config
            if 'architectures' in merged_config.get('hfa_config', {}):
                enabled_archs = [
                    arch_name for arch_name, arch_config in merged_config['hfa_config']['architectures'].items()
                    if arch_config.get('enabled', False)
                ]
            else:
                enabled_archs = ['hfa', 'simplemind']
            
            # Build model configs for all enabled architectures
            model_configs = {}
            for arch in enabled_archs:
                try:
                    model_configs[arch] = cls.get_model_config(merged_config, arch)
                except Exception as e:
                    bt.logging.warning(f"Failed to get config for {arch}: {e}")
                    model_configs[arch] = cls._get_default_model_config(arch)
            
            # Create runtime configuration
            runtime_config = {
                'subnet': merged_config.get('subnet_config', {}),
                'hfa': merged_config['hfa_config'],
                'model': cls.get_model_config(merged_config, final_arch_type),
                'architecture_type': final_arch_type,
                'enabled_architectures': enabled_archs,
                'model_configs': model_configs,
                'bittensor': {
                    'netuid': getattr(bt_config, 'netuid', 1),
                    'network': getattr(bt_config.subtensor, 'network', 'test') if hasattr(bt_config, 'subtensor') else 'test',
                    'wallet_name': getattr(bt_config.wallet, 'name', 'default') if hasattr(bt_config, 'wallet') else 'default',
                    'wallet_hotkey': getattr(bt_config.wallet, 'hotkey', 'default') if hasattr(bt_config, 'wallet') else 'default',
                }
            }
            
            # Add miner-specific config if available
            if 'miner_overrides' in merged_config:
                runtime_config['miner'] = merged_config['miner_overrides']
            
            # Add validator-specific config if available
            if 'validator_overrides' in merged_config.get('subnet_config', {}):
                runtime_config['validator'] = merged_config['subnet_config']['validator_overrides']
            
            bt.logging.info(f"Runtime configuration created for architecture: {runtime_config['architecture_type']}")
            return runtime_config
            
        except Exception as e:
            bt.logging.error(f"Failed to create runtime configuration: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            return None
    
    @classmethod
    def save_runtime_config(cls, runtime_config: Dict[str, Any], output_path: str):
        """
        Save runtime configuration to a file.
        
        Args:
            runtime_config: Runtime configuration dictionary
            output_path: Path to save the configuration file
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(runtime_config, f, indent=2)
            bt.logging.info(f"Runtime configuration saved to: {output_path}")
        except Exception as e:
            bt.logging.error(f"Failed to save runtime configuration: {e}")
            raise