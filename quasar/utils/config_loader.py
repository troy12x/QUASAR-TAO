# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2026 SILX AI Research Team

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
Configuration Loader for Quasar Subnet

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
            'quasar_config': validated_configs.get('quasar', {}),
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
        # We no longer support multiple architectures.
        # This returns the primary Quasar model configuration.
        quasar_config = merged_config.get('quasar_config', {})
        
        # Try to get model config from Quasar config
        try:
            if quasar_config:
                model_config = quasar_config.get('model', {})
                if not model_config:
                    raise ValueError("Quasar config misses 'model' section")
            else:
                raise ValueError("Quasar configuration is missing or empty")
        except Exception as e:
            bt.logging.error(f"Failed to get model config: {e}")
            raise
        
        # Apply any overrides from merged config
        overrides = merged_config.get('model_overrides', {})
        model_config.update(overrides)
        
        return model_config
    
    
    @classmethod
    def _apply_cli_overrides(cls, config: Dict[str, Any], bt_config: "bt.Config"):
        """Apply command-line argument overrides to configuration."""
        # Max context length override
        if hasattr(bt_config, 'neuron') and bt_config.neuron and hasattr(bt_config.neuron, 'max_context_length'):
            config['quasar_config']['max_context_length'] = bt_config.neuron.max_context_length
        
        # Baseline comparison override
        if hasattr(bt_config, 'neuron') and bt_config.neuron and hasattr(bt_config.neuron, 'baseline_comparison_required'):
            config['subnet_config']['architecture_support']['baseline_comparison_required'] = bt_config.neuron.baseline_comparison_required
        
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
        # Miner-specific overrides
        if hasattr(bt_config, 'miner') and bt_config.miner:
            miner_config = config.setdefault('miner_overrides', {})
            
            if hasattr(bt_config.miner, 'model_checkpoint_path'):
                # Note: This would typically be a local path or HF identifier
                miner_config['model_checkpoint_path'] = bt_config.miner.model_checkpoint_path
        
        # Architecture config override (JSON string) - renamed to model_config_override
        if hasattr(bt_config, 'neuron') and bt_config.neuron and hasattr(bt_config.neuron, 'architecture_config_override') and bt_config.neuron.architecture_config_override:
            try:
                override_config = json.loads(bt_config.neuron.architecture_config_override)
                model_overrides = config.setdefault('model_overrides', {})
                model_overrides.update(override_config)
            except json.JSONDecodeError as e:
                bt.logging.error(f"Invalid JSON in architecture_config_override: {e}")
    
    @classmethod
    def _apply_env_overrides(cls, config: Dict[str, Any]):
        """Apply environment variable overrides to configuration."""
        # Architecture selection
        # Environment selection
        env_arch = os.getenv('QUASAR_MODEL_CONFIG')
        if env_arch:
            try:
                config['quasar_config']['model'] = json.loads(env_arch)
            except:
                bt.logging.warning(f"Invalid QUASAR_MODEL_CONFIG JSON")
        
        # Max context length
        env_max_context = os.getenv('QUASAR_MAX_CONTEXT_LENGTH')
        if env_max_context:
            try:
                config['quasar_config']['max_context_length'] = int(env_max_context)
            except ValueError:
                bt.logging.warning(f"Invalid QUASAR_MAX_CONTEXT_LENGTH: {env_max_context}")
        
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
        bt_config: "bt.Config"
    ) -> Dict[str, Any]:
        """
        Create a complete runtime configuration for the Quasar subnet.
        
        Args:
            config_dir: Directory containing configuration files
            bt_config: Bittensor configuration object
            
        Returns:
            Complete runtime configuration dictionary or None if configuration fails
        """
        try:
            # Load merged configuration
            merged_config = cls.load_subnet_config(config_dir, bt_config)
            
            # Check if we have valid quasar_config
            if not merged_config.get('quasar_config'):
                bt.logging.error("Quasar configuration is missing or empty")
                return None
            
            # Create runtime configuration
            runtime_config = {
                'subnet': merged_config.get('subnet_config', {}),
                'quasar': merged_config['quasar_config'],
                'model': cls.get_model_config(merged_config),
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
            
            bt.logging.info("Runtime configuration created successfully")
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