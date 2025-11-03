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
Configuration Validation for Unified HFA-SimpleMind Subnet

This module provides comprehensive validation for subnet configuration files,
including support for multiple architectures and enhanced evaluation settings.
"""

import json
import os
from typing import Dict, Any, List, Optional, Union
import bittensor as bt


class ConfigValidator:
    """
    Validates configuration files for the unified subnet.
    
    Supports validation of:
    - HFA configuration with multi-architecture support
    - Subnet configuration with enhanced evaluation settings
    - Model-specific configurations for each architecture type
    """
    
    SUPPORTED_ARCHITECTURES = ['hfa', 'simplemind', 'hybrid', 'standard']
    SUPPORTED_BENCHMARKS = ['longbench', 'hotpotqa_distractor', 'govreport', 'needle_in_haystack']
    SUPPORTED_PERTURBATION_TYPES = ['paraphrase', 'reorder', 'noise_injection']
    SUPPORTED_MIXING_STRATEGIES = ['alternating', 'parallel', 'sequential']
    SUPPORTED_ROUTER_TYPES = ['dynamic', 'static', 'learned']
    SUPPORTED_AGGREGATION_TYPES = ['learnable', 'attention', 'weighted_sum']
    
    @classmethod
    def validate_hfa_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Validate HFA configuration file.
        
        Args:
            config_path: Path to hfa_config.json file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"HFA config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in HFA config file: {e}")
        
        # Validate required top-level fields
        required_fields = [
            'model_name', 'max_context_length', 'architectures',
            'model_selection', 'evaluation_metrics'
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in HFA config: {field}")
        
        # Validate supported architectures
        cls._validate_architectures(config['architectures'])
        
        # Validate model selection configuration
        cls._validate_model_selection(config['model_selection'])
        
        # Validate evaluation metrics
        cls._validate_evaluation_metrics(config['evaluation_metrics'])
        
        # Validate numeric fields
        if not isinstance(config['max_context_length'], int) or config['max_context_length'] <= 0:
            raise ValueError("max_context_length must be a positive integer")
        
        if 'memory_retention_target' in config:
            if not isinstance(config['memory_retention_target'], (int, float)) or config['memory_retention_target'] <= 0:
                raise ValueError("memory_retention_target must be a positive number")
        
        bt.logging.info(f"HFA configuration validated successfully: {config_path}")
        return config
    
    @classmethod
    def validate_subnet_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Validate subnet configuration file.
        
        Args:
            config_path: Path to subnet_config.json file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Subnet config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in subnet config file: {e}")
        
        # Validate required top-level fields
        required_fields = [
            'subnet_name', 'version', 'evaluation_cycle_seconds',
            'max_miners_per_cycle', 'context_length_tests', 'scoring_weights'
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in subnet config: {field}")
        
        # Validate scoring weights
        cls._validate_scoring_weights(config['scoring_weights'])
        
        # Validate context length tests
        cls._validate_context_length_tests(config['context_length_tests'])
        
        # Validate architecture support if present
        if 'architecture_support' in config:
            cls._validate_architecture_support(config['architecture_support'])
        
        # Validate benchmark evaluation if present
        if 'benchmark_evaluation' in config:
            cls._validate_benchmark_evaluation(config['benchmark_evaluation'])
        
        # Validate perturbation testing if present
        if 'perturbation_testing' in config:
            cls._validate_perturbation_testing(config['perturbation_testing'])
        
        # Validate diversity incentives if present
        if 'diversity_incentives' in config:
            cls._validate_diversity_incentives(config['diversity_incentives'])
        
        # Validate numeric fields
        if not isinstance(config['evaluation_cycle_seconds'], (int, float)) or config['evaluation_cycle_seconds'] <= 0:
            raise ValueError("evaluation_cycle_seconds must be a positive number")
        
        if not isinstance(config['max_miners_per_cycle'], int) or config['max_miners_per_cycle'] <= 0:
            raise ValueError("max_miners_per_cycle must be a positive integer")
        
        bt.logging.info(f"Subnet configuration validated successfully: {config_path}")
        return config
    
    @classmethod
    def _validate_architectures(cls, architectures_config: Dict[str, Any]):
        """Validate supported architectures configuration."""
        for arch_name, arch_config in architectures_config.items():
            if arch_name not in cls.SUPPORTED_ARCHITECTURES:
                raise ValueError(f"Unsupported architecture: {arch_name}")
            
            if not isinstance(arch_config, dict):
                raise ValueError(f"Architecture config for {arch_name} must be a dictionary")
            
            if 'enabled' not in arch_config:
                raise ValueError(f"Missing 'enabled' field for architecture: {arch_name}")
            
            if 'default_config' not in arch_config:
                raise ValueError(f"Missing 'default_config' field for architecture: {arch_name}")
            
            # Validate architecture-specific default config
            cls._validate_architecture_default_config(arch_name, arch_config['default_config'])
    
    @classmethod
    def _validate_architecture_default_config(cls, arch_name: str, config: Dict[str, Any]):
        """Validate default configuration for a specific architecture."""
        # Common required fields
        common_fields = ['vocab_size', 'd_model', 'max_seq_len']
        for field in common_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in {arch_name} default config")
        
        # Architecture-specific validation
        if arch_name == 'hfa':
            hfa_fields = ['num_layers', 'num_heads', 'd_ff']
            for field in hfa_fields:
                if field not in config:
                    raise ValueError(f"Missing HFA field '{field}' in default config")
        
        elif arch_name == 'simplemind':
            simplemind_fields = ['num_layers', 'num_channels', 'router_type', 'aggregation_type']
            for field in simplemind_fields:
                if field not in config:
                    raise ValueError(f"Missing SimpleMind field '{field}' in default config")
            
            if config['router_type'] not in cls.SUPPORTED_ROUTER_TYPES:
                raise ValueError(f"Unsupported router_type: {config['router_type']}")
            
            if config['aggregation_type'] not in cls.SUPPORTED_AGGREGATION_TYPES:
                raise ValueError(f"Unsupported aggregation_type: {config['aggregation_type']}")
        
        elif arch_name == 'hybrid':
            hybrid_fields = ['mixing_strategy', 'hfa_config', 'simplemind_config']
            for field in hybrid_fields:
                if field not in config:
                    raise ValueError(f"Missing hybrid field '{field}' in default config")
            
            if config['mixing_strategy'] not in cls.SUPPORTED_MIXING_STRATEGIES:
                raise ValueError(f"Unsupported mixing_strategy: {config['mixing_strategy']}")
            
            # Validate sub-configs (these don't need vocab_size as they're nested)
            hfa_subconfig = config['hfa_config']
            hfa_required = ['num_layers', 'num_heads', 'd_ff']
            for field in hfa_required:
                if field not in hfa_subconfig:
                    raise ValueError(f"Missing HFA field '{field}' in hybrid hfa_config")
            
            simplemind_subconfig = config['simplemind_config']
            simplemind_required = ['num_layers', 'num_channels', 'router_type', 'aggregation_type']
            for field in simplemind_required:
                if field not in simplemind_subconfig:
                    raise ValueError(f"Missing SimpleMind field '{field}' in hybrid simplemind_config")
        
        elif arch_name == 'standard':
            standard_fields = ['num_layers', 'num_heads', 'd_ff']
            for field in standard_fields:
                if field not in config:
                    raise ValueError(f"Missing standard transformer field '{field}' in default config")
    
    @classmethod
    def _validate_model_selection(cls, selection_config: Dict[str, Any]):
        """Validate model selection configuration."""
        required_fields = ['default_architecture', 'architecture_preference_order']
        for field in required_fields:
            if field not in selection_config:
                raise ValueError(f"Missing model selection field: {field}")
        
        # Validate default architecture
        if selection_config['default_architecture'] not in cls.SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Invalid default_architecture: {selection_config['default_architecture']}")
        
        # Validate preference order
        preference_order = selection_config['architecture_preference_order']
        if not isinstance(preference_order, list):
            raise ValueError("architecture_preference_order must be a list")
        
        for arch in preference_order:
            if arch not in cls.SUPPORTED_ARCHITECTURES:
                raise ValueError(f"Invalid architecture in preference order: {arch}")
    
    @classmethod
    def _validate_evaluation_metrics(cls, metrics: List[str]):
        """Validate evaluation metrics list."""
        if not isinstance(metrics, list):
            raise ValueError("evaluation_metrics must be a list")
        
        valid_metrics = [
            'memory_retention_score', 'position_understanding_score', 'coherence_score',
            'tokens_per_second', 'scaling_efficiency', 'factual_accuracy_score',
            'diversity_bonus'
        ]
        
        for metric in metrics:
            if metric not in valid_metrics:
                bt.logging.warning(f"Unknown evaluation metric: {metric}")
    
    @classmethod
    def _validate_scoring_weights(cls, weights: Dict[str, float]):
        """Validate scoring weights configuration."""
        if not isinstance(weights, dict):
            raise ValueError("scoring_weights must be a dictionary")
        
        # Check that weights sum to approximately 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")
        
        # Check that all weights are non-negative
        for metric, weight in weights.items():
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Weight for {metric} must be a non-negative number")
    
    @classmethod
    def _validate_context_length_tests(cls, tests: List[int]):
        """Validate context length tests configuration."""
        if not isinstance(tests, list):
            raise ValueError("context_length_tests must be a list")
        
        for length in tests:
            if not isinstance(length, int) or length <= 0:
                raise ValueError("All context lengths must be positive integers")
        
        # Check that tests are in ascending order
        if tests != sorted(tests):
            raise ValueError("context_length_tests should be in ascending order")
    
    @classmethod
    def _validate_architecture_support(cls, support_config: Dict[str, Any]):
        """Validate architecture support configuration."""
        if 'enabled_architectures' in support_config:
            enabled = support_config['enabled_architectures']
            if not isinstance(enabled, list):
                raise ValueError("enabled_architectures must be a list")
            
            for arch in enabled:
                if arch not in cls.SUPPORTED_ARCHITECTURES:
                    raise ValueError(f"Unsupported architecture: {arch}")
    
    @classmethod
    def _validate_benchmark_evaluation(cls, benchmark_config: Dict[str, Any]):
        """Validate benchmark evaluation configuration."""
        if 'enabled_benchmarks' in benchmark_config:
            enabled = benchmark_config['enabled_benchmarks']
            if not isinstance(enabled, list):
                raise ValueError("enabled_benchmarks must be a list")
            
            for benchmark in enabled:
                if benchmark not in cls.SUPPORTED_BENCHMARKS:
                    raise ValueError(f"Unsupported benchmark: {benchmark}")
    
    @classmethod
    def _validate_perturbation_testing(cls, perturbation_config: Dict[str, Any]):
        """Validate perturbation testing configuration."""
        if 'perturbation_types' in perturbation_config:
            types = perturbation_config['perturbation_types']
            if not isinstance(types, list):
                raise ValueError("perturbation_types must be a list")
            
            for ptype in types:
                if ptype not in cls.SUPPORTED_PERTURBATION_TYPES:
                    raise ValueError(f"Unsupported perturbation type: {ptype}")
        
        if 'consistency_threshold' in perturbation_config:
            threshold = perturbation_config['consistency_threshold']
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                raise ValueError("consistency_threshold must be a number between 0 and 1")
    
    @classmethod
    def _validate_diversity_incentives(cls, diversity_config: Dict[str, Any]):
        """Validate diversity incentives configuration."""
        if 'cosine_similarity_threshold' in diversity_config:
            threshold = diversity_config['cosine_similarity_threshold']
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                raise ValueError("cosine_similarity_threshold must be a number between 0 and 1")
        
        if 'diversity_penalty_factor' in diversity_config:
            factor = diversity_config['diversity_penalty_factor']
            if not isinstance(factor, (int, float)) or factor < 0:
                raise ValueError("diversity_penalty_factor must be a non-negative number")
    
    @classmethod
    def validate_all_configs(cls, config_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        Validate all configuration files in the given directory.
        
        Args:
            config_dir: Directory containing configuration files
            
        Returns:
            Dictionary containing all validated configurations
        """
        configs = {}
        
        # Validate HFA config
        hfa_config_path = os.path.join(config_dir, 'hfa_config.json')
        if os.path.exists(hfa_config_path):
            configs['hfa'] = cls.validate_hfa_config(hfa_config_path)
        
        # Validate subnet config
        subnet_config_path = os.path.join(config_dir, 'subnet_config.json')
        if os.path.exists(subnet_config_path):
            configs['subnet'] = cls.validate_subnet_config(subnet_config_path)
        
        bt.logging.info(f"All configurations validated successfully in: {config_dir}")
        return configs
    
    @classmethod
    def get_model_config(cls, hfa_config: Dict[str, Any], architecture_type: str) -> Dict[str, Any]:
        """
        Extract model configuration for a specific architecture from HFA config.
        
        Args:
            hfa_config: Validated HFA configuration
            architecture_type: Type of architecture to get config for
            
        Returns:
            Model configuration dictionary for the specified architecture
        """
        if architecture_type not in cls.SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture type: {architecture_type}")
        
        if 'architectures' not in hfa_config:
            raise ValueError("HFA config missing architectures section")
        
        if architecture_type not in hfa_config['architectures']:
            raise ValueError(f"Architecture {architecture_type} not configured in HFA config")
        
        arch_config = hfa_config['architectures'][architecture_type]
        
        if not arch_config.get('enabled', False):
            raise ValueError(f"Architecture {architecture_type} is not enabled")
        
        return arch_config['default_config'].copy()