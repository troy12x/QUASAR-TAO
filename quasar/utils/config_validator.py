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
Configuration Validation for Quasar Subnet

This module provides comprehensive validation for subnet configuration files.
"""

import json
import os
from typing import Dict, Any, List, Optional, Union
import bittensor as bt


class ConfigValidator:
    """
    Validates configuration files for the Quasar subnet.
    
    Supports validation of:
    - Quasar configuration
    - Subnet configuration with evaluation settings
    - Model-specific configurations
    """
    
    SUPPORTED_ARCHITECTURES = ['deepseek_v32', 'kimi_linear', 'qwen3', 'qwen3_next']
    SUPPORTED_BENCHMARKS = ['longbench', 'hotpotqa_distractor', 'govreport', 'needle_in_haystack']
    SUPPORTED_PERTURBATION_TYPES = ['paraphrase', 'reorder', 'noise_injection']
    
    @classmethod
    def validate_quasar_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Validate Quasar configuration file.
        
        Args:
            config_path: Path to quasar_config.json file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Quasar config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in Quasar config file: {e}")
        
        # Validate required top-level fields
        required_fields = [
            'model_name', 'max_context_length', 'model',
            'evaluation_metrics'
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in HFA config: {field}")
        
        # Validate model configuration
        cls._validate_model_config(config['model'])
        
        # Validate evaluation metrics
        cls._validate_evaluation_metrics(config['evaluation_metrics'])
        
        # Validate numeric fields
        if not isinstance(config['max_context_length'], int) or config['max_context_length'] <= 0:
            raise ValueError("max_context_length must be a positive integer")
        
        if 'memory_retention_target' in config:
            if not isinstance(config['memory_retention_target'], (int, float)) or config['memory_retention_target'] <= 0:
                raise ValueError("memory_retention_target must be a positive number")
        
        bt.logging.info(f"Quasar configuration validated successfully: {config_path}")
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
            # Note: Legacy architectures are no longer restricted here
            pass
        
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
    def _validate_model_config(cls, config: Dict[str, Any]):
        """Validate Quasar model configuration."""
        # Required fields
        required_fields = ['vocab_size', 'd_model', 'max_seq_len', 'num_layers', 'num_heads', 'd_ff']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in model config")
    
    
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
        
        # Validate Quasar config
        quasar_config_path = os.path.join(config_dir, 'quasar_config.json')
        if os.path.exists(quasar_config_path):
            configs['quasar'] = cls.validate_quasar_config(quasar_config_path)
        
        # Validate subnet config
        subnet_config_path = os.path.join(config_dir, 'subnet_config.json')
        if os.path.exists(subnet_config_path):
            configs['subnet'] = cls.validate_subnet_config(subnet_config_path)
        
        bt.logging.info(f"All configurations validated successfully in: {config_dir}")
        return configs
    
    @classmethod
    def get_model_config(cls, quasar_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract model configuration from Quasar config.
        
        Args:
            quasar_config: Validated Quasar configuration
            
        Returns:
            Model configuration dictionary
        """
        if 'model' not in quasar_config:
            raise ValueError("Quasar config missing 'model' section")
        
        return quasar_config['model'].copy()