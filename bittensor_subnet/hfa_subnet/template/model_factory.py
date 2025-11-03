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
Model Architecture Factory for Unified HFA-SimpleMind Subnet

This module provides the factory pattern for creating different model architectures
within the unified subnet framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type
import bittensor as bt

from .base_model import BaseModel, ModelOutput


class ModelArchitectureFactory:
    """
    Factory for creating different model architectures in the unified subnet.
    
    Supports:
    - HFA (Hierarchical Flow Anchoring) models
    - SimpleMind block models  
    - Hybrid models combining HFA and SimpleMind
    - Standard transformer models for comparison
    """
    
    # Registry of available model architectures
    _model_registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register_model(cls, architecture_type: str, model_class: Type[BaseModel]):
        """
        Register a new model architecture.
        
        Args:
            architecture_type: String identifier for the architecture
            model_class: Model class that implements BaseModel interface
        """
        cls._model_registry[architecture_type.lower()] = model_class
        bt.logging.info(f"Registered model architecture: {architecture_type}")
    
    @classmethod
    def get_available_architectures(cls) -> list[str]:
        """Get list of available model architectures."""
        cls._lazy_load_models()
        return list(cls._model_registry.keys())
    
    @classmethod
    def _lazy_load_models(cls):
        """Lazy load model classes to avoid circular imports."""
        if not cls._model_registry:
            bt.logging.info("[ModelFactory] 🔄 Loading model classes...")
            
            # Load each model with error handling
            models_to_load = [
                ('hfa', '.models.hfa_model', 'HFAModel'),
                ('simplemind', '.models.simplemind_model', 'SimpleMindModel'),
                ('hybrid', '.models.hybrid_model', 'HybridModel'),
                ('standard', '.models.standard_model', 'StandardTransformerModel'),
            ]
            
            import time
            for arch_name, module_name, class_name in models_to_load:
                try:
                    bt.logging.info(f"[ModelFactory]   Importing {arch_name} from {module_name}...")
                    start = time.time()
                    module = __import__(f'template{module_name}', fromlist=[class_name])
                    bt.logging.info(f"[ModelFactory]   Module imported in {time.time() - start:.2f}s")
                    
                    bt.logging.info(f"[ModelFactory]   Getting class {class_name}...")
                    model_class = getattr(module, class_name)
                    cls._model_registry[arch_name] = model_class
                    bt.logging.info(f"[ModelFactory]   ✅ Loaded {arch_name} model class")
                except Exception as e:
                    bt.logging.warning(f"[ModelFactory]   ⚠️ Failed to load {arch_name} model: {e}")
                    import traceback
                    bt.logging.debug(f"[ModelFactory]   Traceback: {traceback.format_exc()}")
                    continue
            
            # Add alias
            if 'standard' in cls._model_registry:
                cls._model_registry['transformer'] = cls._model_registry['standard']
                bt.logging.info(f"[ModelFactory]   Added 'transformer' alias for 'standard'")
            
            bt.logging.info(f"[ModelFactory] ✅ Loaded {len(cls._model_registry)} model architectures: {list(cls._model_registry.keys())}")
        else:
            bt.logging.info(f"[ModelFactory] Model registry already loaded with {len(cls._model_registry)} architectures")
    
    @classmethod
    def create_model(cls, architecture_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on architecture type and configuration.
        
        Args:
            architecture_type: Type of architecture ('hfa', 'simplemind', 'hybrid', 'standard')
            config: Configuration dictionary for the model
            
        Returns:
            BaseModel instance of the specified architecture
            
        Raises:
            ValueError: If architecture_type is not supported
            Exception: If model creation fails
        """
        bt.logging.info(f"[ModelFactory] create_model called for: {architecture_type}")
        architecture_type = architecture_type.lower()
        
        # Lazy load models if not already loaded
        bt.logging.info(f"[ModelFactory] Lazy loading model classes...")
        cls._lazy_load_models()
        bt.logging.info(f"[ModelFactory] Model registry has {len(cls._model_registry)} entries")
        
        if architecture_type not in cls._model_registry:
            available = ', '.join(cls._model_registry.keys())
            raise ValueError(
                f"Unsupported architecture type: {architecture_type}. "
                f"Available architectures: {available}"
            )
        
        try:
            # Validate configuration before creating model
            bt.logging.info(f"[ModelFactory] Validating config for {architecture_type}...")
            cls.validate_config(architecture_type, config)
            bt.logging.info(f"[ModelFactory] Config validated successfully")
            
            # Add architecture type to config
            config = config.copy()
            config['architecture_type'] = architecture_type
            
            # Create model instance
            bt.logging.info(f"[ModelFactory] Getting model class for {architecture_type}...")
            model_class = cls._model_registry[architecture_type]
            bt.logging.info(f"[ModelFactory] Model class: {model_class.__name__}")
            
            bt.logging.info(f"[ModelFactory] Instantiating {architecture_type} model...")
            import time
            start = time.time()
            model = model_class(config)
            bt.logging.info(f"[ModelFactory] Model instantiated in {time.time() - start:.2f}s")
            
            param_count = model.count_parameters()
            bt.logging.info(
                f"[ModelFactory] Created {architecture_type} model with {param_count:,} parameters"
            )
            
            return model
            
        except Exception as e:
            bt.logging.error(f"[ModelFactory] Failed to create {architecture_type} model: {e}")
            import traceback
            bt.logging.error(f"[ModelFactory] Traceback: {traceback.format_exc()}")
            raise
    
    @classmethod
    def validate_config(cls, architecture_type: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a specific architecture type.
        
        Args:
            architecture_type: Type of architecture to validate for
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Common required fields
        required_fields = ['vocab_size', 'd_model']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Architecture-specific validation
        if architecture_type == 'hfa':
            cls._validate_hfa_config(config)
        elif architecture_type == 'simplemind':
            cls._validate_simplemind_config(config)
        elif architecture_type == 'hybrid':
            cls._validate_hybrid_config(config)
        elif architecture_type in ['standard', 'transformer']:
            cls._validate_standard_config(config)
        
        return True
    
    @classmethod
    def _validate_hfa_config(cls, config: Dict[str, Any]):
        """Validate HFA-specific configuration."""
        hfa_fields = ['num_layers', 'num_heads']
        for field in hfa_fields:
            if field not in config:
                raise ValueError(f"Missing HFA config field: {field}")
        
        # HFA-specific validation logic
        if config.get('num_layers', 0) <= 0:
            raise ValueError("HFA num_layers must be positive")
        if config.get('num_heads', 0) <= 0:
            raise ValueError("HFA num_heads must be positive")
    
    @classmethod
    def _validate_simplemind_config(cls, config: Dict[str, Any]):
        """Validate SimpleMind-specific configuration."""
        simplemind_fields = ['num_layers', 'num_channels']
        for field in simplemind_fields:
            if field not in config:
                raise ValueError(f"Missing SimpleMind config field: {field}")
        
        # SimpleMind-specific validation logic
        if config.get('num_layers', 0) <= 0:
            raise ValueError("SimpleMind num_layers must be positive")
        if config.get('num_channels', 0) <= 0:
            raise ValueError("SimpleMind num_channels must be positive")
    
    @classmethod
    def _validate_hybrid_config(cls, config: Dict[str, Any]):
        """Validate hybrid model configuration."""
        hybrid_fields = ['hfa_config', 'simplemind_config', 'mixing_strategy']
        for field in hybrid_fields:
            if field not in config:
                raise ValueError(f"Missing hybrid config field: {field}")
        
        # Validate sub-configurations
        cls._validate_hfa_config(config['hfa_config'])
        cls._validate_simplemind_config(config['simplemind_config'])
        
        # Validate mixing strategy
        valid_strategies = ['alternating', 'parallel', 'sequential']
        if config['mixing_strategy'] not in valid_strategies:
            raise ValueError(f"Invalid mixing strategy. Must be one of: {valid_strategies}")
    
    @classmethod
    def _validate_standard_config(cls, config: Dict[str, Any]):
        """Validate standard transformer configuration."""
        standard_fields = ['num_layers', 'num_heads']
        for field in standard_fields:
            if field not in config:
                raise ValueError(f"Missing standard transformer config field: {field}")
        
        # Standard transformer validation logic
        if config.get('num_layers', 0) <= 0:
            raise ValueError("Standard transformer num_layers must be positive")
        if config.get('num_heads', 0) <= 0:
            raise ValueError("Standard transformer num_heads must be positive")
    
    @classmethod
    def create_model_from_checkpoint(
        cls, 
        checkpoint_path: str, 
        architecture_type: Optional[str] = None
    ) -> BaseModel:
        """
        Create model from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            architecture_type: Optional architecture type override
            
        Returns:
            BaseModel instance loaded from checkpoint
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract configuration and architecture type from checkpoint
            config = checkpoint.get('config', {})
            arch_type = architecture_type or checkpoint.get('architecture_type') or config.get('architecture_type')
            
            if not arch_type:
                raise ValueError("Architecture type not found in checkpoint or provided")
            
            # Create model and load state dict
            model = cls.create_model(arch_type, config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            bt.logging.info(f"Loaded {arch_type} model from checkpoint: {checkpoint_path}")
            return model
            
        except Exception as e:
            bt.logging.error(f"Failed to load model from checkpoint {checkpoint_path}: {e}")
            raise
    
    @classmethod
    def get_default_config(cls, architecture_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific architecture type.
        
        Args:
            architecture_type: Type of architecture
            
        Returns:
            Default configuration dictionary
        """
        base_config = {
            'vocab_size': 50257,  # GPT-2 vocab size
            'd_model': 512,
            'max_seq_len': 2048,
            'dropout': 0.1,
        }
        
        if architecture_type == 'hfa':
            base_config.update({
                'num_layers': 6,
                'num_heads': 8,
                'd_ff': 2048,
            })
        elif architecture_type == 'simplemind':
            base_config.update({
                'num_layers': 6,
                'num_channels': 64,
                'router_type': 'dynamic',
                'aggregation_type': 'learnable',
            })
        elif architecture_type == 'hybrid':
            base_config.update({
                'hfa_config': cls.get_default_config('hfa'),
                'simplemind_config': cls.get_default_config('simplemind'),
                'mixing_strategy': 'alternating',
            })
        elif architecture_type in ['standard', 'transformer']:
            base_config.update({
                'num_layers': 6,
                'num_heads': 8,
                'd_ff': 2048,
            })
        
        return base_config
    
    @classmethod
    def create_model_from_config_files(
        cls, 
        config_dir: str, 
        architecture_type: Optional[str] = None,
        bt_config: Optional[Any] = None
    ) -> BaseModel:
        """
        Create a model using configuration files and optional overrides.
        
        Args:
            config_dir: Directory containing configuration files
            architecture_type: Optional architecture type override
            bt_config: Optional Bittensor configuration for overrides
            
        Returns:
            BaseModel instance configured from files
        """
        try:
            from .utils.config_loader import ConfigLoader
            
            # Create runtime configuration
            runtime_config = ConfigLoader.create_runtime_config(
                config_dir, bt_config or type('MockConfig', (), {'netuid': 1})(), architecture_type
            )
            
            # Extract model configuration
            model_config = runtime_config['model']
            arch_type = runtime_config['architecture_type']
            
            # Create model
            model = cls.create_model(arch_type, model_config)
            
            bt.logging.info(f"Model created from config files: {arch_type}")
            return model
            
        except ImportError:
            bt.logging.warning("ConfigLoader not available, falling back to default config")
            arch_type = architecture_type or 'hfa'
            default_config = cls.get_default_config(arch_type)
            return cls.create_model(arch_type, default_config)
        except Exception as e:
            bt.logging.error(f"Failed to create model from config files: {e}")
            raise