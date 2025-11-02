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
Unified Base Model Interface for HFA-SimpleMind Subnet

This module provides the base interface that all model architectures must implement
to work within the unified subnet framework.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class ModelOutput:
    """Standardized output format for all model architectures."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    complexity_info: Optional[Dict[str, str]] = None
    performance_metrics: Optional[Dict[str, float]] = None


class BaseModel(ABC, nn.Module):
    """
    Base interface for all model architectures in the unified subnet.
    
    This interface ensures compatibility between HFA, SimpleMind, hybrid,
    and standard transformer models within the subnet framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.architecture_type = config.get('architecture_type', 'unknown')
        
    @abstractmethod
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling loss [batch_size, seq_len]
            **kwargs: Additional model-specific arguments
            
        Returns:
            ModelOutput containing logits, loss, and additional information
        """
        pass
        
    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text continuation.
        
        Args:
            input_ids: Initial token IDs [batch_size, initial_seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs [batch_size, max_length]
        """
        pass
        
    @abstractmethod
    def get_complexity_info(self, seq_len: int) -> Dict[str, str]:
        """
        Get computational complexity information for the model.
        
        Args:
            seq_len: Sequence length to analyze
            
        Returns:
            Dictionary containing complexity information
        """
        pass
        
    @abstractmethod
    def get_performance_metrics(self, seq_len: int) -> Dict[str, float]:
        """
        Get performance metrics for the model at given sequence length.
        
        Args:
            seq_len: Sequence length to analyze
            
        Returns:
            Dictionary containing performance metrics
        """
        pass
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get general model information.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'architecture_type': self.architecture_type,
            'config': self.config,
            'parameter_count': self.count_parameters(),
            'model_size': self.get_model_size()
        }
        
    def count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
        
    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def get_model_size(self) -> str:
        """Get human-readable model size."""
        param_count = self.count_parameters()
        if param_count >= 1e9:
            return f"{param_count / 1e9:.1f}B"
        elif param_count >= 1e6:
            return f"{param_count / 1e6:.1f}M"
        elif param_count >= 1e3:
            return f"{param_count / 1e3:.1f}K"
        else:
            return str(param_count)


class ModelWrapper(BaseModel):
    """
    Base wrapper class for integrating existing models into the unified interface.
    
    This class provides common functionality for wrapping existing model implementations
    to conform to the BaseModel interface.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(config)
        self.model = model
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        """Default forward implementation that wraps the underlying model."""
        # This should be overridden by specific wrapper implementations
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            return ModelOutput(
                logits=outputs.get('logits'),
                loss=outputs.get('loss'),
                hidden_states=outputs.get('hidden_states'),
                complexity_info=self.get_complexity_info(input_ids.size(1)),
                performance_metrics=self.get_performance_metrics(input_ids.size(1))
            )
        elif isinstance(outputs, tuple):
            return ModelOutput(
                logits=outputs[1] if len(outputs) > 1 else outputs[0],
                loss=outputs[0] if len(outputs) > 1 and outputs[0] is not None else None,
                complexity_info=self.get_complexity_info(input_ids.size(1)),
                performance_metrics=self.get_performance_metrics(input_ids.size(1))
            )
        else:
            return ModelOutput(
                logits=outputs,
                complexity_info=self.get_complexity_info(input_ids.size(1)),
                performance_metrics=self.get_performance_metrics(input_ids.size(1))
            )