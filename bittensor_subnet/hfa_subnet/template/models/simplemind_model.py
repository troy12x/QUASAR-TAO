# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 HFA Research Team

"""
SimpleMind Model Wrapper for Unified Subnet

This module provides a wrapper around the SimpleMind architecture to integrate
it with the unified subnet framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import bittensor as bt

from ..base_model import BaseModel, ModelOutput
from ...models.simplemind.model import MindTransformer


class SimpleMindModel(BaseModel):
    """
    SimpleMind model wrapper implementing the unified BaseModel interface.
    
    This wrapper integrates the SimpleMind block architecture with O(N) complexity
    into the unified subnet framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract SimpleMind-specific configuration
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.num_layers = config['num_layers']
        self.num_channels = config.get('num_channels', 64)
        self.max_seq_len = config.get('max_seq_len', 2048)
        self.router_type = config.get('router_type', 'dynamic')
        self.aggregation_type = config.get('aggregation_type', 'learnable')
        self.num_heads = config.get('num_heads', 8)
        self.d_ff = config.get('d_ff', 4 * self.d_model)
        self.dropout = config.get('dropout', 0.1)
        self.temperature = config.get('temperature', 1.0)
        self.channel_scaling = config.get('channel_scaling', 'sqrt')
        self.pad_token_id = config.get('pad_token_id', 0)
        
        # Create the underlying SimpleMind model
        self.model = MindTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_channels=self.num_channels,
            max_seq_len=self.max_seq_len,
            router_type=self.router_type,
            aggregation_type=self.aggregation_type,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            temperature=self.temperature,
            channel_scaling=self.channel_scaling,
            pad_token_id=self.pad_token_id
        )
        
        bt.logging.info(f"Initialized SimpleMind model with {self.count_parameters():,} parameters")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through the SimpleMind model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling loss [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            ModelOutput containing logits, loss, and complexity information
        """
        # Call the underlying SimpleMind model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        seq_len = input_ids.size(1)
        
        return ModelOutput(
            logits=outputs['logits'],
            loss=outputs.get('loss'),
            complexity_info=outputs.get('complexity', self.get_complexity_info(seq_len)),
            performance_metrics=self.get_performance_metrics(seq_len)
        )
    
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
        Generate text using the SimpleMind model.
        
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
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.pad_token_id,
            **kwargs
        )
    
    def get_complexity_info(self, seq_len: int) -> Dict[str, str]:
        """
        Get SimpleMind complexity information.
        
        Args:
            seq_len: Sequence length to analyze
            
        Returns:
            Dictionary containing complexity information
        """
        return {
            'architecture': 'SimpleMind',
            'total_complexity': f"O({self.num_layers} × {seq_len})",
            'per_layer_complexity': f"O({seq_len})",
            'vs_transformer': f"O({self.num_layers} × {seq_len}) vs O({self.num_layers} × {seq_len}²)",
            'scaling_advantage': f"{seq_len}x faster than transformer at seq_len={seq_len}",
            'memory_advantage': f"{seq_len}x less memory than transformer attention",
            'router_type': self.router_type,
            'aggregation_type': self.aggregation_type,
            'num_channels': str(self.num_channels)
        }
    
    def get_performance_metrics(self, seq_len: int) -> Dict[str, float]:
        """
        Get SimpleMind performance metrics.
        
        Args:
            seq_len: Sequence length to analyze
            
        Returns:
            Dictionary containing performance metrics
        """
        # Theoretical performance advantages of SimpleMind
        transformer_ops = seq_len ** 2 * self.num_layers
        simplemind_ops = seq_len * self.num_layers
        
        return {
            'theoretical_speedup': float(transformer_ops / simplemind_ops) if simplemind_ops > 0 else 1.0,
            'memory_efficiency': float(seq_len),  # Linear vs quadratic memory usage
            'parameter_efficiency': float(self.count_parameters() / (self.vocab_size * self.d_model)),
            'context_scaling': 1.0,  # Linear scaling with context length
            'num_channels': float(self.num_channels),
            'router_efficiency': 1.0,  # Placeholder for router-specific metrics
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get SimpleMind-specific model information."""
        base_info = super().get_model_info()
        base_info.update({
            'router_type': self.router_type,
            'aggregation_type': self.aggregation_type,
            'num_channels': self.num_channels,
            'channel_scaling': self.channel_scaling,
            'temperature': self.temperature,
            'complexity_class': 'O(N)',
            'architecture_family': 'SimpleMind'
        })
        return base_info