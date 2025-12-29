# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 HFA Research Team

"""
Standard Transformer Model for Unified Subnet

This module provides a standard transformer implementation for comparison
with HFA and SimpleMind architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import bittensor as bt

from ..base_model import BaseModel, ModelOutput


class StandardTransformerModel(BaseModel):
    """
    Standard transformer model implementing the unified BaseModel interface.
    
    This model serves as a baseline for comparison with HFA and SimpleMind
    architectures, demonstrating the O(N²) complexity of traditional attention.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract transformer configuration
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.num_layers = config['num_layers']
        self.num_heads = config.get('num_heads', 8)
        self.max_seq_len = config.get('max_seq_len', 2048)
        self.d_ff = config.get('d_ff', 4 * self.d_model)
        self.dropout = config.get('dropout', 0.1)
        self.pad_token_id = config.get('pad_token_id', 0)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id)
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Output head
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Tie weights between input and output embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        self.reset_parameters()
        
        bt.logging.info(f"Initialized Standard Transformer model with {self.count_parameters():,} parameters")
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through the standard transformer model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling loss [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            ModelOutput containing logits, loss, and complexity information
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings
        x = token_emb + pos_emb
        x = self.dropout_layer(x)
        
        # Create padding mask for transformer (True for padding tokens)
        src_key_padding_mask = ~attention_mask.bool()
        
        # Apply transformer layers
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return ModelOutput(
            logits=logits,
            loss=loss,
            complexity_info=self.get_complexity_info(seq_len),
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
        Generate text using the standard transformer model.
        
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
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(generated)
                logits = outputs.logits
                
                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering if specified
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                if generated.size(1) >= max_length:
                    break
        
        return generated
    
    def get_complexity_info(self, seq_len: int) -> Dict[str, str]:
        """
        Get standard transformer complexity information.
        
        Args:
            seq_len: Sequence length to analyze
            
        Returns:
            Dictionary containing complexity information
        """
        return {
            'architecture': 'Standard Transformer',
            'total_complexity': f"O({self.num_layers} × {seq_len}²)",
            'per_layer_complexity': f"O({seq_len}²)",
            'attention_complexity': f"O({seq_len}²)",
            'scaling_limitation': f"Quadratic scaling with sequence length",
            'memory_usage': f"O({seq_len}²) for attention matrices",
            'num_heads': str(self.num_heads),
            'attention_type': 'Multi-Head Self-Attention'
        }
    
    def get_performance_metrics(self, seq_len: int) -> Dict[str, float]:
        """
        Get standard transformer performance metrics.
        
        Args:
            seq_len: Sequence length to analyze
            
        Returns:
            Dictionary containing performance metrics
        """
        # Standard transformer has quadratic complexity
        attention_ops = seq_len ** 2 * self.num_layers
        
        return {
            'theoretical_speedup': 1.0,  # Baseline for comparison
            'memory_efficiency': 1.0 / seq_len,  # Quadratic memory usage
            'parameter_efficiency': float(self.count_parameters() / (self.vocab_size * self.d_model)),
            'context_scaling': 1.0 / seq_len,  # Quadratic scaling penalty
            'attention_operations': float(attention_ops),
            'quadratic_penalty': float(seq_len),  # Penalty increases with sequence length
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get standard transformer-specific model information."""
        base_info = super().get_model_info()
        base_info.update({
            'num_heads': self.num_heads,
            'complexity_class': 'O(N²)',
            'architecture_family': 'Transformer',
            'attention_type': 'Multi-Head Self-Attention',
            'scaling_behavior': 'Quadratic'
        })
        return base_info