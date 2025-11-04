"""
Optimized MindBlock - Fast and Efficient O(N) Attention Replacement

This version focuses on speed and effectiveness while maintaining O(N) complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class FastRouter(nn.Module):
    """Improved router that actually learns useful patterns."""
    
    def __init__(self, d_model: int, num_channels: int):
        super().__init__()
        self.d_model = d_model
        self.num_channels = num_channels
        
        # Better routing network with more capacity
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_channels)
        )
        
        # Better initialization
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Improved routing with better learning capacity."""
        # Compute routing logits with more capacity
        logits = self.router(x)  # [batch, seq, channels]
        
        # Use temperature scaling for better gradients
        temperature = 0.5  # Lower temperature for sharper routing
        routing_weights = F.softmax(logits / temperature, dim=-1)
        
        # Use all channels but with proper weighting
        # This ensures all routing parameters get gradients
        return None, routing_weights  # Return None for indices, full weights


class FastAggregator(nn.Module):
    """Improved aggregation that learns meaningful representations."""
    
    def __init__(self, d_model: int, num_channels: int):
        super().__init__()
        self.d_model = d_model
        self.num_channels = num_channels
        
        # Learnable channel transformations
        self.channel_transforms = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_channels)
        ])
        
        # Output mixing layer
        self.output_mix = nn.Linear(d_model, d_model)
        
        # Initialize properly
        for transform in self.channel_transforms:
            nn.init.xavier_uniform_(transform.weight)
        nn.init.xavier_uniform_(self.output_mix.weight)
        nn.init.zeros_(self.output_mix.bias)
        
    def forward(self, values: torch.Tensor, channel_ids: torch.Tensor, 
                routing_weights: torch.Tensor) -> torch.Tensor:
        """Improved aggregation using learnable channel transformations."""
        batch_size, seq_len, d_model = values.shape
        
        # Apply channel-specific transformations
        transformed_values = torch.zeros_like(values)
        
        for i, transform in enumerate(self.channel_transforms):
            # Get routing weight for this channel
            channel_weight = routing_weights[:, :, i:i+1]  # [batch, seq, 1]
            
            # Apply transformation and weight
            transformed = transform(values)  # [batch, seq, d_model]
            weighted_transformed = transformed * channel_weight  # [batch, seq, d_model]
            
            transformed_values = transformed_values + weighted_transformed
        
        # Final mixing
        output = self.output_mix(transformed_values)
        
        # Residual connection
        return values + output


class OptimizedMindBlock(nn.Module):
    """
    Optimized MindBlock for speed and effectiveness.
    
    Key optimizations:
    1. Simplified routing with top-k selection
    2. Minimal parameter aggregation
    3. Better initialization
    4. Reduced computational overhead
    """
    
    def __init__(
        self,
        d_model: int,
        num_channels: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        # Use fewer channels for better learning
        self.num_channels = num_channels or max(2, d_model // 64)
        
        # Standard projections (like attention)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Fast routing and aggregation
        self.router = FastRouter(d_model, self.num_channels)
        self.aggregator = FastAggregator(d_model, self.num_channels)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Better initialization for faster convergence."""
        # Xavier initialization for projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Optimized forward pass."""
        batch_size, seq_len, d_model = x.shape
        
        # Residual connection
        residual = x
        
        # Layer norm first (pre-norm)
        x = self.layer_norm(x)
        
        # Project to Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            queries = queries * mask
            keys = keys * mask
            values = values * mask
        
        # Enhanced routing using queries and keys
        routing_input = queries + 0.1 * keys  # Combine for better routing decisions
        channel_ids, routing_weights = self.router(routing_input)
        
        # Improved aggregation
        aggregated = self.aggregator(values, channel_ids, routing_weights)
        
        # Output projection
        output = self.out_proj(aggregated)
        output = self.dropout(output)
        
        # Residual connection
        return output + residual
    
    def get_complexity(self, seq_len: int) -> str:
        return f"O({seq_len})"


class OptimizedMindTransformer(nn.Module):
    """Optimized MindTransformer for better performance."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_channels: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Optimized MindBlocks
        self.layers = nn.ModuleList([
            OptimizedMindBlock(d_model, num_channels, dropout)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks (smaller for speed)
        d_ff = d_model * 2  # Reduced from 4x to 2x
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU(),  # GELU instead of ReLU
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Better initialization."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fast forward pass."""
        batch_size, seq_len = input_ids.shape
        
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Optimized layers
        for mind_block, ff in zip(self.layers, self.feed_forwards):
            # MindBlock
            x = mind_block(x, attention_mask)
            
            # Feed-forward with residual
            x = x + ff(x)
        
        # Output
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits
