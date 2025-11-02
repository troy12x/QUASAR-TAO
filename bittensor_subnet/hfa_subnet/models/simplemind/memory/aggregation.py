"""
Channel Aggregation Implementations for O(N) Information Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .base import BaseAggregator


class MeanAggregator(BaseAggregator):
    """
    Simple mean aggregation within channels.
    Complexity: O(N)
    """
    
    def __init__(self, d_model: int, num_channels: int):
        super().__init__(d_model, num_channels)
        
    def forward(self, values: torch.Tensor, channel_ids: torch.Tensor, 
                routing_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            values: [batch_size, seq_len, d_model]
            channel_ids: [batch_size, seq_len] for hard routing or [batch_size, seq_len, k] for soft
            routing_weights: [batch_size, seq_len, k] for soft routing
            
        Returns:
            aggregated: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = values.shape
        
        if routing_weights is None:
            # Hard routing case
            channel_ids = channel_ids.long()
            aggregated = torch.zeros_like(values)
            
            for b in range(batch_size):
                for c in range(self.num_channels):
                    mask = (channel_ids[b] == c)
                    if mask.any():
                        channel_values = values[b][mask]
                        aggregated_value = channel_values.mean(dim=0)
                        aggregated[b][mask] = aggregated_value
                        
        else:
            # Soft routing case
            batch_size, seq_len, k = channel_ids.shape
            aggregated = torch.zeros_like(values)
            
            for b in range(batch_size):
                for c in range(self.num_channels):
                    # Find all tokens that route to this channel
                    channel_mask = (channel_ids[b] == c)  # [seq_len, k]
                    
                    if channel_mask.any():
                        # Get weights for this channel
                        weights = routing_weights[b] * channel_mask.float()  # [seq_len, k]
                        total_weight = weights.sum(dim=-1, keepdim=True)  # [seq_len, 1]
                        
                        # Avoid division by zero
                        total_weight = torch.clamp(total_weight, min=1e-8)
                        weights = weights.sum(dim=-1, keepdim=True) / total_weight  # [seq_len, 1]
                        
                        # Weighted aggregation
                        weighted_values = values[b] * weights  # [seq_len, d_model]
                        channel_sum = weighted_values.sum(dim=0, keepdim=True)  # [1, d_model]
                        
                        # Broadcast to all tokens in this channel
                        token_mask = channel_mask.any(dim=-1)  # [seq_len]
                        aggregated[b][token_mask] += channel_sum
            
        return aggregated
    
    def get_complexity(self, seq_len: int) -> str:
        return f"O({seq_len})"


class SumAggregator(BaseAggregator):
    """
    Sum aggregation within channels.
    Complexity: O(N)
    """
    
    def __init__(self, d_model: int, num_channels: int):
        super().__init__(d_model, num_channels)
        
    def forward(self, values: torch.Tensor, channel_ids: torch.Tensor, 
                routing_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = values.shape
        
        if routing_weights is None:
            # Hard routing case
            channel_ids = channel_ids.long()
            aggregated = torch.zeros_like(values)
            
            for b in range(batch_size):
                for c in range(self.num_channels):
                    mask = (channel_ids[b] == c)
                    if mask.any():
                        channel_values = values[b][mask]
                        aggregated_value = channel_values.sum(dim=0)
                        aggregated[b][mask] = aggregated_value
                        
        else:
            # Soft routing case - similar to mean but without normalization
            batch_size, seq_len, k = channel_ids.shape
            aggregated = torch.zeros_like(values)
            
            for b in range(batch_size):
                for c in range(self.num_channels):
                    channel_mask = (channel_ids[b] == c)
                    
                    if channel_mask.any():
                        weights = routing_weights[b] * channel_mask.float()
                        weighted_values = values[b] * weights.sum(dim=-1, keepdim=True)
                        channel_sum = weighted_values.sum(dim=0, keepdim=True)
                        
                        token_mask = channel_mask.any(dim=-1)
                        aggregated[b][token_mask] += channel_sum
            
        return aggregated
    
    def get_complexity(self, seq_len: int) -> str:
        return f"O({seq_len})"


class LearnableAggregator(BaseAggregator):
    """
    Learnable aggregation with attention-like mechanisms within channels.
    Complexity: O(N) average case, O(N²/C) worst case where C is number of channels
    """
    
    def __init__(self, d_model: int, num_channels: int, num_heads: int = 8):
        super().__init__(d_model, num_channels)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Learnable aggregation parameters
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Channel-specific parameters
        self.channel_embeddings = nn.Parameter(torch.randn(num_channels, d_model))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.channel_embeddings)
        
    def forward(self, values: torch.Tensor, channel_ids: torch.Tensor, 
                routing_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = values.shape
        
        # Project to Q, K, V
        queries = self.query_proj(values)  # [batch_size, seq_len, d_model]
        keys = self.key_proj(values)
        vals = self.value_proj(values)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        vals = vals.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        aggregated = torch.zeros_like(vals)
        
        if routing_weights is None:
            # Hard routing case
            channel_ids = channel_ids.long()
            
            for b in range(batch_size):
                for c in range(self.num_channels):
                    mask = (channel_ids[b] == c)
                    if mask.any():
                        # Get tokens in this channel
                        channel_queries = queries[b][mask]  # [num_tokens_in_channel, num_heads, head_dim]
                        channel_keys = keys[b][mask]
                        channel_vals = vals[b][mask]
                        
                        # Add channel embedding
                        channel_emb = self.channel_embeddings[c].view(1, self.num_heads, self.head_dim)
                        channel_queries = channel_queries + channel_emb
                        
                        # Compute attention within channel
                        if channel_queries.size(0) > 1:
                            # Multi-token channel: use attention
                            attn_scores = torch.einsum('nhd,mhd->nhm', channel_queries, channel_keys)
                            attn_scores = attn_scores / math.sqrt(self.head_dim)
                            attn_weights = F.softmax(attn_scores, dim=-1)
                            
                            # Apply attention
                            channel_output = torch.einsum('nhm,mhd->nhd', attn_weights, channel_vals)
                        else:
                            # Single token channel: pass through
                            channel_output = channel_vals
                        
                        aggregated[b][mask] = channel_output
        else:
            # Soft routing case - simplified for efficiency
            for b in range(batch_size):
                for i in range(seq_len):
                    # For each token, aggregate from its assigned channels
                    token_channels = channel_ids[b, i]  # [k]
                    token_weights = routing_weights[b, i]  # [k]
                    
                    aggregated_output = torch.zeros(self.num_heads, self.head_dim, 
                                                  device=values.device, dtype=values.dtype)
                    
                    for j, (ch, weight) in enumerate(zip(token_channels, token_weights)):
                        if weight > 1e-6:  # Skip negligible weights
                            ch = ch.item()
                            channel_emb = self.channel_embeddings[ch].view(self.num_heads, self.head_dim)
                            enhanced_query = queries[b, i] + channel_emb
                            
                            # Simple weighted contribution
                            aggregated_output += weight * (enhanced_query + vals[b, i])
                    
                    aggregated[b, i] = aggregated_output
        
        # Reshape and project output
        aggregated = aggregated.view(batch_size, seq_len, d_model)
        output = self.out_proj(aggregated)
        
        return output
    
    def get_complexity(self, seq_len: int) -> str:
        return f"O({seq_len}) average, O({seq_len}²/{self.num_channels}) worst case"


class ChannelAggregator(BaseAggregator):
    """
    Unified channel aggregator that can switch between different aggregation strategies.
    """
    
    def __init__(self, d_model: int, num_channels: int, aggregation_type: str = "mean", **kwargs):
        super().__init__(d_model, num_channels)
        
        if aggregation_type == "mean":
            self.aggregator = MeanAggregator(d_model, num_channels)
        elif aggregation_type == "sum":
            self.aggregator = SumAggregator(d_model, num_channels)
        elif aggregation_type == "learnable":
            self.aggregator = LearnableAggregator(d_model, num_channels, **kwargs)
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
            
        self.aggregation_type = aggregation_type
        
    def forward(self, values: torch.Tensor, channel_ids: torch.Tensor, 
                routing_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.aggregator(values, channel_ids, routing_weights)
    
    def get_complexity(self, seq_len: int) -> str:
        return self.aggregator.get_complexity(seq_len)
