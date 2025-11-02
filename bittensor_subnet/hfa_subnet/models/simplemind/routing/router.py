"""
Concrete Router Implementations for O(N) Token-to-Channel Mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

from .base import BaseRouter


class HardRouter(BaseRouter):
    """
    Hard routing: Each token is assigned to exactly one channel.
    Complexity: O(N)
    """
    
    def __init__(self, d_model: int, num_channels: int, temperature: float = 1.0):
        super().__init__(d_model, num_channels)
        self.temperature = temperature
        self.routing_layer = nn.Linear(d_model, num_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.routing_layer.weight)
        nn.init.zeros_(self.routing_layer.bias)
        
    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            queries: [batch_size, seq_len, d_model]
            
        Returns:
            channel_ids: [batch_size, seq_len] - hard channel assignments
            routing_weights: None (hard routing doesn't use weights)
        """
        batch_size, seq_len, d_model = queries.shape
        
        # Compute routing logits: [batch_size, seq_len, num_channels]
        routing_logits = self.routing_layer(queries) / self.temperature
        
        # Hard assignment with straight-through estimator for gradients
        if self.training:
            # During training, use Gumbel-Softmax for differentiability
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(routing_logits) + 1e-8) + 1e-8)
            gumbel_logits = (routing_logits + gumbel_noise) / self.temperature
            soft_assignment = F.softmax(gumbel_logits, dim=-1)
            
            # Straight-through: hard assignment in forward, soft in backward
            hard_assignment = F.one_hot(torch.argmax(soft_assignment, dim=-1), self.num_channels).float()
            channel_assignment = hard_assignment - soft_assignment.detach() + soft_assignment
            
            # Get channel IDs for aggregation
            channel_ids = torch.argmax(channel_assignment, dim=-1)
        else:
            # During inference, use standard argmax
            channel_ids = torch.argmax(routing_logits, dim=-1)
        
        return channel_ids, None
    
    def get_complexity(self, seq_len: int) -> str:
        return f"O({seq_len})"


class SoftRouter(BaseRouter):
    """
    Soft routing: Each token can belong to multiple channels with weights.
    Complexity: O(N)
    """
    
    def __init__(self, d_model: int, num_channels: int, temperature: float = 1.0, 
                 top_k: Optional[int] = None):
        super().__init__(d_model, num_channels)
        self.temperature = temperature
        self.top_k = top_k or num_channels  # Use all channels by default
        self.routing_layer = nn.Linear(d_model, num_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.routing_layer.weight)
        nn.init.zeros_(self.routing_layer.bias)
        
    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: [batch_size, seq_len, d_model]
            
        Returns:
            channel_ids: [batch_size, seq_len, top_k] - top-k channel indices
            routing_weights: [batch_size, seq_len, top_k] - corresponding weights
        """
        batch_size, seq_len, d_model = queries.shape
        
        # Compute routing logits: [batch_size, seq_len, num_channels]
        routing_logits = self.routing_layer(queries) / self.temperature
        
        # Get top-k channels and their weights
        top_k = min(self.top_k, self.num_channels)
        routing_weights, channel_ids = torch.topk(routing_logits, top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return channel_ids, routing_weights
    
    def get_complexity(self, seq_len: int) -> str:
        return f"O({seq_len} * log({self.num_channels}))" if self.top_k < self.num_channels else f"O({seq_len})"


class DynamicRouter(BaseRouter):
    """
    Dynamic routing with learnable channel scaling.
    Adapts number of active channels based on sequence complexity.
    Complexity: O(N)
    """
    
    def __init__(self, d_model: int, max_channels: int, min_channels: int = 1, 
                 temperature: float = 1.0, gating_threshold: float = 0.1):
        super().__init__(d_model, max_channels)
        self.max_channels = max_channels
        self.min_channels = min_channels
        self.temperature = temperature
        self.gating_threshold = gating_threshold
        
        # Routing network
        self.routing_layer = nn.Linear(d_model, max_channels)
        
        # Channel gating network (decides how many channels to use)
        self.gate_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.routing_layer.weight)
        nn.init.zeros_(self.routing_layer.bias)
        
        for layer in self.gate_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: [batch_size, seq_len, d_model]
            
        Returns:
            channel_ids: [batch_size, seq_len, active_channels] - active channel indices
            routing_weights: [batch_size, seq_len, active_channels] - corresponding weights
        """
        batch_size, seq_len, d_model = queries.shape
        
        # Compute routing logits: [batch_size, seq_len, max_channels]
        routing_logits = self.routing_layer(queries) / self.temperature
        
        # Compute gating scores to determine number of active channels per token
        gate_scores = self.gate_layer(queries).squeeze(-1)  # [batch_size, seq_len]
        
        # Dynamic channel selection
        num_active = torch.clamp(
            (gate_scores * self.max_channels).round().long(),
            min=self.min_channels,
            max=self.max_channels
        )
        
        # For simplicity, use top-k based on the maximum active channels in batch
        max_active = num_active.max().item()
        routing_weights, channel_ids = torch.topk(routing_logits, max_active, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Apply gating threshold
        mask = routing_weights > self.gating_threshold
        routing_weights = routing_weights * mask.float()
        
        # Renormalize
        routing_weights = F.normalize(routing_weights, p=1, dim=-1)
        
        return channel_ids, routing_weights
    
    def get_complexity(self, seq_len: int) -> str:
        return f"O({seq_len} * log({self.max_channels}))"


class LearnableHashRouter(BaseRouter):
    """
    Learnable hash-based routing for ultra-fast O(1) per-token routing.
    Uses learnable hash functions to map tokens to channels.
    """
    
    def __init__(self, d_model: int, num_channels: int, num_hash_functions: int = 4):
        super().__init__(d_model, num_channels)
        self.num_hash_functions = num_hash_functions
        
        # Multiple hash functions for better distribution
        self.hash_layers = nn.ModuleList([
            nn.Linear(d_model, 1, bias=False) for _ in range(num_hash_functions)
        ])
        
        # Learnable channel embeddings
        self.channel_embeddings = nn.Parameter(torch.randn(num_channels, d_model))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.hash_layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.channel_embeddings)
    
    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: [batch_size, seq_len, d_model]
            
        Returns:
            channel_ids: [batch_size, seq_len, num_hash_functions] - hash-based channel assignments
            routing_weights: [batch_size, seq_len, num_hash_functions] - uniform weights
        """
        batch_size, seq_len, d_model = queries.shape
        
        channel_ids = []
        for hash_layer in self.hash_layers:
            # Compute hash values
            hash_values = hash_layer(queries).squeeze(-1)  # [batch_size, seq_len]
            
            # Map to channel indices using modulo
            channel_id = (hash_values.abs() * self.num_channels).long() % self.num_channels
            channel_ids.append(channel_id)
        
        # Stack channel assignments: [batch_size, seq_len, num_hash_functions]
        channel_ids = torch.stack(channel_ids, dim=-1)
        
        # Uniform weights for all hash functions
        routing_weights = torch.ones_like(channel_ids, dtype=queries.dtype) / self.num_hash_functions
        
        return channel_ids, routing_weights
    
    def get_complexity(self, seq_len: int) -> str:
        return f"O({seq_len})"
