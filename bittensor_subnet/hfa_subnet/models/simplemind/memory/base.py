"""
Base Aggregator Interface for MindBlock Architecture
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseAggregator(nn.Module, ABC):
    """
    Abstract base class for all channel aggregation mechanisms.
    
    The aggregator is responsible for combining tokens within the same channel
    to create context-aware representations in O(N) time.
    """
    
    def __init__(self, d_model: int, num_channels: int):
        super().__init__()
        self.d_model = d_model
        self.num_channels = num_channels
        
    @abstractmethod
    def forward(self, values: torch.Tensor, channel_ids: torch.Tensor, 
                routing_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregate values within channels.
        
        Args:
            values: Token values [batch_size, seq_len, d_model]
            channel_ids: Channel assignments [batch_size, seq_len] or [batch_size, seq_len, k]
            routing_weights: Optional routing weights [batch_size, seq_len, k]
            
        Returns:
            aggregated: Channel-aggregated representations [batch_size, seq_len, d_model]
        """
        pass
    
    @abstractmethod
    def get_complexity(self, seq_len: int) -> str:
        """Return the computational complexity of this aggregator."""
        pass
