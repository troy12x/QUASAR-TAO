"""
Base Router Interface for MindBlock Architecture
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseRouter(nn.Module, ABC):
    """
    Abstract base class for all routing mechanisms.
    
    The router is responsible for mapping tokens to channels in O(N) time,
    replacing the O(N²) attention mechanism.
    """
    
    def __init__(self, d_model: int, num_channels: int):
        super().__init__()
        self.d_model = d_model
        self.num_channels = num_channels
        
    @abstractmethod
    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Route tokens to channels.
        
        Args:
            queries: Token representations [batch_size, seq_len, d_model]
            
        Returns:
            channel_ids: Channel assignments [batch_size, seq_len] or [batch_size, seq_len, num_channels]
            routing_weights: Optional soft routing weights [batch_size, seq_len, num_channels]
        """
        pass
    
    @abstractmethod
    def get_complexity(self, seq_len: int) -> str:
        """Return the computational complexity of this router."""
        pass
