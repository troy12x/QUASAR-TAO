"""
MindBlock Architecture - O(N) Attention Killer

A revolutionary post-transformer architecture that achieves O(N) complexity
while maintaining or exceeding the expressivity of multi-head attention.
"""

from .core.mind_block import MindBlock
from .routing.router import DynamicRouter, SoftRouter, HardRouter
from .memory.aggregation import ChannelAggregator
from .model import MindTransformer

__version__ = "0.1.0"
__all__ = [
    "MindBlock",
    "DynamicRouter", 
    "SoftRouter",
    "HardRouter",
    "ChannelAggregator",
    "MindTransformer"
]
