"""
Simple MindBlock - Focus on Learning Quality First

This version prioritizes learning effectiveness over complexity.
The goal is to match transformer learning while maintaining O(N) complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class SimpleMindBlock(nn.Module):
    """
    Simplified MindBlock that focuses on learning quality.
    
    Key principles:
    1. Minimal information loss
    2. Strong gradient flow
    3. Simple but effective O(N) operations
    4. Match transformer expressivity
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Standard Q, K, V projections (same as attention)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Simple O(N) mixing instead of O(N²) attention
        # Use local convolution + global pooling
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=num_heads)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters for good learning."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.global_proj.weight)
        
        # Initialize conv weights
        nn.init.xavier_uniform_(self.local_conv.weight)
        nn.init.zeros_(self.local_conv.bias)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.global_proj.bias)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Simple O(N) forward pass that preserves information.
        
        Strategy:
        1. Use Q, K, V like attention but avoid O(N²) operations
        2. Local convolution for local dependencies (O(N))
        3. Global pooling for global context (O(N))
        4. Combine local + global information
        """
        batch_size, seq_len, d_model = x.shape
        
        # Residual connection
        residual = x
        
        # Pre-norm
        x = self.layer_norm(x)
        
        # Project to Q, K, V (ensure all are used for gradients)
        queries = self.q_proj(x)  # [batch, seq, d_model]
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            queries = queries * mask
            keys = keys * mask
            values = values * mask
        
        # Method 1: Local dependencies via convolution O(N)
        # Transpose for conv1d: [batch, d_model, seq_len]
        values_t = values.transpose(1, 2)
        local_features = self.local_conv(values_t)  # [batch, d_model, seq_len]
        local_features = local_features.transpose(1, 2)  # [batch, seq_len, d_model]
        
        # Method 2: Global context via pooling O(N)
        # Use queries to compute global context
        queries_t = queries.transpose(1, 2)  # [batch, d_model, seq_len]
        global_context = self.global_pool(queries_t)  # [batch, d_model, 1]
        global_context = global_context.squeeze(-1)  # [batch, d_model]
        global_context = self.global_proj(global_context)  # [batch, d_model]
        
        # Broadcast global context to all positions
        global_features = global_context.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_model]
        
        # Method 3: Use keys for position-wise modulation
        position_weights = torch.sigmoid(keys)  # [batch, seq_len, d_model]
        
        # Combine all information sources
        # Local + Global + Position-aware
        combined = local_features + 0.1 * global_features
        combined = combined * position_weights  # Position-wise gating
        
        # Output projection
        output = self.out_proj(combined)
        output = self.dropout(output)
        
        # Residual connection
        return output + residual
    
    def get_complexity(self, seq_len: int) -> str:
        return f"O({seq_len})"


class SimpleMindTransformer(nn.Module):
    """Simple MindTransformer focused on learning quality."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
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
        
        # Simple MindBlocks
        self.layers = nn.ModuleList([
            SimpleMindBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks (standard size for fair comparison)
        d_ff = d_model * 4
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU(),
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
        """Initialize for good learning."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass focused on learning quality."""
        batch_size, seq_len = input_ids.shape
        
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Simple MindBlock layers
        for mind_block, ff in zip(self.layers, self.feed_forwards):
            # MindBlock (O(N) attention replacement)
            x = mind_block(x, attention_mask)
            
            # Feed-forward with residual
            x = x + ff(x)
        
        # Output
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits
