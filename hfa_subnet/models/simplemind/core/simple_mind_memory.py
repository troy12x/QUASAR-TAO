"""
SimpleMind with Enhanced Memory
Adds explicit memory mechanisms to SimpleMindBlock for better fact retention.

Key improvements:
1. External memory bank for storing key-value associations
2. Memory-augmented attention mechanism
3. Write and read gates for selective memory storage
4. Progressive memory consolidation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MemoryAugmentedMindBlock(nn.Module):
    """
    SimpleMindBlock with explicit memory augmentation.
    
    Adds:
    - Memory bank: Stores key-value pairs explicitly
    - Write gate: Decides what to memorize
    - Read gate: Retrieves relevant memories
    - O(N) complexity maintained through fixed-size memory
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        memory_slots: int = 128,  # Fixed number of memory slots
        memory_key_size: int = 64
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.memory_slots = memory_slots
        self.memory_key_size = memory_key_size
        
        # Standard Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Memory projections
        self.memory_key_proj = nn.Linear(d_model, memory_key_size)
        self.memory_value_proj = nn.Linear(d_model, d_model)
        self.memory_query_proj = nn.Linear(d_model, memory_key_size)
        
        # Memory write and read gates
        self.write_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self.read_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Local context (O(N))
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=num_heads)
        
        # Global context (O(N))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(d_model, d_model)
        
        # Memory attention
        self.memory_attn_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.memory_key_proj.weight)
        nn.init.xavier_uniform_(self.memory_value_proj.weight)
        nn.init.xavier_uniform_(self.memory_query_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.global_proj.weight)
        nn.init.xavier_uniform_(self.memory_attn_proj.weight)
        
    def forward(
        self, 
        x: torch.Tensor, 
        memory_keys: Optional[torch.Tensor] = None,
        memory_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with memory augmentation.
        
        Returns:
            output: [batch, seq_len, d_model]
            updated_memory_keys: [batch, memory_slots, memory_key_size]
            updated_memory_values: [batch, memory_slots, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        residual = x
        x = self.layer_norm(x)
        
        # Initialize memory if not provided
        if memory_keys is None:
            memory_keys = torch.zeros(
                batch_size, self.memory_slots, self.memory_key_size,
                device=x.device, dtype=x.dtype
            )
        if memory_values is None:
            memory_values = torch.zeros(
                batch_size, self.memory_slots, d_model,
                device=x.device, dtype=x.dtype
            )
        
        # === MEMORY WRITE ===
        # Decide what to write to memory
        write_gates = self.write_gate(x)  # [batch, seq_len, 1]
        
        # Create memory keys and values from current input
        mem_keys = self.memory_key_proj(x)  # [batch, seq_len, memory_key_size]
        mem_values = self.memory_value_proj(x)  # [batch, seq_len, d_model]
        
        # Weighted average of new memories (using write gate)
        weighted_keys = mem_keys * write_gates  # [batch, seq_len, memory_key_size]
        weighted_values = mem_values * write_gates  # [batch, seq_len, d_model]
        
        # Aggregate new memories to write (average pooling)
        new_mem_key = torch.mean(weighted_keys, dim=1, keepdim=True)  # [batch, 1, memory_key_size]
        new_mem_value = torch.mean(weighted_values, dim=1, keepdim=True)  # [batch, 1, d_model]
        
        # Update memory bank (shift and add new)
        # Simple strategy: circular buffer (shift left, add new at end)
        updated_memory_keys = torch.cat([
            memory_keys[:, 1:, :],  # Remove oldest
            new_mem_key
        ], dim=1)
        
        updated_memory_values = torch.cat([
            memory_values[:, 1:, :],
            new_mem_value
        ], dim=1)
        
        # === MEMORY READ ===
        # Query memory based on current input
        read_gates = self.read_gate(x)  # [batch, seq_len, 1]
        mem_queries = self.memory_query_proj(x)  # [batch, seq_len, memory_key_size]
        
        # Compute similarity between queries and memory keys
        # [batch, seq_len, memory_key_size] @ [batch, memory_key_size, memory_slots]
        # = [batch, seq_len, memory_slots]
        memory_scores = torch.bmm(
            mem_queries, 
            updated_memory_keys.transpose(1, 2)
        ) / math.sqrt(self.memory_key_size)
        
        # Softmax to get attention weights
        memory_weights = F.softmax(memory_scores, dim=-1)  # [batch, seq_len, memory_slots]
        
        # Retrieve from memory
        # [batch, seq_len, memory_slots] @ [batch, memory_slots, d_model]
        # = [batch, seq_len, d_model]
        memory_retrieved = torch.bmm(memory_weights, updated_memory_values)
        
        # Apply read gate
        memory_retrieved = memory_retrieved * read_gates
        
        # === STANDARD PROCESSING ===
        # Q, K, V projections
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Local context
        values_t = values.transpose(1, 2)
        local_features = self.local_conv(values_t).transpose(1, 2)
        
        # Global context
        queries_t = queries.transpose(1, 2)
        global_context = self.global_pool(queries_t).squeeze(-1)
        global_context = self.global_proj(global_context).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Position gating
        position_weights = torch.sigmoid(keys)
        
        # Combine: local + global + memory
        combined = local_features + 0.1 * global_context + 0.3 * memory_retrieved
        combined = combined * position_weights
        
        # Output projection
        output = self.out_proj(combined)
        output = self.dropout(output)
        
        # Residual
        output = output + residual
        
        return output, updated_memory_keys, updated_memory_values


class SimpleMindTransformerWithMemory(nn.Module):
    """SimpleMindTransformer with memory augmentation."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        memory_slots: int = 128,
        memory_key_size: int = 64,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.memory_slots = memory_slots
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Memory-augmented MindBlocks
        self.layers = nn.ModuleList([
            MemoryAugmentedMindBlock(d_model, num_heads, dropout, memory_slots, memory_key_size)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
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
        """Initialize parameters."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        memory_keys_list: Optional[list] = None,
        memory_values_list: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list, list]:
        """
        Forward pass with memory.
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            updated_memory_keys_list: List of memory keys for each layer
            updated_memory_values_list: List of memory values for each layer
        """
        batch_size, seq_len = input_ids.shape
        
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Initialize memory if not provided
        if memory_keys_list is None:
            memory_keys_list = [None] * self.num_layers
        if memory_values_list is None:
            memory_values_list = [None] * self.num_layers
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Process through layers with memory
        new_memory_keys_list = []
        new_memory_values_list = []
        
        for i, (mind_block, ff) in enumerate(zip(self.layers, self.feed_forwards)):
            # Memory-augmented MindBlock
            x, mem_keys, mem_values = mind_block(
                x, 
                memory_keys_list[i], 
                memory_values_list[i],
                attention_mask
            )
            new_memory_keys_list.append(mem_keys)
            new_memory_values_list.append(mem_values)
            
            # Feed-forward
            x = x + ff(x)
        
        # Output
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits, new_memory_keys_list, new_memory_values_list
