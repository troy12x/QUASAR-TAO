"""
SimpleMind with EXPLICIT Key-Value Memory
Uses a separate key-value memory store that can be directly written to and read from.
Like a neural Turing machine but simpler.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class ExplicitMemoryBlock(nn.Module):
    """
    Block with explicit key-value memory that persists across forward passes.
    
    Key idea: Model learns to:
    1. Extract key-value pairs from input
    2. Write them to external memory
    3. Query memory to retrieve values
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        memory_size: int = 1000,  # Large memory for many facts
        key_size: int = 128,
        value_size: int = 128
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.memory_size = memory_size
        self.key_size = key_size
        self.value_size = value_size
        
        # Standard Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Memory key and value extraction
        self.memory_key_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, key_size)
        )
        
        self.memory_value_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, value_size)
        )
        
        # Memory query
        self.memory_query_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, key_size)
        )
        
        # Memory read projection
        self.memory_read_proj = nn.Linear(value_size, d_model)
        
        # Write gate - decides what to write
        self.write_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Standard components
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=num_heads)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        memory_dict: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with explicit memory.
        
        memory_dict: [batch, memory_size, key_size + value_size]
        """
        batch_size, seq_len, d_model = x.shape
        
        residual = x
        x = self.layer_norm(x)
        
        # Initialize memory if needed
        if memory_dict is None:
            memory_dict = torch.zeros(
                batch_size, self.memory_size, self.key_size + self.value_size,
                device=x.device, dtype=x.dtype
            )
        
        # === MEMORY WRITE ===
        # Extract keys and values to write
        mem_keys = self.memory_key_net(x)  # [batch, seq_len, key_size]
        mem_values = self.memory_value_net(x)  # [batch, seq_len, value_size]
        write_gates = self.write_gate(x)  # [batch, seq_len, 1]
        
        # Write to memory (top-k important positions)
        write_importance = write_gates.squeeze(-1)  # [batch, seq_len]
        num_writes = min(seq_len, self.memory_size // 2)  # Don't fill entire memory
        
        top_write_vals, top_write_idx = torch.topk(write_importance, k=num_writes, dim=1)
        
        # Gather keys and values to write
        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, num_writes)
        selected_keys = mem_keys[batch_idx, top_write_idx]  # [batch, num_writes, key_size]
        selected_values = mem_values[batch_idx, top_write_idx]  # [batch, num_writes, value_size]
        
        # Concatenate key+value
        new_entries = torch.cat([selected_keys, selected_values], dim=-1)  # [batch, num_writes, key_size+value_size]
        
        # Update memory (keep recent + add new)
        keep_size = self.memory_size - num_writes
        updated_memory = torch.cat([
            memory_dict[:, :keep_size, :],
            new_entries
        ], dim=1)
        
        # === MEMORY READ ===
        # Query memory
        mem_queries = self.memory_query_net(x)  # [batch, seq_len, key_size]
        
        # Split memory into keys and values
        stored_keys = updated_memory[:, :, :self.key_size]  # [batch, memory_size, key_size]
        stored_values = updated_memory[:, :, self.key_size:]  # [batch, memory_size, value_size]
        
        # Compute similarity: [batch, seq_len, key_size] @ [batch, key_size, memory_size]
        similarity = torch.bmm(mem_queries, stored_keys.transpose(1, 2)) / math.sqrt(self.key_size)
        # [batch, seq_len, memory_size]
        
        # Softmax over memory
        attention_weights = F.softmax(similarity, dim=-1)
        
        # Retrieve values: [batch, seq_len, memory_size] @ [batch, memory_size, value_size]
        retrieved = torch.bmm(attention_weights, stored_values)  # [batch, seq_len, value_size]
        
        # Project retrieved values
        memory_output = self.memory_read_proj(retrieved)  # [batch, seq_len, d_model]
        
        # === STANDARD PROCESSING ===
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Local
        values_t = values.transpose(1, 2)
        local_features = self.local_conv(values_t).transpose(1, 2)
        
        # Global
        queries_t = queries.transpose(1, 2)
        global_context = self.global_pool(queries_t).squeeze(-1)
        global_context = self.global_proj(global_context).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Position gating
        position_weights = torch.sigmoid(keys)
        
        # Combine: local + global + STRONG memory influence
        combined = local_features + 0.1 * global_context + 0.5 * memory_output
        combined = combined * position_weights
        
        # Output
        output = self.out_proj(combined)
        output = self.dropout(output)
        
        return output + residual, updated_memory


class SimpleMindWithExplicitMemory(nn.Module):
    """SimpleMind with explicit external memory."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        memory_size: int = 1000,
        key_size: int = 128,
        value_size: int = 128,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.memory_size = memory_size
        self.key_size = key_size
        self.value_size = value_size
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Explicit memory blocks
        self.layers = nn.ModuleList([
            ExplicitMemoryBlock(d_model, num_heads, dropout, memory_size, key_size, value_size)
            for _ in range(num_layers)
        ])
        
        # Feed-forwards
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
        self.lm_head.weight = self.token_embedding.weight
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        memory_dicts: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """Forward with persistent memory."""
        batch_size, seq_len = input_ids.shape
        
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        if memory_dicts is None:
            memory_dicts = [None] * self.num_layers
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Process through layers
        new_memory_dicts = []
        
        for i, (memory_block, ff) in enumerate(zip(self.layers, self.feed_forwards)):
            x, mem_dict = memory_block(x, memory_dicts[i], attention_mask)
            new_memory_dicts.append(mem_dict)
            x = x + ff(x)
        
        # Output
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits, new_memory_dicts
