"""
Neural Plastic Memory Block (NPMB) for SimpleMindTransformer

Implements inference-time plasticity where model parameters adapt during inference
without backpropagation. Allows the model to learn and remember within the inference loop.

Architecture:
    - W_core: Stable parameters (frozen during inference)
    - W_mem: Plastic parameters (dynamically updated during inference)
    - Hebbian-like update rules for memory formation
    - Optional consolidation from W_mem → W_core

Mathematical Foundation:
    1. Read:  h_t = f_core(x_t, W_core) + f_mem(x_t, W_mem)
    2. Encode: z_t = E(h_t), k_t = K(h_t)
    3. Write: ΔW_mem = η * (z_t ⊗ k_t)
    4. Update: W_mem ← α * W_mem + (1 - α) * ΔW_mem
    5. Consolidate (optional): W_core ← W_core + λ * W_mem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class NeuralPlasticMemoryBlock(nn.Module):
    """
    Neural Plastic Memory Block - Inference-time learning module
    
    Divides computation into:
    - Core pathway: Stable, pretrained parameters
    - Memory pathway: Plastic, adaptive parameters
    
    Memory updates occur during inference via Hebbian-like rules.
    """
    
    def __init__(
        self,
        d_model: int,
        memory_rank: int = 64,
        plasticity_rate: float = 0.01,
        decay_factor: float = 0.99,
        consolidation_rate: float = 0.001,
        consolidation_steps: int = 100,
        enable_gating: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            memory_rank: Low-rank dimension for plastic memory (for efficiency)
            plasticity_rate: η - learning rate for memory updates
            decay_factor: α - decay/forgetting factor (0=full forget, 1=no decay)
            consolidation_rate: λ - rate of consolidating W_mem → W_core
            consolidation_steps: Consolidate every N steps
            enable_gating: Use gating to prevent memory from corrupting core
        """
        super().__init__()
        
        self.d_model = d_model
        self.memory_rank = memory_rank
        self.plasticity_rate = plasticity_rate
        self.decay_factor = decay_factor
        self.consolidation_rate = consolidation_rate
        self.consolidation_steps = consolidation_steps
        self.enable_gating = enable_gating
        
        # ══════════════════════════════════════════════════════════
        # CORE PATHWAY (Stable Parameters - W_core)
        # ══════════════════════════════════════════════════════════
        self.core_transform = nn.Linear(d_model, d_model, bias=False)
        
        # ══════════════════════════════════════════════════════════
        # MEMORY PATHWAY (Plastic Parameters - W_mem)
        # Low-rank factorization: W_mem = U @ V^T
        # ══════════════════════════════════════════════════════════
        self.mem_down = nn.Parameter(torch.zeros(d_model, memory_rank))
        self.mem_up = nn.Parameter(torch.zeros(memory_rank, d_model))
        
        # ══════════════════════════════════════════════════════════
        # MEMORY ENCODER & KEY GENERATOR
        # ══════════════════════════════════════════════════════════
        # E(h_t): Compress hidden activations into memory representation
        self.encoder = nn.Linear(d_model, memory_rank, bias=False)
        
        # K(h_t): Generate addressing key for memory update
        self.key_generator = nn.Linear(d_model, d_model, bias=False)
        
        # ══════════════════════════════════════════════════════════
        # GATING MECHANISM
        # ══════════════════════════════════════════════════════════
        if enable_gating:
            self.gate = nn.Linear(d_model, d_model, bias=True)
        
        # ══════════════════════════════════════════════════════════
        # CONSOLIDATION BUFFER
        # ══════════════════════════════════════════════════════════
        # Accumulated memory to consolidate into core
        self.register_buffer('consolidation_buffer', torch.zeros(d_model, d_model))
        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters"""
        # Core pathway: Standard initialization
        nn.init.xavier_uniform_(self.core_transform.weight)
        
        # Memory pathway: Start with zeros (no initial memory)
        nn.init.zeros_(self.mem_down)
        nn.init.zeros_(self.mem_up)
        
        # Encoders: Normal initialization (not too small!)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.key_generator.weight)
        
        # Gate: Initialize to pass core pathway initially
        if self.enable_gating:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 1.0)  # Start with gate open to core
    
    def compute_memory_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute output from plastic memory pathway
        W_mem @ x where W_mem = U @ V^T (low-rank)
        """
        # Low-rank multiplication: x @ V^T @ U^T
        mem_out = x @ self.mem_up.t()  # [batch, seq, rank]
        mem_out = mem_out @ self.mem_down.t()  # [batch, seq, d_model]
        return mem_out
    
    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = True,
        consolidate: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with optional memory update
        
        Args:
            x: Input tensor [batch, seq, d_model]
            update_memory: If True, update W_mem during inference
            consolidate: If True, consolidate W_mem → W_core
            
        Returns:
            output: Combined output from core + memory pathways
            info: Dictionary with memory statistics
        """
        batch_size, seq_len, d_model = x.shape
        
        # ══════════════════════════════════════════════════════════
        # 1️⃣ READ PHASE: f_core(x) + f_mem(x)
        # ══════════════════════════════════════════════════════════
        core_out = self.core_transform(x)  # Stable pathway
        mem_out = self.compute_memory_output(x)  # Plastic pathway
        
        # Combine with gating
        if self.enable_gating:
            gate_values = torch.sigmoid(self.gate(x))
            h_t = gate_values * core_out + (1 - gate_values) * mem_out
        else:
            h_t = core_out + mem_out
        
        # ══════════════════════════════════════════════════════════
        # 2️⃣ ENCODE PHASE: Generate memory representation
        # ══════════════════════════════════════════════════════════
        if update_memory:  # Update memory when requested (train or eval)
            with torch.no_grad():
                # Detach to prevent gradient issues
                h_t_detached = h_t.detach()
                
                # Encode hidden state
                z_t = self.encoder(h_t_detached)  # [batch, seq, memory_rank]
                
                # Generate addressing key
                k_t = self.key_generator(h_t_detached)  # [batch, seq, d_model]
                
                # ══════════════════════════════════════════════════════════
                # 3️⃣ WRITE PHASE: Update W_mem via Hebbian-like rule
                # ΔW_mem = η * (z_t ⊗ k_t)
                # ══════════════════════════════════════════════════════════
                
                # Aggregate over batch and sequence
                # Use mean to get stable gradients
                z_mean = z_t.mean(dim=(0, 1))  # [memory_rank]
                k_mean = k_t.mean(dim=(0, 1))  # [d_model]
                
                # Check for NaN or Inf - skip update but continue
                if torch.isnan(z_mean).any() or torch.isinf(z_mean).any():
                    pass  # Skip update, will return info at end
                else:
                    # Normalize to unit vectors to prevent explosion
                    z_norm = torch.norm(z_mean, p=2)
                    k_norm = torch.norm(k_mean, p=2)
                    
                    # Only update if norms are valid
                    if z_norm >= 1e-8 and k_norm >= 1e-8:
                        z_normalized = z_mean / z_norm
                        k_normalized = k_mean / k_norm
                        
                        # Proper Hebbian outer product update
                        # W_mem = mem_down @ mem_up where:
                        #   mem_down: [d_model, memory_rank]
                        #   mem_up: [memory_rank, d_model]
                        # Full W_mem would be [d_model, d_model]
                        
                        # We want: ΔW_mem ≈ k ⊗ z to bind patterns together
                        
                        # Update mem_down with outer product: k ⊗ z
                        delta_down = torch.outer(k_normalized, z_normalized)  # [d_model, memory_rank]
                        
                        # Update mem_up with outer product: z ⊗ k  
                        delta_up = torch.outer(z_normalized, k_normalized)  # [memory_rank, d_model]
                        
                        # Apply decay and update with Hebbian rule
                        # W_mem ← α * W_mem + η * ΔW_mem
                        self.mem_down.mul_(self.decay_factor).add_(
                            delta_down, alpha=self.plasticity_rate
                        )
                        self.mem_up.mul_(self.decay_factor).add_(
                            delta_up, alpha=self.plasticity_rate
                        )
                        
                        # Increment step counter
                        self.step_counter += 1
                        
                        # ══════════════════════════════════════════════════════════
                        # 4️⃣ CONSOLIDATION PHASE (Optional)
                        # W_core ← W_core + λ * W_mem
                        # ══════════════════════════════════════════════════════════
                        if consolidate or (self.step_counter % self.consolidation_steps == 0):
                            # Reconstruct full W_mem
                            W_mem = self.mem_down @ self.mem_up  # [d_model, d_model]
                            
                            # Consolidate into core
                            self.core_transform.weight.add_(
                                W_mem.t(), alpha=self.consolidation_rate
                            )
                            
                            # Reset plastic memory
                            self.mem_down.zero_()
                            self.mem_up.zero_()
        
        # ══════════════════════════════════════════════════════════
        # RETURN
        # ══════════════════════════════════════════════════════════
        info = {
            'memory_norm': torch.norm(self.mem_down).item(),
            'core_norm': torch.norm(self.core_transform.weight).item(),
            'step': self.step_counter.item(),
            'gate_avg': gate_values.mean().item() if self.enable_gating else 1.0
        }
        
        return h_t, info
    
    def reset_memory(self):
        """Clear plastic memory (useful for starting new contexts)"""
        with torch.no_grad():
            self.mem_down.zero_()
            self.mem_up.zero_()
            self.step_counter.zero_()
    
    def get_memory_capacity_used(self) -> float:
        """Get percentage of memory capacity being used (0-100%)"""
        mem_magnitude = torch.norm(self.mem_down).item()
        max_magnitude = math.sqrt(self.d_model * self.memory_rank)
        return min(100.0, (mem_magnitude / max_magnitude) * 100)


class SimpleMindBlockWithNPMB(nn.Module):
    """
    SimpleMindBlock augmented with Neural Plastic Memory
    
    Integrates NPMB into the SimpleMind architecture:
    - Standard SimpleMind operations (local conv + global pooling)
    - NPMB for adaptive inference-time learning
    - Combined via residual connection
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        memory_rank: int = 64,
        plasticity_rate: float = 0.01,
        enable_npmb: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.enable_npmb = enable_npmb
        
        # Standard SimpleMind components
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=num_heads)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Neural Plastic Memory Block
        if enable_npmb:
            self.npmb = NeuralPlasticMemoryBlock(
                d_model=d_model,
                memory_rank=memory_rank,
                plasticity_rate=plasticity_rate
            )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.global_proj.weight)
        nn.init.xavier_uniform_(self.local_conv.weight)
        nn.init.zeros_(self.local_conv.bias)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.global_proj.bias)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward with optional memory update
        
        Returns:
            output: Transformed representation
            memory_info: Statistics about memory updates
        """
        batch_size, seq_len, d_model = x.shape
        
        residual = x
        x = self.layer_norm(x)
        
        # Standard SimpleMind pathway
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            queries = queries * mask
            keys = keys * mask
            values = values * mask
        
        # Local + Global features
        values_t = values.transpose(1, 2)
        local_features = self.local_conv(values_t).transpose(1, 2)
        
        queries_t = queries.transpose(1, 2)
        global_context = self.global_pool(queries_t).squeeze(-1)
        global_context = self.global_proj(global_context)
        global_features = global_context.unsqueeze(1).expand(-1, seq_len, -1)
        
        position_weights = torch.sigmoid(keys)
        combined = local_features + 0.1 * global_features
        combined = combined * position_weights
        
        # NPMB pathway (if enabled)
        memory_info = {}
        if self.enable_npmb:
            mem_out, memory_info = self.npmb(combined, update_memory=update_memory)
            combined = combined + mem_out  # Residual from memory
        
        # Output
        output = self.out_proj(combined)
        output = self.dropout(output)
        
        return output + residual, memory_info


if __name__ == "__main__":
    # Test the NPMB
    print("Testing Neural Plastic Memory Block...")
    
    d_model = 128
    batch_size = 2
    seq_len = 10
    
    npmb = NeuralPlasticMemoryBlock(d_model=d_model, memory_rank=32)
    npmb.eval()  # Inference mode
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("\nInitial state:")
    output, info = npmb(x, update_memory=False)
    print(f"  Memory norm: {info['memory_norm']:.4f}")
    print(f"  Core norm: {info['core_norm']:.4f}")
    
    print("\nAfter 10 inference steps with memory updates:")
    for i in range(10):
        x = torch.randn(batch_size, seq_len, d_model)
        output, info = npmb(x, update_memory=True)
    
    print(f"  Memory norm: {info['memory_norm']:.4f}")
    print(f"  Memory capacity used: {npmb.get_memory_capacity_used():.2f}%")
    print(f"  Steps: {info['step']}")
    
    print("\n✅ NPMB test complete!")
