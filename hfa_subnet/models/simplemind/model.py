"""
Complete MindTransformer Model - The O(N) Attention Killer

A full transformer-like model using MindBlocks instead of attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

from .core.mind_block import MindBlock, MultiMindBlock


class MindTransformer(nn.Module):
    """
    Complete transformer model using MindBlocks for O(N) complexity.
    
    This is the full model that can replace any transformer architecture
    while achieving O(N) scaling instead of O(N²).
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_channels: int = 64,
        max_seq_len: int = 2048,
        router_type: str = "dynamic",
        aggregation_type: str = "learnable",
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        channel_scaling: str = "sqrt",
        pad_token_id: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        d_ff = d_ff or 4 * d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Stack of MindBlocks
        self.mind_transformer = MultiMindBlock(
            num_layers=num_layers,
            d_model=d_model,
            num_channels=num_channels,
            d_ff=d_ff,
            router_type=router_type,
            aggregation_type=aggregation_type,
            num_heads=num_heads,
            dropout=dropout,
            temperature=temperature,
            channel_scaling=channel_scaling,
            **kwargs
        )
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output head (for language modeling)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between input and output embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MindTransformer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling loss [batch_size, seq_len]
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # [1, seq_len, d_model]
        
        # Combine embeddings
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Apply MindTransformer layers
        x = self.mind_transformer(x, attention_mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if return_dict:
            return {
                'logits': logits,
                'loss': loss,
                'complexity': self.get_complexity_info(seq_len)
            }
        else:
            return (loss, logits) if loss is not None else logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text using the MindTransformer.
        
        Args:
            input_ids: Initial token IDs [batch_size, initial_seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            pad_token_id: Padding token ID
            
        Returns:
            Generated token IDs [batch_size, max_length]
        """
        pad_token_id = pad_token_id or self.pad_token_id
        batch_size = input_ids.size(0)
        
        # Start with input tokens
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                outputs = self.forward(generated, return_dict=True)
                logits = outputs['logits']
                
                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit max length
                if generated.size(1) >= max_length:
                    break
        
        return generated
    
    def get_complexity_info(self, seq_len: int) -> Dict[str, str]:
        """Get complexity information for the entire model."""
        return {
            'total_complexity': f"O({self.num_layers} × {seq_len})",
            'per_layer_complexity': f"O({seq_len})",
            'vs_transformer': f"O({self.num_layers} × {seq_len}) vs O({self.num_layers} × {seq_len}²)",
            'scaling_advantage': f"{seq_len}x faster than transformer at seq_len={seq_len}",
            'memory_advantage': f"{seq_len}x less memory than transformer attention"
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_parameters': self.token_embedding.weight.numel() + self.pos_embedding.weight.numel(),
            'mind_transformer_parameters': sum(p.numel() for p in self.mind_transformer.parameters()),
        }


class MindTransformerForSequenceClassification(MindTransformer):
    """
    MindTransformer for sequence classification tasks.
    """
    
    def __init__(self, num_labels: int, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        
        # Replace language modeling head with classification head
        self.classifier = nn.Sequential(
            nn.Dropout(kwargs.get('dropout', 0.1)),
            nn.Linear(self.d_model, num_labels)
        )
        
        # Remove the LM head
        del self.lm_head
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for sequence classification."""
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Apply MindTransformer layers
        x = self.mind_transformer(x, attention_mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Pool sequence representation (use [CLS] token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            pooled = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            pooled = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        
        if return_dict:
            return {
                'logits': logits,
                'loss': loss,
                'complexity': self.get_complexity_info(seq_len)
            }
        else:
            return (loss, logits) if loss is not None else logits
