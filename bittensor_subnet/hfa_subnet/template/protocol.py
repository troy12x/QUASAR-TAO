# The MIT License (MIT)
# Copyright 2023 Yuma Rao
# Copyright 2024 HFA Research Team

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import typing
import bittensor as bt
from typing import List, Optional, Dict, Any
import time

# HFA Infinite Context Subnet Protocol
# This protocol enables evaluation of infinite context language modeling capabilities
# using the revolutionary Hierarchical Flow Anchoring architecture

# ---- miner ----
# Example usage:
#   def infinite_context_forward(synapse: InfiniteContextSynapse) -> InfiniteContextSynapse:
#       # Process with HFA model
#       synapse.response = hfa_model.generate(synapse.context, synapse.prompt)
#       synapse.memory_retention_score = calculate_memory_score(synapse)
#       return synapse
#   axon = bt.axon().attach(infinite_context_forward).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   result = dendrite.query(InfiniteContextSynapse(
#       context="very long context...",
#       prompt="question about early context",
#       max_tokens=100
#   ))

class InfiniteContextSynapse(bt.Synapse):
    """
    HFA Infinite Context Protocol Synapse
    
    This synapse enables evaluation of infinite context capabilities using the
    breakthrough Hierarchical Flow Anchoring architecture. It tests:
    - Memory retention across ultra-long sequences
    - Pattern recognition in extended contexts
    - Coherence maintenance over infinite sequences
    - Speed vs accuracy trade-offs
    
    Attributes:
    - context: The long context sequence (can be 1K-100K+ tokens)
    - prompt: The query/task to perform on the context
    - max_tokens: Maximum tokens to generate in response
    - response: The model's generated response
    - memory_retention_score: Score measuring memory retention accuracy
    - processing_time: Time taken to process the request
    - context_length: Actual length of the context in tokens
    - hfa_model_config: Configuration of the HFA model used
    """

    # Required request inputs, filled by validator
    context: str  # The long context sequence
    prompt: str   # The query/task about the context
    max_tokens: int = 100  # Maximum response tokens
    
    # Evaluation parameters
    evaluation_type: str = "memory_retention"  # Type of evaluation
    target_position: Optional[int] = None  # Position in context to focus on
    pattern_type: Optional[str] = None  # Type of pattern to recognize
    
    # Optional request outputs, filled by miner
    response: Optional[str] = None
    memory_retention_score: Optional[float] = None
    processing_time: Optional[float] = None
    context_length: Optional[int] = None
    hfa_model_config: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    tokens_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    checkpoint_count: Optional[int] = None  # Number of HFA checkpoints created
    
    # Quality metrics
    coherence_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    position_understanding_score: Optional[float] = None
    
    # Model information
    model_info: Optional[Dict[str, Any]] = None

    def deserialize(self) -> Dict[str, Any]:
        """
        Deserialize the infinite context response. This method retrieves the response
        from the miner and returns comprehensive evaluation metrics.

        Returns:
        - Dict[str, Any]: Complete evaluation results including response and metrics

        Example:
        >>> synapse = InfiniteContextSynapse(context="...", prompt="...")
        >>> synapse.response = "Generated response"
        >>> synapse.memory_retention_score = 0.95
        >>> result = synapse.deserialize()
        >>> print(result["memory_retention_score"])  # 0.95
        """
        return {
            "response": self.response,
            "memory_retention_score": self.memory_retention_score,
            "processing_time": self.processing_time,
            "context_length": self.context_length,
            "model_config": self.model_config,
            "tokens_per_second": self.tokens_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "checkpoint_count": self.checkpoint_count,
            "coherence_score": self.coherence_score,
            "accuracy_score": self.accuracy_score,
            "position_understanding_score": self.position_understanding_score,
            "evaluation_type": self.evaluation_type,
            "target_position": self.target_position,
            "pattern_type": self.pattern_type
        }


class MemoryRetentionSynapse(bt.Synapse):
    """
    Specialized synapse for testing memory retention across long sequences.
    Tests the core breakthrough of HFA: 100% memory retention at all positions.
    """
    
    # Input sequence with embedded information at different positions
    sequence: str
    memory_targets: List[Dict[str, Any]]  # Information to remember at different positions
    query_position: int  # Position to query about
    
    # Response
    retrieved_info: Optional[str] = None
    confidence_score: Optional[float] = None
    position_accuracy: Optional[float] = None

    def deserialize(self) -> Dict[str, Any]:
        return {
            "retrieved_info": self.retrieved_info,
            "confidence_score": self.confidence_score,
            "position_accuracy": self.position_accuracy
        }


class PatternRecognitionSynapse(bt.Synapse):
    """
    Specialized synapse for testing pattern recognition in extended contexts.
    Leverages HFA's superior long-context pattern recognition capabilities.
    """
    
    # Long sequence with embedded patterns
    sequence: str
    pattern_type: str  # "fibonacci", "prime", "alternating", etc.
    sequence_length: int
    
    # Response
    detected_patterns: Optional[List[str]] = None
    pattern_accuracy: Optional[float] = None
    detection_confidence: Optional[float] = None

    def deserialize(self) -> Dict[str, Any]:
        return {
            "detected_patterns": self.detected_patterns,
            "pattern_accuracy": self.pattern_accuracy,
            "detection_confidence": self.detection_confidence
        }


class ScalingTestSynapse(bt.Synapse):
    """
    Specialized synapse for testing infinite context scaling capabilities.
    Validates HFA's ability to maintain performance as context length increases.
    """
    
    # Scaling test parameters
    base_context: str
    target_length: int  # Target context length to scale to
    scaling_factor: int  # How much to extend the context
    
    # Performance tracking
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Response
    scaled_response: Optional[str] = None
    scaling_efficiency: Optional[float] = None
    memory_stability: Optional[float] = None

    def deserialize(self) -> Dict[str, Any]:
        return {
            "scaled_response": self.scaled_response,
            "scaling_efficiency": self.scaling_efficiency,
            "memory_stability": self.memory_stability,
            "processing_time": self.end_time - self.start_time if self.start_time and self.end_time else None
        }
