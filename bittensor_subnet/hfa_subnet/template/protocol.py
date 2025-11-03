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
import hashlib
from dataclasses import dataclass
from datetime import datetime

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

@dataclass
class BenchmarkTaskInfo:
    """Information about a benchmark task for protocol transmission"""
    task_id: str
    task_type: str  # "longbench", "hotpotqa", "govreport", "needle_haystack", "synthetic"
    dataset_name: str
    difficulty_level: str
    evaluation_metrics: List[str]
    expected_output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    context_length: Optional[int] = None
    requires_exact_match: bool = False
    requires_semantic_similarity: bool = False
    supports_perturbation_testing: bool = True
    
    def __post_init__(self):
        """Initialize derived fields based on evaluation metrics"""
        if self.evaluation_metrics:
            self.requires_exact_match = "exact_match" in self.evaluation_metrics
            self.requires_semantic_similarity = "semantic_similarity" in self.evaluation_metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
            "difficulty_level": self.difficulty_level,
            "evaluation_metrics": self.evaluation_metrics,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
            "context_length": self.context_length,
            "requires_exact_match": self.requires_exact_match,
            "requires_semantic_similarity": self.requires_semantic_similarity,
            "supports_perturbation_testing": self.supports_perturbation_testing
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkTaskInfo':
        """Create from dictionary"""
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate benchmark task information"""
        if not self.task_id or not self.task_type or not self.dataset_name:
            return False
        
        valid_task_types = ["longbench", "hotpotqa", "govreport", "needle_haystack", "synthetic"]
        if self.task_type not in valid_task_types:
            return False
        
        valid_difficulty_levels = ["easy", "medium", "hard", "extreme"]
        if self.difficulty_level not in valid_difficulty_levels:
            return False
        
        return True


class InfiniteContextSynapse(bt.Synapse):
    """
    HFA Infinite Context Protocol Synapse
    
    This synapse enables evaluation of infinite context capabilities using the
    breakthrough Hierarchical Flow Anchoring architecture. It tests:
    - Memory retention across ultra-long sequences
    - Pattern recognition in extended contexts
    - Coherence maintenance over infinite sequences
    - Speed vs accuracy trade-offs
    - Real-world benchmark tasks (LongBench, HotpotQA, GovReport, Needle-in-Haystack)
    
    Attributes:
    - context: The long context sequence (can be 1K-100K+ tokens)
    - prompt: The query/task to perform on the context
    - max_tokens: Maximum tokens to generate in response
    - response: The model's generated response
    - memory_retention_score: Score measuring memory retention accuracy
    - processing_time: Time taken to process the request
    - context_length: Actual length of the context in tokens
    - hfa_model_config: Configuration of the HFA model used
    - benchmark_task: Information about benchmark task (if applicable)
    """

    # Required request inputs, filled by validator
    context: str  # The long context sequence
    prompt: str   # The query/task about the context
    max_tokens: int = 100  # Maximum response tokens
    
    # Evaluation parameters
    evaluation_type: str = "memory_retention"  # Type of evaluation
    target_position: Optional[int] = None  # Position in context to focus on
    pattern_type: Optional[str] = None  # Type of pattern to recognize
    
    # Benchmark task information (enhanced)
    benchmark_task: Optional[BenchmarkTaskInfo] = None
    is_benchmark_task: bool = False
    benchmark_seed: Optional[int] = None  # For reproducible evaluation
    benchmark_shard_id: Optional[str] = None  # For dataset sharding
    supports_perturbation: bool = True  # Whether task supports perturbation testing
    
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
    
    # Benchmark-specific quality metrics (enhanced)
    exact_match_score: Optional[float] = None
    f1_score: Optional[float] = None
    rouge_l_score: Optional[float] = None
    semantic_similarity_score: Optional[float] = None
    needle_retrieval_score: Optional[float] = None
    multi_hop_reasoning_score: Optional[float] = None
    
    # Additional benchmark metrics for comprehensive evaluation
    bleu_score: Optional[float] = None  # For translation/generation tasks
    bertscore: Optional[float] = None  # Semantic similarity using BERT
    factual_consistency_score: Optional[float] = None  # For factual accuracy
    reading_comprehension_score: Optional[float] = None  # For LongBench tasks
    summarization_quality_score: Optional[float] = None  # For GovReport tasks
    retrieval_precision: Optional[float] = None  # For needle-in-haystack precision
    retrieval_recall: Optional[float] = None  # For needle-in-haystack recall
    
    # Per-item evaluation metrics (for detailed analysis)
    per_item_scores: Optional[Dict[str, float]] = None
    evaluation_breakdown: Optional[Dict[str, Any]] = None
    
    # Model information
    model_info: Optional[Dict[str, Any]] = None
    
    # Architecture support fields (unified architecture support)
    architecture_type: Optional[str] = None  # "hfa", "simplemind", "hybrid", "standard"
    model_configuration: Optional[Dict[str, Any]] = None  # Architecture-specific configuration
    architecture_preference: Optional[str] = None  # Preferred architecture for this task
    
    # Architecture-specific performance metrics
    hfa_checkpoint_count: Optional[int] = None  # Number of HFA checkpoints created
    simplemind_block_count: Optional[int] = None  # Number of SimpleMind blocks used
    hybrid_component_usage: Optional[Dict[str, float]] = None  # Usage ratio of hybrid components
    architecture_switching_count: Optional[int] = None  # Number of architecture switches
    
    # Audit and hash fields for sealed scoring harness
    logit_hash: Optional[str] = None  # Hash of model logits for audit verification
    model_signature: Optional[str] = None  # Signature of model configuration
    audit_trail: Optional[List[str]] = None  # Audit trail entries
    response_hash: Optional[str] = None  # Hash of response content
    evaluation_timestamp: Optional[float] = None  # Timestamp of evaluation
    validator_signature: Optional[str] = None  # Signature of validating node
    protocol_version: Optional[str] = None  # Protocol version for compatibility
    
    # Perturbation testing support
    is_perturbation_test: bool = False  # Whether this is a perturbation test
    original_task_id: Optional[str] = None  # ID of original task if this is perturbation
    perturbation_type: Optional[str] = None  # Type of perturbation applied
    expected_consistency_threshold: Optional[float] = None  # Expected consistency level

    def deserialize(self) -> Dict[str, Any]:
        """
        Deserialize the infinite context response. This method retrieves the response
        from the miner and returns comprehensive evaluation metrics including
        benchmark-specific metrics.

        Returns:
        - Dict[str, Any]: Complete evaluation results including response and metrics

        Example:
        >>> synapse = InfiniteContextSynapse(context="...", prompt="...")
        >>> synapse.response = "Generated response"
        >>> synapse.memory_retention_score = 0.95
        >>> result = synapse.deserialize()
        >>> print(result["memory_retention_score"])  # 0.95
        """
        result = {
            "response": self.response,
            "memory_retention_score": self.memory_retention_score,
            "processing_time": self.processing_time,
            "context_length": self.context_length,
            "hfa_model_config": self.hfa_model_config,
            "tokens_per_second": self.tokens_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "checkpoint_count": self.checkpoint_count,
            "coherence_score": self.coherence_score,
            "accuracy_score": self.accuracy_score,
            "position_understanding_score": self.position_understanding_score,
            "evaluation_type": self.evaluation_type,
            "target_position": self.target_position,
            "pattern_type": self.pattern_type,
            # Benchmark task information
            "is_benchmark_task": self.is_benchmark_task,
            "benchmark_task": self.benchmark_task.to_dict() if self.benchmark_task else None,
            "benchmark_seed": self.benchmark_seed,
            "benchmark_shard_id": self.benchmark_shard_id,
            "supports_perturbation": self.supports_perturbation,
            # Benchmark-specific metrics
            "exact_match_score": self.exact_match_score,
            "f1_score": self.f1_score,
            "rouge_l_score": self.rouge_l_score,
            "semantic_similarity_score": self.semantic_similarity_score,
            "needle_retrieval_score": self.needle_retrieval_score,
            "multi_hop_reasoning_score": self.multi_hop_reasoning_score,
            "bleu_score": self.bleu_score,
            "bertscore": self.bertscore,
            "factual_consistency_score": self.factual_consistency_score,
            "reading_comprehension_score": self.reading_comprehension_score,
            "summarization_quality_score": self.summarization_quality_score,
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "per_item_scores": self.per_item_scores,
            "evaluation_breakdown": self.evaluation_breakdown,
            "model_info": self.model_info,
            # Architecture support information
            "architecture_type": self.architecture_type,
            "model_configuration": self.model_configuration,
            "architecture_preference": self.architecture_preference,
            "hfa_checkpoint_count": self.hfa_checkpoint_count,
            "simplemind_block_count": self.simplemind_block_count,
            "hybrid_component_usage": self.hybrid_component_usage,
            "architecture_switching_count": self.architecture_switching_count,
            # Audit and hash information
            "logit_hash": self.logit_hash,
            "model_signature": self.model_signature,
            "audit_trail": self.audit_trail,
            "response_hash": self.response_hash,
            "evaluation_timestamp": self.evaluation_timestamp,
            "validator_signature": self.validator_signature,
            "protocol_version": self.protocol_version,
            # Perturbation testing information
            "is_perturbation_test": self.is_perturbation_test,
            "original_task_id": self.original_task_id,
            "perturbation_type": self.perturbation_type,
            "expected_consistency_threshold": self.expected_consistency_threshold
        }
        
        # Preserve audit information during deserialization
        result = self.preserve_audit_on_deserialize(result)
        
        return result
    
    def validate_benchmark_task(self) -> bool:
        """Validate benchmark task configuration"""
        if not self.is_benchmark_task:
            return True
        
        if not self.benchmark_task:
            return False
        
        return self.benchmark_task.validate()
    
    def get_primary_benchmark_score(self) -> Optional[float]:
        """Get the primary evaluation score based on benchmark task type"""
        if not self.is_benchmark_task or not self.benchmark_task:
            return self.accuracy_score
        
        task_type = self.benchmark_task.task_type
        
        if task_type == "needle_haystack":
            return self.needle_retrieval_score or self.exact_match_score
        elif task_type == "hotpotqa":
            return self.multi_hop_reasoning_score or self.f1_score
        elif task_type == "govreport":
            return self.summarization_quality_score or self.rouge_l_score
        elif task_type == "longbench":
            return self.reading_comprehension_score or self.f1_score
        else:
            return self.exact_match_score or self.accuracy_score
    
    def get_composite_benchmark_score(self) -> float:
        """Calculate composite score across all available benchmark metrics"""
        if not self.is_benchmark_task:
            return self.accuracy_score or 0.0
        
        scores = []
        
        # Add available benchmark scores
        for score in [
            self.exact_match_score, self.f1_score, self.rouge_l_score,
            self.semantic_similarity_score, self.needle_retrieval_score,
            self.multi_hop_reasoning_score, self.bleu_score, self.bertscore,
            self.factual_consistency_score, self.reading_comprehension_score,
            self.summarization_quality_score, self.retrieval_precision,
            self.retrieval_recall
        ]:
            if score is not None:
                scores.append(score)
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def supports_metric(self, metric_name: str) -> bool:
        """Check if the current benchmark task supports a specific metric"""
        if not self.is_benchmark_task or not self.benchmark_task:
            return False
        
        return metric_name in self.benchmark_task.evaluation_metrics
    
    def get_context_length_category(self) -> str:
        """Categorize the task by context length for evaluation purposes"""
        if not self.context_length:
            return "unknown"
        
        if self.context_length < 2000:
            return "short"  # < 2K tokens
        elif self.context_length < 8000:
            return "medium"  # 2K-8K tokens
        elif self.context_length < 32000:
            return "long"  # 8K-32K tokens
        elif self.context_length < 100000:
            return "ultra_long"  # 32K-100K tokens
        else:
            return "infinite"  # 100K+ tokens
    
    def add_audit_entry(self, entry: str):
        """Add entry to audit trail"""
        if self.audit_trail is None:
            self.audit_trail = []
        
        timestamp_str = datetime.fromtimestamp(time.time()).isoformat()
        self.audit_trail.append(f"[{timestamp_str}] {entry}")
    
    def compute_response_hash(self) -> str:
        """Compute hash of response content for audit verification"""
        import hashlib
        
        if not self.response:
            return ""
        
        response_content = str(self.response)
        return hashlib.sha256(response_content.encode('utf-8')).hexdigest()[:16]
    
    def set_evaluation_timestamp(self):
        """Set evaluation timestamp to current time"""
        self.evaluation_timestamp = time.time()
    
    def validate_audit_integrity(self) -> bool:
        """Validate audit trail integrity"""
        if not self.audit_trail:
            return True  # No audit trail to validate
        
        # Check that audit entries are properly formatted
        for entry in self.audit_trail:
            if not entry.startswith('[') or '] ' not in entry:
                return False
        
        return True
    
    def create_perturbation_variant(self, perturbation_type: str, 
                                  original_task_id: str) -> 'InfiniteContextSynapse':
        """Create a perturbation variant of this synapse"""
        # Create a copy of the current synapse
        variant = InfiniteContextSynapse(
            context=self.context,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            evaluation_type=self.evaluation_type,
            target_position=self.target_position,
            pattern_type=self.pattern_type,
            benchmark_task=self.benchmark_task,
            is_benchmark_task=self.is_benchmark_task,
            benchmark_seed=self.benchmark_seed,
            benchmark_shard_id=self.benchmark_shard_id,
            supports_perturbation=self.supports_perturbation
        )
        
        # Set perturbation-specific fields
        variant.is_perturbation_test = True
        variant.original_task_id = original_task_id
        variant.perturbation_type = perturbation_type
        variant.expected_consistency_threshold = 0.85  # Default threshold
        
        return variant
    
    def validate_architecture_config(self) -> bool:
        """Validate architecture type and configuration"""
        if not self.architecture_type:
            return True  # Architecture type is optional
        
        valid_architectures = ["hfa", "simplemind", "hybrid", "standard"]
        if self.architecture_type not in valid_architectures:
            return False
        
        # Validate architecture-specific configuration
        if self.model_configuration:
            if self.architecture_type == "hybrid" and "components" not in self.model_configuration:
                return False
        
        return True
    
    def get_architecture_complexity_info(self) -> Dict[str, Any]:
        """Get computational complexity information for the current architecture"""
        if not self.architecture_type:
            return {}
        
        complexity_info = {
            "architecture_type": self.architecture_type,
            "context_length": self.context_length or 0
        }
        
        if self.architecture_type == "hfa":
            complexity_info["checkpoint_count"] = self.hfa_checkpoint_count or 0
            complexity_info["complexity"] = "O(n log n)"
        elif self.architecture_type == "simplemind":
            complexity_info["block_count"] = self.simplemind_block_count or 0
            complexity_info["complexity"] = "O(n)"
        elif self.architecture_type == "hybrid":
            complexity_info["component_usage"] = self.hybrid_component_usage or {}
            complexity_info["switching_count"] = self.architecture_switching_count or 0
            complexity_info["complexity"] = "O(n log n) + O(n)"
        else:  # standard
            complexity_info["complexity"] = "O(n²)"
        
        return complexity_info
    
    def set_protocol_version(self, version: str = "2.0"):
        """Set protocol version for compatibility tracking"""
        self.protocol_version = version
    
    def is_compatible_with_version(self, required_version: str) -> bool:
        """Check if current protocol version is compatible with required version"""
        if not self.protocol_version:
            return False
        
        # Simple version comparison (assumes semantic versioning)
        current_parts = self.protocol_version.split('.')
        required_parts = required_version.split('.')
        
        try:
            current_major = int(current_parts[0])
            required_major = int(required_parts[0])
            
            # Major version must match for compatibility
            return current_major == required_major
        except (ValueError, IndexError):
            return False
    
    def preserve_audit_on_deserialize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure audit information is preserved during deserialization"""
        # Ensure audit fields are not lost during serialization/deserialization
        audit_fields = [
            "logit_hash", "audit_trail", "response_hash", 
            "evaluation_timestamp", "validator_signature", "protocol_version"
        ]
        
        for field in audit_fields:
            if hasattr(self, field) and getattr(self, field) is not None:
                result[field] = getattr(self, field)
        
        return result


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


class BenchmarkEvaluationSynapse(bt.Synapse):
    """
    Specialized synapse for real-world benchmark evaluation tasks.
    
    This synapse is optimized for evaluating models on established benchmarks
    like LongBench, HotpotQA, GovReport, and Needle-in-Haystack tasks.
    It provides comprehensive metrics specific to each benchmark type and
    supports the enhanced evaluation requirements.
    """
    
    # Benchmark task specification (enhanced)
    task_id: str
    task_type: str  # "longbench", "hotpotqa", "govreport", "needle_haystack", "synthetic"
    dataset_name: str
    context: str
    prompt: str
    max_tokens: int = 200
    
    # Task metadata (enhanced)
    difficulty_level: str = "medium"
    evaluation_metrics: List[str] = None
    expected_output: Optional[str] = None
    context_length: Optional[int] = None
    
    # Reproducibility and sharding support
    benchmark_seed: Optional[int] = None
    benchmark_shard_id: Optional[str] = None
    supports_perturbation: bool = True
    
    # Response fields
    response: Optional[str] = None
    processing_time: Optional[float] = None
    
    # Performance metrics (enhanced)
    tokens_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    peak_memory_usage_mb: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None
    
    # Benchmark-specific evaluation scores (enhanced)
    exact_match_score: Optional[float] = None
    f1_score: Optional[float] = None
    rouge_l_score: Optional[float] = None
    semantic_similarity_score: Optional[float] = None
    bleu_score: Optional[float] = None
    bertscore: Optional[float] = None
    
    # Task-specific scores (enhanced)
    needle_retrieval_score: Optional[float] = None  # For needle-in-haystack
    multi_hop_reasoning_score: Optional[float] = None  # For HotpotQA
    summarization_quality_score: Optional[float] = None  # For GovReport
    reading_comprehension_score: Optional[float] = None  # For LongBench
    factual_consistency_score: Optional[float] = None  # Cross-task factual accuracy
    
    # Retrieval-specific metrics
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None
    retrieval_f1: Optional[float] = None
    
    # Quality metrics (enhanced)
    coherence_score: Optional[float] = None
    relevance_score: Optional[float] = None
    completeness_score: Optional[float] = None
    fluency_score: Optional[float] = None
    
    # Per-item detailed metrics
    per_item_scores: Optional[Dict[str, float]] = None
    evaluation_breakdown: Optional[Dict[str, Any]] = None
    
    # Model information (enhanced)
    model_architecture: Optional[str] = None  # "hfa", "simplemind", "hybrid", "standard"
    model_configuration: Optional[Dict[str, Any]] = None
    model_signature: Optional[str] = None  # For audit trails
    
    # Audit and hash fields for sealed scoring harness
    logit_hash: Optional[str] = None  # Hash of model logits for audit verification
    audit_trail: Optional[List[str]] = None  # Audit trail entries
    response_hash: Optional[str] = None  # Hash of response content
    evaluation_timestamp: Optional[float] = None  # Timestamp of evaluation
    validator_signature: Optional[str] = None  # Signature of validating node
    protocol_version: Optional[str] = None  # Protocol version for compatibility
    
    # Perturbation testing support
    is_perturbation_test: bool = False  # Whether this is a perturbation test
    original_task_id: Optional[str] = None  # ID of original task if this is perturbation
    perturbation_type: Optional[str] = None  # Type of perturbation applied
    expected_consistency_threshold: Optional[float] = None  # Expected consistency level
    
    def __post_init__(self):
        """Initialize evaluation metrics and derived fields if not provided"""
        if self.evaluation_metrics is None:
            # Set default metrics based on task type
            if self.task_type == "needle_haystack":
                self.evaluation_metrics = ["exact_match", "retrieval_precision", "retrieval_recall"]
            elif self.task_type == "hotpotqa":
                self.evaluation_metrics = ["f1_score", "multi_hop_reasoning", "factual_consistency"]
            elif self.task_type == "govreport":
                self.evaluation_metrics = ["rouge_l", "summarization_quality", "coherence"]
            elif self.task_type == "longbench":
                self.evaluation_metrics = ["f1_score", "reading_comprehension", "semantic_similarity"]
            else:
                self.evaluation_metrics = ["accuracy", "coherence"]
        
        if self.context_length is None and self.context:
            # Approximate token count (words * 1.3 for subword tokens)
            self.context_length = int(len(self.context.split()) * 1.3)
    
    def deserialize(self) -> Dict[str, Any]:
        """Deserialize benchmark evaluation response with comprehensive metrics"""
        return {
            # Task information
            "task_id": self.task_id,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
            "difficulty_level": self.difficulty_level,
            "evaluation_metrics": self.evaluation_metrics,
            "context_length": self.context_length,
            
            # Reproducibility information
            "benchmark_seed": self.benchmark_seed,
            "benchmark_shard_id": self.benchmark_shard_id,
            "supports_perturbation": self.supports_perturbation,
            
            # Response
            "response": self.response,
            "expected_output": self.expected_output,
            "processing_time": self.processing_time,
            
            # Performance metrics (enhanced)
            "tokens_per_second": self.tokens_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "peak_memory_usage_mb": self.peak_memory_usage_mb,
            "gpu_memory_usage_mb": self.gpu_memory_usage_mb,
            
            # Evaluation scores (enhanced)
            "exact_match_score": self.exact_match_score,
            "f1_score": self.f1_score,
            "rouge_l_score": self.rouge_l_score,
            "semantic_similarity_score": self.semantic_similarity_score,
            "bleu_score": self.bleu_score,
            "bertscore": self.bertscore,
            
            # Task-specific scores (enhanced)
            "needle_retrieval_score": self.needle_retrieval_score,
            "multi_hop_reasoning_score": self.multi_hop_reasoning_score,
            "summarization_quality_score": self.summarization_quality_score,
            "reading_comprehension_score": self.reading_comprehension_score,
            "factual_consistency_score": self.factual_consistency_score,
            
            # Retrieval-specific metrics
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "retrieval_f1": self.retrieval_f1,
            
            # Quality metrics (enhanced)
            "coherence_score": self.coherence_score,
            "relevance_score": self.relevance_score,
            "completeness_score": self.completeness_score,
            "fluency_score": self.fluency_score,
            
            # Detailed metrics
            "per_item_scores": self.per_item_scores,
            "evaluation_breakdown": self.evaluation_breakdown,
            
            # Model information (enhanced)
            "model_architecture": self.model_architecture,
            "model_configuration": self.model_configuration,
            "model_signature": self.model_signature,
            
            # Audit and hash information
            "logit_hash": self.logit_hash,
            "audit_trail": self.audit_trail,
            "response_hash": self.response_hash,
            "evaluation_timestamp": self.evaluation_timestamp,
            "validator_signature": self.validator_signature,
            "protocol_version": self.protocol_version,
            
            # Perturbation testing information
            "is_perturbation_test": self.is_perturbation_test,
            "original_task_id": self.original_task_id,
            "perturbation_type": self.perturbation_type,
            "expected_consistency_threshold": self.expected_consistency_threshold
        }
        
        # Preserve audit information during deserialization
        result = self.preserve_audit_on_deserialize(result)
        
        return result
    
    def get_primary_score(self) -> Optional[float]:
        """Get the primary evaluation score based on task type"""
        if self.task_type == "needle_haystack":
            return self.needle_retrieval_score or self.exact_match_score or self.retrieval_f1
        elif self.task_type == "hotpotqa":
            return self.multi_hop_reasoning_score or self.f1_score
        elif self.task_type == "govreport":
            return self.summarization_quality_score or self.rouge_l_score
        elif self.task_type == "longbench":
            return self.reading_comprehension_score or self.f1_score
        elif self.task_type == "synthetic":
            return self.exact_match_score or self.coherence_score
        else:
            return self.exact_match_score or self.coherence_score
    
    def get_composite_score(self) -> float:
        """Calculate composite score across all available metrics"""
        scores = []
        
        # Add available scores with appropriate weighting
        primary_scores = [
            self.exact_match_score, self.f1_score, self.rouge_l_score,
            self.semantic_similarity_score, self.needle_retrieval_score,
            self.multi_hop_reasoning_score, self.summarization_quality_score,
            self.reading_comprehension_score, self.factual_consistency_score
        ]
        
        quality_scores = [
            self.coherence_score, self.relevance_score, self.completeness_score,
            self.fluency_score
        ]
        
        retrieval_scores = [
            self.retrieval_precision, self.retrieval_recall, self.retrieval_f1
        ]
        
        # Weight primary scores more heavily (70%)
        primary_valid = [s for s in primary_scores if s is not None]
        quality_valid = [s for s in quality_scores if s is not None]
        retrieval_valid = [s for s in retrieval_scores if s is not None]
        
        if not primary_valid and not quality_valid and not retrieval_valid:
            return 0.0
        
        composite = 0.0
        weight_sum = 0.0
        
        if primary_valid:
            composite += 0.7 * (sum(primary_valid) / len(primary_valid))
            weight_sum += 0.7
        
        if quality_valid:
            composite += 0.2 * (sum(quality_valid) / len(quality_valid))
            weight_sum += 0.2
        
        if retrieval_valid:
            composite += 0.1 * (sum(retrieval_valid) / len(retrieval_valid))
            weight_sum += 0.1
        
        return composite / weight_sum if weight_sum > 0 else 0.0
    
    def validate_task(self) -> bool:
        """Validate benchmark task configuration"""
        if not self.task_id or not self.task_type or not self.dataset_name:
            return False
        
        valid_task_types = ["longbench", "hotpotqa", "govreport", "needle_haystack", "synthetic"]
        if self.task_type not in valid_task_types:
            return False
        
        valid_difficulty_levels = ["easy", "medium", "hard", "extreme"]
        if self.difficulty_level not in valid_difficulty_levels:
            return False
        
        if not self.context or not self.prompt:
            return False
        
        return True
    
    def supports_metric(self, metric_name: str) -> bool:
        """Check if the current task supports a specific evaluation metric"""
        return metric_name in self.evaluation_metrics
    
    def get_context_length_category(self) -> str:
        """Categorize the task by context length"""
        if not self.context_length:
            return "unknown"
        
        if self.context_length < 2000:
            return "short"  # < 2K tokens
        elif self.context_length < 8000:
            return "medium"  # 2K-8K tokens  
        elif self.context_length < 32000:
            return "long"  # 8K-32K tokens
        elif self.context_length < 100000:
            return "ultra_long"  # 32K-100K tokens
        else:
            return "infinite"  # 100K+ tokens
    
    def add_audit_entry(self, entry: str):
        """Add entry to audit trail"""
        if self.audit_trail is None:
            self.audit_trail = []
        
        timestamp_str = datetime.fromtimestamp(time.time()).isoformat()
        self.audit_trail.append(f"[{timestamp_str}] {entry}")
    
    def compute_response_hash(self) -> str:
        """Compute hash of response content for audit verification"""
        if not self.response:
            return ""
        
        response_content = str(self.response)
        return hashlib.sha256(response_content.encode('utf-8')).hexdigest()[:16]
    
    def set_evaluation_timestamp(self):
        """Set evaluation timestamp to current time"""
        self.evaluation_timestamp = time.time()
    
    def validate_audit_integrity(self) -> bool:
        """Validate audit trail integrity"""
        if not self.audit_trail:
            return True  # No audit trail to validate
        
        # Check that audit entries are properly formatted
        for entry in self.audit_trail:
            if not entry.startswith('[') or '] ' not in entry:
                return False
        
        return True
    
    def set_protocol_version(self, version: str = "2.0"):
        """Set protocol version for compatibility tracking"""
        self.protocol_version = version
    
    def is_compatible_with_version(self, required_version: str) -> bool:
        """Check if current protocol version is compatible with required version"""
        if not self.protocol_version:
            return False
        
        # Simple version comparison (assumes semantic versioning)
        current_parts = self.protocol_version.split('.')
        required_parts = required_version.split('.')
        
        try:
            current_major = int(current_parts[0])
            required_major = int(required_parts[0])
            
            # Major version must match for compatibility
            return current_major == required_major
        except (ValueError, IndexError):
            return False
    
    def preserve_audit_on_deserialize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure audit information is preserved during deserialization"""
        # Ensure audit fields are not lost during serialization/deserialization
        audit_fields = [
            "logit_hash", "audit_trail", "response_hash", 
            "evaluation_timestamp", "validator_signature", "protocol_version"
        ]
        
        for field in audit_fields:
            if hasattr(self, field) and getattr(self, field) is not None:
                result[field] = getattr(self, field)
        
        return result
    
    def create_perturbation_variant(self, perturbation_type: str, 
                                  original_task_id: str) -> 'BenchmarkEvaluationSynapse':
        """Create a perturbation variant of this synapse"""
        # Create a copy of the current synapse
        variant = BenchmarkEvaluationSynapse(
            task_id=f"{self.task_id}_perturb_{perturbation_type}",
            task_type=self.task_type,
            dataset_name=self.dataset_name,
            context=self.context,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            difficulty_level=self.difficulty_level,
            evaluation_metrics=self.evaluation_metrics,
            expected_output=self.expected_output,
            context_length=self.context_length,
            benchmark_seed=self.benchmark_seed,
            benchmark_shard_id=self.benchmark_shard_id,
            supports_perturbation=self.supports_perturbation
        )
        
        # Set perturbation-specific fields
        variant.is_perturbation_test = True
        variant.original_task_id = original_task_id
        variant.perturbation_type = perturbation_type
        variant.expected_consistency_threshold = 0.85  # Default threshold
        
        return variant
