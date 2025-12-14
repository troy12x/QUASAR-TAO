# The MIT License (MIT)
# Copyright © 2024 HFA Research Team

"""
Benchmark Task for HFA Subnet

Represents individual benchmark tasks for evaluation with support for:
- Variable context lengths (32k to 2M+)
- Multi-modal inputs (text, potential future expansion)
- Diverse evaluation metrics
- Metadata for analysis
"""

import bittensor as bt
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import time

@dataclass
class BenchmarkTask:
    """
    Represents a single benchmark task for Long Context evaluation.
    """
    
    # Core Identity
    task_id: str
    dataset_name: str # e.g., "narrativeqa", "gov_report"
    task_type: str # "qa", "summarization", "retrieval", "code_completion"
    
    # Content
    context: str
    prompt: str
    expected_output: Optional[str]
    
    # Metadata for complexity and routing
    context_length: int = 0
    difficulty_level: str = "medium" # "easy", "medium", "hard", "extreme" (1M+)
    
    # Grading details
    evaluation_metrics: List[str] = field(default_factory=list) # e.g. ["f1", "rouge", "accuracy"]
    all_classes: Optional[List[str]] = None # For classification tasks
    
    # Provenance
    source: str = "longbench" # or "synthetic", "internal"
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.context and not self.context_length:
            # Approx token count (fast)
            self.context_length = len(self.context) // 4 
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization/logging."""
        return {
            'task_id': self.task_id,
            'dataset_name': self.dataset_name,
            'task_type': self.task_type,
            'context_length': self.context_length,
            'difficulty_level': self.difficulty_level,
            'evaluation_metrics': self.evaluation_metrics,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkTask':
        """Create task from dictionary."""
        return cls(
            task_id=data.get('task_id', 'unknown'),
            dataset_name=data.get('dataset_name', 'unknown'),
            task_type=data.get('task_type', 'general'),
            context=data.get('context', ''),
            prompt=data.get('prompt', ''),
            expected_output=data.get('expected_output'),
            context_length=data.get('context_length', 0),
            difficulty_level=data.get('difficulty_level', 'medium'),
            evaluation_metrics=data.get('evaluation_metrics', []),
            all_classes=data.get('all_classes'),
            source=data.get('source', 'unknown')
        )

    def validate(self) -> bool:
        """Basic integrity check."""
        return bool(self.context and self.prompt and self.task_id)
