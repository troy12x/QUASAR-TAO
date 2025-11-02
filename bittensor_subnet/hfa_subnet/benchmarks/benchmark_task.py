# The MIT License (MIT)
# Copyright © 2024 HFA Research Team

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

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class BenchmarkTask:
    """
    Represents a single benchmark evaluation task for infinite context evaluation.
    
    This class standardizes benchmark tasks across different datasets (LongBench,
    HotpotQA, GovReport, Needle-in-Haystack) while maintaining compatibility with
    the existing HFA subnet evaluation system.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of benchmark (longbench, hotpotqa, govreport, needle_haystack)
        dataset_name: Specific dataset within the benchmark type
        context: The long context sequence for evaluation
        prompt: The query/task to perform on the context
        expected_output: Expected/reference output for evaluation (if available)
        evaluation_metrics: List of metrics to use for evaluation
        context_length: Length of context in tokens
        difficulty_level: Difficulty classification (easy, medium, hard, extreme)
        metadata: Additional task-specific information
    """
    
    task_id: str
    task_type: str  # "longbench", "hotpotqa", "govreport", "needle_haystack", "synthetic"
    dataset_name: str  # Specific dataset within benchmark type
    context: str
    prompt: str
    expected_output: Optional[str] = None
    evaluation_metrics: List[str] = field(default_factory=lambda: ["accuracy", "coherence"])
    context_length: int = 0
    difficulty_level: str = "medium"  # "easy", "medium", "hard", "extreme"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate context length if not provided"""
        if self.context_length == 0:
            # Simple token approximation (words * 1.3 for subword tokens)
            self.context_length = int(len(self.context.split()) * 1.3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
            "context": self.context,
            "prompt": self.prompt,
            "expected_output": self.expected_output,
            "evaluation_metrics": self.evaluation_metrics,
            "context_length": self.context_length,
            "difficulty_level": self.difficulty_level,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkTask':
        """Create task from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert task to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BenchmarkTask':
        """Create task from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_context_category(self) -> str:
        """Categorize task by context length"""
        if self.context_length < 2000:
            return "short"
        elif self.context_length < 8000:
            return "medium"
        elif self.context_length < 32000:
            return "long"
        else:
            return "ultra_long"
    
    def is_infinite_context_task(self) -> bool:
        """Check if task requires infinite context capabilities"""
        return self.context_length > 16000 or self.difficulty_level == "extreme"
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration for this task"""
        return {
            "metrics": self.evaluation_metrics,
            "context_length": self.context_length,
            "difficulty": self.difficulty_level,
            "task_type": self.task_type,
            "requires_exact_match": "exact_match" in self.evaluation_metrics,
            "requires_semantic_similarity": "semantic_similarity" in self.evaluation_metrics
        }