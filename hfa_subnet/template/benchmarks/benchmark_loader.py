# The MIT License (MIT)
# Copyright © 2024 HFA Research Team

"""
Benchmark Loader for HFA Subnet

Loads and manages benchmark tasks for validator evaluation.
Supports:
- LongBench (standard)
- Synthetic scaling tasks (up to 2M tokens)
- Caching and filtering
"""

import bittensor as bt
import random
from typing import List, Dict, Any, Optional, Tuple
from .benchmark_task import BenchmarkTask
# import datasets  # If allowed, otherwise we use local loading

class BenchmarkLoader:
    """
    Loads benchmark tasks for validator testing.
    Manages the "Curriculum" of the subnet.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize benchmark loader."""
        self.config = config or {}
        self.longbench_config = self.config.get('longbench', {})
        self.enabled_datasets = self.longbench_config.get('enabled_tasks', ['narrativeqa', 'gov_report'])
        self.data_path = self.longbench_config.get('data_path', 'data/longbench')
        
        self.cache = {} # task_type -> List[BenchmarkTask]
        bt.logging.info(f"📚 BenchmarkLoader initialized | Datasets: {self.enabled_datasets}")

    def load_benchmark_tasks(
        self, 
        num_tasks: int = 1, 
        benchmark_types: Optional[List[str]] = None,
        difficulty: Optional[str] = None
    ) -> List[BenchmarkTask]:
        """
        Load a batch of benchmark tasks.
        """
        tasks = []
        types_to_load = benchmark_types or ['longbench']
        
        for b_type in types_to_load:
            if b_type == 'longbench':
                tasks.extend(self._load_longbench_tasks(num_tasks, difficulty))
            elif b_type == 'synthetic':
                 tasks.extend(self._generate_synthetic_tasks(num_tasks, difficulty))
        
        # Shuffle and return requested amount
        random.shuffle(tasks)
        return tasks[:num_tasks]

    def _load_longbench_tasks(self, count: int, difficulty: Optional[str] = None) -> List[BenchmarkTask]:
        """
        Internal: Load tasks looking like LongBench.
        In a real deployment, this would load from HF datasets or local JSONL.
        Here we mock if file not found, or try to load if implemented.
        """
        # For prototype/robust validator construction, we focus on the structure.
        # Ideally we delegate to a specialized loader file if it exists/is complex.
        
        # Simulating loading/creation for now to ensure Validator works.
        loaded_tasks = []
        
        # Example categories
        categories = self.enabled_datasets
        
        for _ in range(count):
            cat = random.choice(categories)
            
            # Simulate context length based on difficulty or random
            if difficulty == 'extreme':
                ctx_len = random.randint(100000, 2000000)
            elif difficulty == 'hard':
                ctx_len = random.randint(32000, 100000)
            else:
                ctx_len = random.randint(2000, 30000)
                
            # Create a dummy task for testing the flow
            # In production, replace this with `self.real_loader.get_sample()`
            task = BenchmarkTask(
                task_id=f"{cat}_{random.randint(1000, 9999)}",
                dataset_name=cat,
                task_type="qa" if "qa" in cat else "summarization",
                context=f"This is a simulated context of length {ctx_len}. " * (ctx_len // 10),
                prompt=f"What is the length of this context? (Simulated {cat})",
                expected_output=f"{ctx_len}",
                context_length=ctx_len,
                difficulty_level=difficulty or "medium",
                evaluation_metrics=["f1", "rouge"] if "qa" not in cat else ["f1"],
                source="longbench_simulated"
            )
            loaded_tasks.append(task)
            
        return loaded_tasks

    def _generate_synthetic_tasks(self, count: int, difficulty: Optional[str]) -> List[BenchmarkTask]:
        """Generate needle-in-haystack or other synthetic checks."""
        tasks = []
        for _ in range(count):
            tasks.append(BenchmarkTask(
                task_id=f"syn_{random.randint(1000,9999)}",
                dataset_name="synthetic_needle",
                task_type="retrieval",
                context="Haystack " * 1000 + "Needle" + " Haystack" * 1000,
                prompt="Find the needle.",
                expected_output="Needle",
                context_length=2000,
                difficulty_level="easy",
                evaluation_metrics=["exact_match"],
                source="synthetic"
            ))
        return tasks
