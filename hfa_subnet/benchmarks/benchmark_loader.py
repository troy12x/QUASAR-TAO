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

import random
import logging
from typing import List, Dict, Any, Optional
from .benchmark_task import BenchmarkTask
from .longbench_loader import LongBenchLoader
from .hotpotqa_loader import HotpotQALoader

import bittensor as bt


class BenchmarkLoader:
    """
    LongBench-focused benchmark loader for QUASAR-TAO subnet.
    
    This class provides comprehensive long-context evaluation using LongBench,
    the standardized benchmark for evaluating language models on realistic
    long-context tasks across multiple domains:
    
    - Multi-document QA (HotpotQA, 2WikiMultihopQA, MuSiQue, DuReader)
    - Single-document QA (MultiFieldQA, NarrativeQA, Qasper)
    - Summarization (GovReport, QMSum, MultiNews, VCSUM)
    - Few-shot Learning (TriviaQA, SAMSum, TREC, LSHT)
    - Synthetic Tasks (PassageRetrieval, PassageCount)
    - Code Understanding (LCC, RepoBench-P)
    
    Context lengths: 8k-128k+ words
    Evaluation metrics: F1, ROUGE-L, Accuracy, Edit Similarity
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LongBench-only benchmark loader.
        
        Args:
            config: Configuration dictionary with benchmark settings
        """
        self.config = config or {}
        
        # Initialize LongBench loader (primary benchmark)
        self.longbench_loader = LongBenchLoader(self.config.get('longbench', {}))
        
        # Benchmark availability tracking
        self.available_benchmarks = self._check_benchmark_availability()
        
        # Task generation settings for LongBench
        self.max_tasks_per_benchmark = self.config.get('max_tasks_per_benchmark', 5)
        self.context_length_ranges = self.config.get('context_length_ranges', [
            (8000, 16000),    # Short long-context
            (16000, 32000),   # Medium long-context  
            (32000, 64000),   # Long context
            (64000, 128000)   # Ultra-long context
        ])
        
        bt.logging.info(f"QUASAR-TAO LongBench Loader initialized - Experimental subnet for long-context evaluation")
    
    def _check_benchmark_availability(self) -> List[str]:
        """Check which benchmarks are available for loading"""
        available = []
        
        # Check LongBench loader (primary benchmark for QUASAR-TAO)
        loaders = {
            'longbench': self.longbench_loader,
        }
        
        for name, loader in loaders.items():
            try:
                if loader.is_available():
                    available.append(name)
                    bt.logging.info(f" LongBench benchmark is available")
                else:
                    bt.logging.warning(f" LongBench benchmark is not available")
            except Exception as e:
                bt.logging.error(f"Error checking availability of {name}: {e}")
        
        return available
    
    def load_benchmark_tasks(self, 
                           benchmark_types: Optional[List[str]] = None,
                           num_tasks: int = 10,
                           context_length_range: Optional[tuple] = None) -> List[BenchmarkTask]:
        """
        Load benchmark tasks from specified benchmark types.
        
        Args:
            benchmark_types: List of benchmark types to load from. If None, uses all available.
            num_tasks: Total number of tasks to load
            context_length_range: Tuple of (min_length, max_length) for context filtering
            
        Returns:
            List of BenchmarkTask objects
        """
        if benchmark_types is None:
            benchmark_types = self.available_benchmarks
        
        # Filter to only available benchmarks
        benchmark_types = [bt for bt in benchmark_types if bt in self.available_benchmarks]
        
        if not benchmark_types:
            bt.logging.warning("No available benchmarks found, falling back to synthetic tasks")
            return self._generate_synthetic_fallback_tasks(num_tasks, context_length_range)
        
        tasks = []
        tasks_per_benchmark = max(1, num_tasks // len(benchmark_types))
        
        for benchmark_type in benchmark_types:
            try:
                benchmark_tasks = self._load_from_benchmark(
                    benchmark_type, 
                    tasks_per_benchmark,
                    context_length_range
                )
                tasks.extend(benchmark_tasks)
                bt.logging.info(f"Loaded {len(benchmark_tasks)} tasks from {benchmark_type}")
                
            except Exception as e:
                bt.logging.error(f"Error loading from {benchmark_type}: {e}")
                # Continue with other benchmarks
        
        # If we don't have enough tasks, fill with synthetic ones
        if len(tasks) < num_tasks:
            remaining = num_tasks - len(tasks)
            synthetic_tasks = self._generate_synthetic_fallback_tasks(remaining, context_length_range)
            tasks.extend(synthetic_tasks)
            bt.logging.info(f"Added {len(synthetic_tasks)} synthetic fallback tasks")
        
        # Randomize task order
        random.shuffle(tasks)
        return tasks[:num_tasks]
    
    def _load_from_benchmark(self, 
                           benchmark_type: str, 
                           num_tasks: int,
                           context_length_range: Optional[tuple] = None) -> List[BenchmarkTask]:
        """Load tasks from a specific benchmark type"""
        
        if benchmark_type == 'longbench':
            return self.longbench_loader.load_tasks(num_tasks, context_length_range)
        else:
            raise ValueError(f"QUASAR-TAO only supports LongBench. Got: {benchmark_type}")
    
    def _generate_synthetic_fallback_tasks(self, 
                                         num_tasks: int,
                                         context_length_range: Optional[tuple] = None) -> List[BenchmarkTask]:
        """Generate synthetic tasks when real benchmarks are unavailable"""
        
        tasks = []
        
        for i in range(num_tasks):
            # Select random context length range if not specified
            if context_length_range is None:
                min_len, max_len = random.choice(self.context_length_ranges)
            else:
                min_len, max_len = context_length_range
            
            target_length = random.randint(min_len, max_len)
            
            # Generate different types of synthetic tasks
            task_types = ['memory_retention', 'pattern_recognition', 'summarization', 'qa']
            task_type = random.choice(task_types)
            
            if task_type == 'memory_retention':
                task = self._generate_memory_task(target_length, i)
            elif task_type == 'pattern_recognition':
                task = self._generate_pattern_task(target_length, i)
            elif task_type == 'summarization':
                task = self._generate_summarization_task(target_length, i)
            else:  # qa
                task = self._generate_qa_task(target_length, i)
            
            tasks.append(task)
        
        return tasks
    
    def _generate_memory_task(self, target_length: int, task_id: int) -> BenchmarkTask:
        """Generate synthetic memory retention task"""
        
        # Create context with embedded information
        words = []
        memory_info = f"MEMORY_ANCHOR_{task_id}_{random.randint(1000, 9999)}"
        anchor_position = random.randint(100, min(target_length - 100, 1000))
        
        for i in range(target_length):
            if i == anchor_position:
                words.append(memory_info)
            else:
                words.append(random.choice([
                    "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
                    "he", "was", "for", "on", "are", "as", "with", "his", "they", "at"
                ]))
        
        context = " ".join(words)
        prompt = f"What specific information was stored at position {anchor_position}?"
        
        return BenchmarkTask(
            task_id=f"synthetic_memory_{task_id}",
            task_type="synthetic",
            dataset_name="memory_retention",
            context=context,
            prompt=prompt,
            expected_output=memory_info,
            evaluation_metrics=["exact_match", "memory_retention"],
            difficulty_level="medium" if target_length < 32000 else "hard",
            metadata={"anchor_position": anchor_position, "memory_info": memory_info}
        )
    
    def _generate_pattern_task(self, target_length: int, task_id: int) -> BenchmarkTask:
        """Generate synthetic pattern recognition task"""
        
        patterns = {
            "fibonacci": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
            "prime": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            "even": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        }
        
        pattern_type = random.choice(list(patterns.keys()))
        pattern_sequence = patterns[pattern_type]
        
        # Embed pattern in context
        filler_words = ["word", "text", "data", "info"] * (target_length // 4)
        context_words = filler_words[:target_length - len(pattern_sequence)]
        
        # Insert pattern at random position
        insert_pos = random.randint(50, len(context_words) - 50)
        context_words[insert_pos:insert_pos] = [str(x) for x in pattern_sequence]
        
        context = " ".join(context_words[:target_length])
        prompt = f"Identify the {pattern_type} sequence in the text."
        expected = " ".join([str(x) for x in pattern_sequence])
        
        return BenchmarkTask(
            task_id=f"synthetic_pattern_{task_id}",
            task_type="synthetic", 
            dataset_name="pattern_recognition",
            context=context,
            prompt=prompt,
            expected_output=expected,
            evaluation_metrics=["pattern_accuracy", "coherence"],
            difficulty_level="medium",
            metadata={"pattern_type": pattern_type, "pattern_sequence": pattern_sequence}
        )
    
    def _generate_summarization_task(self, target_length: int, task_id: int) -> BenchmarkTask:
        """Generate synthetic summarization task"""
        
        # Create a coherent narrative
        topics = ["technology", "science", "history", "economics", "environment"]
        topic = random.choice(topics)
        
        base_content = f"This document discusses {topic} and its implications. "
        content_parts = [base_content] * (target_length // len(base_content.split()))
        context = " ".join(content_parts[:target_length])
        
        prompt = f"Provide a comprehensive summary of this {topic} document."
        expected = f"This document provides an overview of {topic} and discusses its various implications and aspects."
        
        return BenchmarkTask(
            task_id=f"synthetic_summary_{task_id}",
            task_type="synthetic",
            dataset_name="summarization", 
            context=context,
            prompt=prompt,
            expected_output=expected,
            evaluation_metrics=["semantic_similarity", "coherence", "completeness"],
            difficulty_level="medium" if target_length < 16000 else "hard",
            metadata={"topic": topic}
        )
    
    def _generate_qa_task(self, target_length: int, task_id: int) -> BenchmarkTask:
        """Generate synthetic question answering task"""
        
        # Embed answer in context
        answer = f"The answer is {random.randint(100, 999)}"
        question_topic = random.choice(["number", "value", "result", "outcome"])
        
        # Create context with embedded answer
        filler_text = "This is background information. " * (target_length // 6)
        answer_position = random.randint(target_length // 4, 3 * target_length // 4)
        
        context_parts = filler_text.split()
        if answer_position < len(context_parts):
            context_parts[answer_position] = answer
        
        context = " ".join(context_parts[:target_length])
        prompt = f"What {question_topic} is mentioned in the document?"
        
        return BenchmarkTask(
            task_id=f"synthetic_qa_{task_id}",
            task_type="synthetic",
            dataset_name="question_answering",
            context=context,
            prompt=prompt,
            expected_output=answer,
            evaluation_metrics=["exact_match", "semantic_similarity"],
            difficulty_level="easy" if target_length < 8000 else "medium",
            metadata={"answer": answer, "answer_position": answer_position}
        )
    
    def get_benchmark_stats(self) -> Dict[str, Any]:
        """Get statistics about available benchmarks"""
        
        stats = {
            "available_benchmarks": self.available_benchmarks,
            "total_available": len(self.available_benchmarks),
            "benchmark_details": {}
        }
        
        # Get details from each available benchmark
        loaders = {
            'longbench': self.longbench_loader,
            'hotpotqa': self.hotpotqa_loader,
            'govreport': self.govreport_loader, 
            'needle_haystack': self.needle_haystack_loader
        }
        
        for name in self.available_benchmarks:
            if name in loaders:
                try:
                    loader_stats = loaders[name].get_stats()
                    stats["benchmark_details"][name] = loader_stats
                except Exception as e:
                    bt.logging.error(f"Error getting stats for {name}: {e}")
                    stats["benchmark_details"][name] = {"error": str(e)}
        
        return stats