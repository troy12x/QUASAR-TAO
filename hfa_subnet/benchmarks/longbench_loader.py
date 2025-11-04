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

import json
import random
import os
from typing import List, Dict, Any, Optional
from .benchmark_task import BenchmarkTask

import bittensor as bt


class LongBenchLoader:
    """
    Loader for LongBench dataset - a comprehensive long-context evaluation suite.
    
    LongBench includes multiple tasks for evaluating long-context understanding:
    - Single-document QA
    - Multi-document QA  
    - Summarization
    - Few-shot learning
    - Synthetic tasks
    - Code completion
    
    This loader integrates LongBench tasks with the HFA subnet evaluation system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LongBench loader.
        
        Args:
            config: Configuration dictionary with LongBench settings
        """
        self.config = config
        self.data_path = config.get('data_path', 'data/longbench')
        self.enabled_tasks = config.get('enabled_tasks', [
            'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', 
            'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa'
        ])
        
        # Task metadata
        self.task_metadata = {
            'narrativeqa': {
                'description': 'Reading comprehension on full-length books',
                'avg_context_length': 18000,
                'difficulty': 'hard',
                'metrics': ['f1', 'exact_match']
            },
            'qasper': {
                'description': 'Question answering on scientific papers',
                'avg_context_length': 5000,
                'difficulty': 'medium',
                'metrics': ['f1', 'exact_match']
            },
            'multifieldqa_en': {
                'description': 'Multi-field question answering',
                'avg_context_length': 4000,
                'difficulty': 'medium',
                'metrics': ['f1', 'exact_match']
            },
            'hotpotqa': {
                'description': 'Multi-hop reasoning questions',
                'avg_context_length': 6000,
                'difficulty': 'hard',
                'metrics': ['f1', 'exact_match']
            },
            'gov_report': {
                'description': 'Government report summarization',
                'avg_context_length': 8000,
                'difficulty': 'medium',
                'metrics': ['rouge_l', 'semantic_similarity']
            },
            'qmsum': {
                'description': 'Query-based meeting summarization',
                'avg_context_length': 10000,
                'difficulty': 'hard',
                'metrics': ['rouge_l', 'semantic_similarity']
            },
            'multi_news': {
                'description': 'Multi-document news summarization',
                'avg_context_length': 12000,
                'difficulty': 'medium',
                'metrics': ['rouge_l', 'semantic_similarity']
            },
            'trec': {
                'description': 'Question classification',
                'avg_context_length': 2000,
                'difficulty': 'easy',
                'metrics': ['accuracy']
            },
            'triviaqa': {
                'description': 'Trivia question answering',
                'avg_context_length': 3000,
                'difficulty': 'easy',
                'metrics': ['exact_match', 'f1']
            }
        }
        
        # Cache for loaded data
        self._data_cache = {}
        
        bt.logging.info(f"LongBenchLoader initialized with {len(self.enabled_tasks)} enabled tasks")
    
    def is_available(self) -> bool:
        """Check if LongBench data is available"""
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_path):
                bt.logging.warning(f"LongBench data path not found: {self.data_path}")
                return False
            
            # Check if at least one task file exists
            for task in self.enabled_tasks:
                task_file = os.path.join(self.data_path, f"{task}.jsonl")
                if os.path.exists(task_file):
                    return True
            
            bt.logging.warning("No LongBench task files found")
            return False
            
        except Exception as e:
            bt.logging.error(f"Error checking LongBench availability: {e}")
            return False
    
    def load_tasks(self, 
                   num_tasks: int,
                   context_length_range: Optional[tuple] = None) -> List[BenchmarkTask]:
        """
        Load LongBench tasks.
        
        Args:
            num_tasks: Number of tasks to load
            context_length_range: Optional tuple of (min_length, max_length) for filtering
            
        Returns:
            List of BenchmarkTask objects
        """
        tasks = []
        tasks_per_dataset = max(1, num_tasks // len(self.enabled_tasks))
        
        for task_name in self.enabled_tasks:
            try:
                dataset_tasks = self._load_task_dataset(task_name, tasks_per_dataset, context_length_range)
                tasks.extend(dataset_tasks)
                
                if len(dataset_tasks) > 0:
                    bt.logging.info(f"Loaded {len(dataset_tasks)} tasks from LongBench {task_name}")
                
            except Exception as e:
                bt.logging.error(f"Error loading LongBench task {task_name}: {e}")
                # Continue with other tasks
        
        # Randomize and limit to requested number
        random.shuffle(tasks)
        return tasks[:num_tasks]
    
    def _load_task_dataset(self, 
                          task_name: str, 
                          num_tasks: int,
                          context_length_range: Optional[tuple] = None) -> List[BenchmarkTask]:
        """Load tasks from a specific LongBench dataset"""
        
        # Check cache first
        if task_name in self._data_cache:
            raw_data = self._data_cache[task_name]
        else:
            raw_data = self._load_raw_data(task_name)
            if raw_data:
                self._data_cache[task_name] = raw_data
        
        if not raw_data:
            return []
        
        tasks = []
        task_metadata = self.task_metadata.get(task_name, {})
        
        # Filter by context length if specified
        if context_length_range:
            min_len, max_len = context_length_range
            filtered_data = []
            for item in raw_data:
                context_len = len(item.get('context', '').split()) * 1.3  # Approximate tokens
                if min_len <= context_len <= max_len:
                    filtered_data.append(item)
            raw_data = filtered_data
        
        # Sample requested number of tasks
        if len(raw_data) > num_tasks:
            raw_data = random.sample(raw_data, num_tasks)
        
        # Convert to BenchmarkTask objects
        for i, item in enumerate(raw_data):
            try:
                task = self._convert_to_benchmark_task(item, task_name, i, task_metadata)
                if task:
                    tasks.append(task)
            except Exception as e:
                bt.logging.error(f"Error converting LongBench item to task: {e}")
        
        return tasks
    
    def _load_raw_data(self, task_name: str) -> List[Dict[str, Any]]:
        """Load raw data from LongBench task file"""
        
        task_file = os.path.join(self.data_path, f"{task_name}.jsonl")
        
        if not os.path.exists(task_file):
            bt.logging.warning(f"LongBench task file not found: {task_file}")
            return []
        
        try:
            data = []
            with open(task_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        data.append(item)
            
            bt.logging.info(f"Loaded {len(data)} items from {task_file}")
            return data
            
        except Exception as e:
            bt.logging.error(f"Error loading {task_file}: {e}")
            return []
    
    def _convert_to_benchmark_task(self, 
                                 item: Dict[str, Any], 
                                 task_name: str, 
                                 item_id: int,
                                 task_metadata: Dict[str, Any]) -> Optional[BenchmarkTask]:
        """Convert raw LongBench item to BenchmarkTask"""
        
        try:
            # Extract fields based on LongBench format
            context = item.get('context', '')
            input_text = item.get('input', '')
            question = item.get('question', input_text)
            answers = item.get('answers', [])
            
            # Combine context and question for prompt
            if context and question:
                full_context = context
                prompt = question
            elif input_text:
                # Some tasks have input as the main content
                full_context = input_text
                prompt = "Please provide a response based on the given content."
            else:
                bt.logging.warning(f"No valid content found in LongBench item for {task_name}")
                return None
            
            # Get expected output
            expected_output = None
            if answers and len(answers) > 0:
                expected_output = answers[0] if isinstance(answers[0], str) else str(answers[0])
            elif 'output' in item:
                expected_output = item['output']
            
            # Determine difficulty based on context length and task type
            context_length = len(full_context.split()) * 1.3  # Approximate tokens
            if context_length > 32000:
                difficulty = "extreme"
            elif context_length > 16000:
                difficulty = "hard"
            elif context_length > 8000:
                difficulty = "medium"
            else:
                difficulty = "easy"
            
            # Override with task-specific difficulty if available
            if 'difficulty' in task_metadata:
                difficulty = task_metadata['difficulty']
            
            # Get evaluation metrics
            metrics = task_metadata.get('metrics', ['accuracy', 'coherence'])
            
            return BenchmarkTask(
                task_id=f"longbench_{task_name}_{item_id}",
                task_type="longbench",
                dataset_name=task_name,
                context=full_context,
                prompt=prompt,
                expected_output=expected_output,
                evaluation_metrics=metrics,
                difficulty_level=difficulty,
                metadata={
                    "original_item": item,
                    "task_description": task_metadata.get('description', ''),
                    "avg_context_length": task_metadata.get('avg_context_length', 0)
                }
            )
            
        except Exception as e:
            bt.logging.error(f"Error converting LongBench item: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about LongBench datasets"""
        
        stats = {
            "loader_type": "longbench",
            "enabled_tasks": self.enabled_tasks,
            "total_tasks": len(self.enabled_tasks),
            "data_path": self.data_path,
            "task_details": {}
        }
        
        # Get details for each task
        for task_name in self.enabled_tasks:
            task_file = os.path.join(self.data_path, f"{task_name}.jsonl")
            task_stats = {
                "file_exists": os.path.exists(task_file),
                "metadata": self.task_metadata.get(task_name, {})
            }
            
            # Count items if file exists
            if task_stats["file_exists"]:
                try:
                    with open(task_file, 'r', encoding='utf-8') as f:
                        item_count = sum(1 for line in f if line.strip())
                    task_stats["item_count"] = item_count
                except Exception as e:
                    task_stats["error"] = str(e)
            
            stats["task_details"][task_name] = task_stats
        
        return stats