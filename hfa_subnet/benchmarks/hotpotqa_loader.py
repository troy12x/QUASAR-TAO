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


class HotpotQALoader:
    """
    Loader for HotpotQA dataset with distractor documents.
    
    HotpotQA is a dataset for multi-hop reasoning over multiple documents.
    The distractor version includes irrelevant documents to test the model's
    ability to identify and use only relevant information for answering questions.
    
    This is particularly valuable for testing infinite context capabilities
    as models must maintain focus on relevant information across long contexts
    with many distractors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HotpotQA loader.
        
        Args:
            config: Configuration dictionary with HotpotQA settings
        """
        self.config = config
        self.data_path = config.get('data_path', 'data/hotpotqa')
        self.use_distractors = config.get('use_distractors', True)
        self.max_distractors = config.get('max_distractors', 8)
        self.min_context_length = config.get('min_context_length', 2000)
        self.max_context_length = config.get('max_context_length', 50000)
        
        # Cache for loaded data
        self._data_cache = {}
        self._distractor_cache = {}
        
        bt.logging.info(f"HotpotQALoader initialized with distractors={'enabled' if self.use_distractors else 'disabled'}")
    
    def is_available(self) -> bool:
        """Check if HotpotQA data is available"""
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_path):
                bt.logging.warning(f"HotpotQA data path not found: {self.data_path}")
                return False
            
            # Check for main data files
            required_files = ['hotpot_dev_distractor_v1.json']
            if self.use_distractors:
                required_files.append('hotpot_train_v1.1.json')  # For additional distractors
            
            for filename in required_files:
                filepath = os.path.join(self.data_path, filename)
                if os.path.exists(filepath):
                    return True
            
            bt.logging.warning("No HotpotQA data files found")
            return False
            
        except Exception as e:
            bt.logging.error(f"Error checking HotpotQA availability: {e}")
            return False
    
    def load_tasks(self, 
                   num_tasks: int,
                   context_length_range: Optional[tuple] = None) -> List[BenchmarkTask]:
        """
        Load HotpotQA tasks with distractors.
        
        Args:
            num_tasks: Number of tasks to load
            context_length_range: Optional tuple of (min_length, max_length) for filtering
            
        Returns:
            List of BenchmarkTask objects
        """
        try:
            # Load main dataset
            if 'main' not in self._data_cache:
                self._data_cache['main'] = self._load_raw_data('hotpot_dev_distractor_v1.json')
            
            raw_data = self._data_cache['main']
            if not raw_data:
                bt.logging.warning("No HotpotQA data available")
                return []
            
            # Load additional distractors if needed
            if self.use_distractors and 'distractors' not in self._distractor_cache:
                self._distractor_cache['distractors'] = self._load_distractor_pool()
            
            tasks = []
            
            # Sample from available data
            if len(raw_data) > num_tasks:
                sampled_data = random.sample(raw_data, num_tasks)
            else:
                sampled_data = raw_data
            
            # Convert to benchmark tasks
            for i, item in enumerate(sampled_data):
                try:
                    task = self._convert_to_benchmark_task(item, i, context_length_range)
                    if task:
                        tasks.append(task)
                except Exception as e:
                    bt.logging.error(f"Error converting HotpotQA item to task: {e}")
            
            bt.logging.info(f"Loaded {len(tasks)} HotpotQA tasks")
            return tasks
            
        except Exception as e:
            bt.logging.error(f"Error loading HotpotQA tasks: {e}")
            return []
    
    def _load_raw_data(self, filename: str) -> List[Dict[str, Any]]:
        """Load raw HotpotQA data from file"""
        
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            bt.logging.warning(f"HotpotQA file not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            bt.logging.info(f"Loaded {len(data)} items from {filepath}")
            return data
            
        except Exception as e:
            bt.logging.error(f"Error loading {filepath}: {e}")
            return []
    
    def _load_distractor_pool(self) -> List[Dict[str, Any]]:
        """Load additional documents to use as distractors"""
        
        distractor_files = [
            'hotpot_train_v1.1.json',
            'hotpot_dev_fullwiki_v1.json'
        ]
        
        all_distractors = []
        
        for filename in distractor_files:
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract context documents as potential distractors
                    for item in data:
                        if 'context' in item:
                            for doc_title, doc_sentences in item['context']:
                                distractor_doc = {
                                    'title': doc_title,
                                    'content': ' '.join(doc_sentences)
                                }
                                all_distractors.append(distractor_doc)
                    
                    bt.logging.info(f"Loaded {len(data)} items from {filepath} for distractors")
                    
                except Exception as e:
                    bt.logging.error(f"Error loading distractor file {filepath}: {e}")
        
        # Shuffle and limit distractor pool
        random.shuffle(all_distractors)
        return all_distractors[:10000]  # Keep reasonable pool size
    
    def _convert_to_benchmark_task(self, 
                                 item: Dict[str, Any], 
                                 item_id: int,
                                 context_length_range: Optional[tuple] = None) -> Optional[BenchmarkTask]:
        """Convert HotpotQA item to BenchmarkTask with distractors"""
        
        try:
            # Extract basic information
            question = item.get('question', '')
            answer = item.get('answer', '')
            question_type = item.get('type', 'bridge')  # bridge or comparison
            level = item.get('level', 'medium')
            
            if not question or not answer:
                bt.logging.warning(f"Missing question or answer in HotpotQA item {item_id}")
                return None
            
            # Build context from supporting documents
            context_parts = []
            supporting_facts = item.get('supporting_facts', [])
            
            # Add original context documents
            if 'context' in item:
                for doc_title, doc_sentences in item['context']:
                    doc_content = f"Document: {doc_title}\n" + ' '.join(doc_sentences)
                    context_parts.append(doc_content)
            
            # Add distractor documents if enabled
            if self.use_distractors and 'distractors' in self._distractor_cache:
                num_distractors = random.randint(2, self.max_distractors)
                available_distractors = self._distractor_cache['distractors']
                
                if len(available_distractors) >= num_distractors:
                    selected_distractors = random.sample(available_distractors, num_distractors)
                    
                    for distractor in selected_distractors:
                        distractor_content = f"Document: {distractor['title']}\n{distractor['content']}"
                        context_parts.append(distractor_content)
            
            # Shuffle context parts to make task more challenging
            random.shuffle(context_parts)
            full_context = '\n\n'.join(context_parts)
            
            # Check context length constraints
            context_length = len(full_context.split()) * 1.3  # Approximate tokens
            
            if context_length_range:
                min_len, max_len = context_length_range
                if not (min_len <= context_length <= max_len):
                    return None
            
            # Ensure minimum context length by adding more distractors if needed
            if context_length < self.min_context_length and self.use_distractors:
                additional_distractors_needed = (self.min_context_length - context_length) // 500
                if 'distractors' in self._distractor_cache:
                    available_distractors = self._distractor_cache['distractors']
                    additional_distractors = random.sample(
                        available_distractors, 
                        min(additional_distractors_needed, len(available_distractors))
                    )
                    
                    for distractor in additional_distractors:
                        distractor_content = f"Document: {distractor['title']}\n{distractor['content']}"
                        context_parts.append(distractor_content)
                    
                    full_context = '\n\n'.join(context_parts)
                    context_length = len(full_context.split()) * 1.3
            
            # Limit context length if too long
            if context_length > self.max_context_length:
                # Truncate context while preserving supporting documents
                words = full_context.split()
                target_words = int(self.max_context_length / 1.3)
                if len(words) > target_words:
                    full_context = ' '.join(words[:target_words])
                    context_length = target_words * 1.3
            
            # Determine difficulty based on question type and context length
            if question_type == 'comparison' or context_length > 32000:
                difficulty = "hard"
            elif context_length > 16000:
                difficulty = "medium"
            else:
                difficulty = "easy"
            
            # Create evaluation metrics
            metrics = ["exact_match", "f1", "multi_hop_reasoning"]
            if self.use_distractors:
                metrics.append("distractor_resistance")
            
            return BenchmarkTask(
                task_id=f"hotpotqa_{item_id}",
                task_type="hotpotqa",
                dataset_name="hotpotqa_distractor",
                context=full_context,
                prompt=question,
                expected_output=answer,
                evaluation_metrics=metrics,
                difficulty_level=difficulty,
                metadata={
                    "question_type": question_type,
                    "level": level,
                    "supporting_facts": supporting_facts,
                    "num_distractors": len(context_parts) - len(item.get('context', [])),
                    "original_context_docs": len(item.get('context', [])),
                    "uses_distractors": self.use_distractors
                }
            )
            
        except Exception as e:
            bt.logging.error(f"Error converting HotpotQA item: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about HotpotQA dataset"""
        
        stats = {
            "loader_type": "hotpotqa",
            "data_path": self.data_path,
            "use_distractors": self.use_distractors,
            "max_distractors": self.max_distractors,
            "context_length_range": (self.min_context_length, self.max_context_length)
        }
        
        # Check main data file
        main_file = os.path.join(self.data_path, 'hotpot_dev_distractor_v1.json')
        if os.path.exists(main_file):
            try:
                with open(main_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                stats["main_dataset_size"] = len(data)
                
                # Analyze question types
                question_types = {}
                for item in data:
                    qtype = item.get('type', 'unknown')
                    question_types[qtype] = question_types.get(qtype, 0) + 1
                
                stats["question_types"] = question_types
                
            except Exception as e:
                stats["main_dataset_error"] = str(e)
        else:
            stats["main_dataset_size"] = 0
        
        # Check distractor pool
        if self.use_distractors:
            if 'distractors' in self._distractor_cache:
                stats["distractor_pool_size"] = len(self._distractor_cache['distractors'])
            else:
                stats["distractor_pool_size"] = 0
        
        return stats