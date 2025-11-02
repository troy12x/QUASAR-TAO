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


class GovReportLoader:
    """
    Loader for GovReport dataset - Government report summarization tasks.
    
    GovReport contains long government reports paired with expert-written summaries.
    This dataset is particularly valuable for testing infinite context capabilities
    as government reports can be extremely long (often 10K-50K+ tokens) and require
    understanding of complex policy information across the entire document.
    
    The task tests:
    - Long document comprehension
    - Information synthesis across long contexts
    - Abstractive summarization capabilities
    - Coherence maintenance over extended sequences
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GovReport loader.
        
        Args:
            config: Configuration dictionary with GovReport settings
        """
        self.config = config
        self.data_path = config.get('data_path', 'data/govreport')
        self.min_context_length = config.get('min_context_length', 5000)
        self.max_context_length = config.get('max_context_length', 100000)
        self.summary_types = config.get('summary_types', ['full', 'executive', 'key_points'])
        
        # Cache for loaded data
        self._data_cache = {}
        
        bt.logging.info(f"GovReportLoader initialized with context range {self.min_context_length}-{self.max_context_length}")
    
    def is_available(self) -> bool:
        """Check if GovReport data is available"""
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_path):
                bt.logging.warning(f"GovReport data path not found: {self.data_path}")
                return False
            
            # Check for data files
            data_files = [
                'govreport_train.jsonl',
                'govreport_test.jsonl',
                'govreport_validation.jsonl'
            ]
            
            for filename in data_files:
                filepath = os.path.join(self.data_path, filename)
                if os.path.exists(filepath):
                    return True
            
            bt.logging.warning("No GovReport data files found")
            return False
            
        except Exception as e:
            bt.logging.error(f"Error checking GovReport availability: {e}")
            return False
    
    def load_tasks(self, 
                   num_tasks: int,
                   context_length_range: Optional[tuple] = None) -> List[BenchmarkTask]:
        """
        Load GovReport summarization tasks.
        
        Args:
            num_tasks: Number of tasks to load
            context_length_range: Optional tuple of (min_length, max_length) for filtering
            
        Returns:
            List of BenchmarkTask objects
        """
        try:
            # Load data from available files
            all_data = []
            data_files = ['govreport_test.jsonl', 'govreport_validation.jsonl', 'govreport_train.jsonl']
            
            for filename in data_files:
                if filename not in self._data_cache:
                    self._data_cache[filename] = self._load_raw_data(filename)
                
                file_data = self._data_cache[filename]
                if file_data:
                    all_data.extend(file_data)
            
            if not all_data:
                bt.logging.warning("No GovReport data available")
                return []
            
            # Filter by context length if specified
            if context_length_range:
                min_len, max_len = context_length_range
                filtered_data = []
                for item in all_data:
                    context_len = len(item.get('report', '').split()) * 1.3  # Approximate tokens
                    if min_len <= context_len <= max_len:
                        filtered_data.append(item)
                all_data = filtered_data
            
            # Sample requested number of tasks
            if len(all_data) > num_tasks:
                sampled_data = random.sample(all_data, num_tasks)
            else:
                sampled_data = all_data
            
            tasks = []
            
            # Convert to benchmark tasks
            for i, item in enumerate(sampled_data):
                try:
                    # Create multiple task variants for different summary types
                    for summary_type in self.summary_types:
                        task = self._convert_to_benchmark_task(item, i, summary_type)
                        if task:
                            tasks.append(task)
                            
                            # Limit total tasks
                            if len(tasks) >= num_tasks:
                                break
                    
                    if len(tasks) >= num_tasks:
                        break
                        
                except Exception as e:
                    bt.logging.error(f"Error converting GovReport item to task: {e}")
            
            bt.logging.info(f"Loaded {len(tasks)} GovReport tasks")
            return tasks[:num_tasks]
            
        except Exception as e:
            bt.logging.error(f"Error loading GovReport tasks: {e}")
            return []
    
    def _load_raw_data(self, filename: str) -> List[Dict[str, Any]]:
        """Load raw GovReport data from JSONL file"""
        
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            bt.logging.warning(f"GovReport file not found: {filepath}")
            return []
        
        try:
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError as e:
                            bt.logging.warning(f"Error parsing line {line_num} in {filepath}: {e}")
            
            bt.logging.info(f"Loaded {len(data)} items from {filepath}")
            return data
            
        except Exception as e:
            bt.logging.error(f"Error loading {filepath}: {e}")
            return []
    
    def _convert_to_benchmark_task(self, 
                                 item: Dict[str, Any], 
                                 item_id: int,
                                 summary_type: str) -> Optional[BenchmarkTask]:
        """Convert GovReport item to BenchmarkTask"""
        
        try:
            # Extract report content and summary
            report_text = item.get('report', '')
            summary = item.get('summary', '')
            title = item.get('title', f'Government Report {item_id}')
            
            if not report_text or not summary:
                bt.logging.warning(f"Missing report or summary in GovReport item {item_id}")
                return None
            
            # Check context length constraints
            context_length = len(report_text.split()) * 1.3  # Approximate tokens
            
            if context_length < self.min_context_length or context_length > self.max_context_length:
                return None
            
            # Create different prompts based on summary type
            if summary_type == 'full':
                prompt = f"Please provide a comprehensive summary of this government report titled '{title}'."
                expected_output = summary
                metrics = ["rouge_l", "semantic_similarity", "completeness"]
                
            elif summary_type == 'executive':
                prompt = f"Provide an executive summary highlighting the key findings and recommendations from this government report titled '{title}'."
                # Create shorter version of summary for executive summary
                summary_sentences = summary.split('.')
                expected_output = '. '.join(summary_sentences[:len(summary_sentences)//2]) + '.'
                metrics = ["rouge_l", "semantic_similarity", "conciseness"]
                
            elif summary_type == 'key_points':
                prompt = f"Extract and list the key points and main findings from this government report titled '{title}'."
                # Create bullet-point version
                summary_sentences = summary.split('.')
                key_points = []
                for i, sentence in enumerate(summary_sentences[:5]):  # Top 5 points
                    if sentence.strip():
                        key_points.append(f"• {sentence.strip()}")
                expected_output = '\n'.join(key_points)
                metrics = ["key_point_extraction", "semantic_similarity"]
                
            else:
                # Default to full summary
                prompt = f"Summarize this government report titled '{title}'."
                expected_output = summary
                metrics = ["rouge_l", "semantic_similarity"]
            
            # Determine difficulty based on context length and complexity
            if context_length > 50000:
                difficulty = "extreme"
            elif context_length > 25000:
                difficulty = "hard"
            elif context_length > 10000:
                difficulty = "medium"
            else:
                difficulty = "easy"
            
            return BenchmarkTask(
                task_id=f"govreport_{summary_type}_{item_id}",
                task_type="govreport",
                dataset_name=f"govreport_{summary_type}",
                context=report_text,
                prompt=prompt,
                expected_output=expected_output,
                evaluation_metrics=metrics,
                difficulty_level=difficulty,
                metadata={
                    "title": title,
                    "summary_type": summary_type,
                    "original_summary_length": len(summary.split()),
                    "report_length": len(report_text.split()),
                    "source": item.get('source', 'unknown')
                }
            )
            
        except Exception as e:
            bt.logging.error(f"Error converting GovReport item: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about GovReport dataset"""
        
        stats = {
            "loader_type": "govreport",
            "data_path": self.data_path,
            "summary_types": self.summary_types,
            "context_length_range": (self.min_context_length, self.max_context_length),
            "file_stats": {}
        }
        
        # Check each data file
        data_files = ['govreport_train.jsonl', 'govreport_test.jsonl', 'govreport_validation.jsonl']
        
        total_items = 0
        context_lengths = []
        
        for filename in data_files:
            filepath = os.path.join(self.data_path, filename)
            file_stats = {
                "exists": os.path.exists(filepath),
                "item_count": 0
            }
            
            if file_stats["exists"]:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                file_stats["item_count"] += 1
                                
                                # Sample some items to get context length distribution
                                if file_stats["item_count"] % 100 == 0:  # Sample every 100th item
                                    try:
                                        item = json.loads(line)
                                        report_text = item.get('report', '')
                                        if report_text:
                                            context_len = len(report_text.split()) * 1.3
                                            context_lengths.append(context_len)
                                    except:
                                        pass
                    
                    total_items += file_stats["item_count"]
                    
                except Exception as e:
                    file_stats["error"] = str(e)
            
            stats["file_stats"][filename] = file_stats
        
        stats["total_items"] = total_items
        
        # Add context length statistics
        if context_lengths:
            stats["context_length_stats"] = {
                "min": min(context_lengths),
                "max": max(context_lengths),
                "avg": sum(context_lengths) / len(context_lengths),
                "sample_size": len(context_lengths)
            }
        
        return stats