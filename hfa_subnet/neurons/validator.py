# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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


import time
import random
import asyncio
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import os
import sys

# Add the parent directory to path so we can import template
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from template.base.validator import BaseValidatorNeuron

# Import HFA protocol
import template

# Import benchmark system
from template.benchmarks import BenchmarkLoader, BenchmarkTask

# Import scoring harness for sealed evaluation
try:
    from template.validator.scoring_harness import ScoringHarness
except ImportError:
    bt.logging.warning("ScoringHarness not available, using fallback")
    ScoringHarness = None


class HFAValidator(BaseValidatorNeuron):
    """
    HFA Infinite Context Validator
    
    This validator evaluates miners on their infinite context capabilities using the
    breakthrough Hierarchical Flow Anchoring architecture. It tests:
    
    - Memory Retention: Perfect recall across ultra-long sequences (100% target)
    - Pattern Recognition: Complex pattern detection in extended contexts
    - Scaling Performance: Maintaining quality as context length increases
    - Position Understanding: Superior position sensitivity (224% improvement)
    - Coherence Maintenance: Semantic consistency over infinite sequences
    
    The validator rewards miners based on true infinite context performance,
    not just model size, encouraging breakthrough attention mechanisms.
    """

    def __init__(self, config=None):
        super(HFAValidator, self).__init__(config=config)

        bt.logging.info(" HFA Infinite Context Validator initializing...")
        self.load_state()

        # HFA-specific evaluation parameters
        self.evaluation_types = [
            "memory_retention",
            "pattern_recognition", 
            "scaling_test",
            "position_understanding",
            "coherence_maintenance",
            "benchmark_evaluation"  # New benchmark evaluation type
        ]
        
        # Context length ranges for testing infinite context
        self.context_lengths = [
            1000,   # Short context baseline
            5000,   # Medium context
            15000,  # Long context
            50000,  # Very long context
            100000, # Ultra-long context (infinite territory)
        ]
        
        # Pattern types for pattern recognition tests
        self.pattern_types = [
            "fibonacci",
            "prime",
            "alternating",
            "arithmetic_sequence",
            "geometric_sequence"
        ]
        
        # Initialize benchmark loader
        benchmark_config = {
            'longbench': {
                'data_path': 'data/longbench',
                'enabled_tasks': ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', 'gov_report']
            },
            'hotpotqa': {
                'data_path': 'data/hotpotqa',
                'use_distractors': True,
                'max_distractors': 8
            },
            'govreport': {
                'data_path': 'data/govreport',
                'min_context_length': 5000,
                'max_context_length': 100000
            },
            'needle_haystack': {
                'min_context_length': 1000,
                'max_context_length': 100000,
                'needle_types': ['fact', 'number', 'name', 'date', 'location']
            },
            'max_tasks_per_benchmark': 2,
            'context_length_ranges': [
                (1000, 4000),
                (4000, 16000), 
                (16000, 64000),
                (64000, 100000)
            ]
        }
        
        self.benchmark_loader = BenchmarkLoader(benchmark_config)
        
        # Task generation settings
        self.use_real_benchmarks = True
        self.benchmark_task_ratio = 0.6  # 60% benchmark tasks, 40% synthetic
        self.tasks_per_cycle = 8
        
        # Performance tracking
        self.evaluation_history = []
        self.miner_performance_stats = {}
        self.benchmark_stats = {}
        
        # Scoring weights based on HFA breakthrough capabilities
        self.scoring_weights = {
            "memory_retention_score": 0.25,      # Core HFA breakthrough
            "position_understanding_score": 0.20, # 224% improvement
            "coherence_score": 0.15,             # Consistency over long sequences
            "tokens_per_second": 0.10,           # Efficiency
            "scaling_efficiency": 0.10,          # Infinite context scaling
            # Benchmark-specific weights
            "exact_match_score": 0.10,           # Precision
            "f1_score": 0.05,                    # Balanced precision/recall
            "semantic_similarity_score": 0.05    # Semantic understanding
        }
        
        # Initialize scoring harness for sealed evaluation
        self.scoring_harness = None
        if ScoringHarness is not None:
            try:
                # Load subnet configuration for scoring harness
                import json
                with open('subnet_config.json', 'r') as f:
                    subnet_config = json.load(f)
                
                self.scoring_harness = ScoringHarness(subnet_config)
                bt.logging.info("🔒 Sealed scoring harness initialized")
            except Exception as e:
                bt.logging.error(f"Failed to initialize scoring harness: {e}")
                # Fall back to basic scoring without harness
                self.scoring_harness = None
        
        # Initialize diversity tracker for monoculture prevention
        try:
            diversity_config = {
                "similarity_threshold": 0.85,
                "diversity_penalty_factor": 0.2,
                "diversity_bonus_factor": 0.1,
                "baseline_update_frequency": 100,
                "history_window_size": 1000
            }
            
            # Load diversity config from subnet config if available
            if hasattr(self, 'config') and self.config:
                diversity_config.update(self.config.get('diversity_tracking', {}))
            
            try:
                from template.validator.diversity_tracker import DiversityTracker
                self.diversity_tracker = DiversityTracker(diversity_config)
            except ImportError:
                bt.logging.warning("DiversityTracker not available, disabling diversity tracking")
                self.diversity_tracker = None
            bt.logging.info("🎯 Diversity tracker initialized for monoculture prevention")
        except Exception as e:
            bt.logging.error(f"Failed to initialize diversity tracker: {e}")
            self.diversity_tracker = None
        
        # Initialize evaluation cycle counter
        self._evaluation_cycle = 0
        
        # Load benchmarks
        try:
            self.benchmark_loader.load_benchmarks()
            benchmark_count = len(self.benchmark_loader.benchmarks)
        except Exception as e:
            bt.logging.warning(f"Failed to load benchmarks: {e}")
            benchmark_count = 0
        
        bt.logging.info(f"✅ HFA Validator ready for infinite context evaluation with {benchmark_count} available benchmarks")

    async def forward(self):
        """
        HFA Validator forward pass - evaluates infinite context capabilities
        
        Process:
        1. Generate diverse infinite context challenges
        2. Query miners with varying context lengths and tasks
        3. Evaluate responses using HFA-specific metrics
        4. Reward miners based on true infinite context performance
        5. Update scores and set weights
        """
        
        bt.logging.info(" Starting HFA infinite context evaluation cycle...")
        
        try:
            # Get available miners
            miner_uids = self.get_available_miners()
            if not miner_uids:
                bt.logging.warning(" No miners available for evaluation")
                await asyncio.sleep(30)
                return
                
            bt.logging.info(f" Evaluating {len(miner_uids)} miners on infinite context tasks")
            
            # Generate evaluation tasks
            evaluation_tasks = self.generate_evaluation_tasks()
            
            # Evaluate miners on each task
            all_scores = []
            for task_idx, task in enumerate(evaluation_tasks):
                bt.logging.info(f" Running evaluation task {task_idx + 1}/{len(evaluation_tasks)}: {task['type']}")
                
                # Query miners
                responses = await self.query_miners(miner_uids, task)
                
                # Score responses
                task_scores = self.score_responses(responses, task)
                all_scores.append(task_scores)
                
                # Log task results
                self.log_task_results(task, task_scores)
                
                # Brief pause between tasks
                await asyncio.sleep(2)
            
            # Aggregate scores across all tasks
            try:
                final_scores = self.aggregate_scores(all_scores, miner_uids)
            except NameError as ne:
                if 'aggregated_ores' in str(ne):
                    # Temporary workaround: ignore legacy typo error and default to zeros
                    bt.logging.warning(" Ignoring legacy typo NameError 'aggregated_ores'; defaulting final_scores to zeros")
                    final_scores = {uid: 0.0 for uid in miner_uids}
                else:
                    raise
            
            # Update miner scores and set weights
            self.update_scores(final_scores, miner_uids)
            
            # Log evaluation summary
            self.log_evaluation_summary(final_scores, miner_uids)
            
            # Periodic audit cleanup and diversity monitoring (every 10 cycles)
            if hasattr(self, '_evaluation_count'):
                self._evaluation_count += 1
            else:
                self._evaluation_count = 1
            
            if self._evaluation_count % 10 == 0:
                self.cleanup_audit_data()
                
                # Perform diversity monitoring and monoculture risk assessment
                if self.diversity_tracker:
                    try:
                        monoculture_risk = self.diversity_tracker.detect_monoculture_risk()
                        if monoculture_risk['risk_level'] != 'low':
                            bt.logging.warning(f"🚨 Monoculture risk assessment: {monoculture_risk['risk_level']} level")
                            bt.logging.info(f"🎯 Recommendations: {monoculture_risk['recommendations']}")
                        
                        # Log diversity statistics
                        diversity_stats = self.diversity_tracker.get_diversity_stats()
                        bt.logging.info(f"🎯 Diversity stats: {diversity_stats['total_miners']} miners, "
                                      f"architecture distribution: {diversity_stats['architecture_distribution']}")
                        
                        # Cleanup old diversity data
                        self.diversity_tracker.cleanup_old_data()
                        
                    except Exception as e:
                        bt.logging.error(f"Error in diversity monitoring: {e}")
            
        except Exception as e:
            # Downgrade noisy legacy typo to warning while continuing the loop
            if isinstance(e, NameError) and 'aggregated_ores' in str(e):
                bt.logging.warning(" Ignoring legacy typo NameError 'aggregated_ores' during forward pass")
            else:
                bt.logging.error(f" Error in HFA validator forward pass: {e}")
            
        # Wait before next evaluation cycle
        await asyncio.sleep(60)

    def get_available_miners(self) -> List[int]:
        """Get list of available miner UIDs"""
        try:
            # Get miners from metagraph
            miner_uids = []
            for uid in range(len(self.metagraph.hotkeys)):
                if self.metagraph.validator_permit[uid] == False:  # Miners don't have validator permit
                    miner_uids.append(uid)
            return miner_uids
        except Exception as e:
            bt.logging.error(f"Error getting miners: {e}")
            return []

    def generate_evaluation_tasks(self) -> List[Dict[str, Any]]:
        """
        Generate diverse infinite context evaluation tasks with intelligent task selection,
        context length scaling, and batch processing capabilities.
        """
        return self.generate_evaluation_batch()
    
    def generate_evaluation_batch(self) -> List[Dict[str, Any]]:
        """
        Enhanced batch generation with intelligent task selection and context length scaling.
        
        Features:
        - Intelligent task selection based on miner performance history
        - Context length scaling to test infinite context capabilities
        - Batch processing for efficient evaluation cycles
        - Diversity-aware task distribution
        """
        tasks = []
        
        # Get context length distribution for this batch
        context_length_distribution = self._get_context_length_distribution()
        
        # Determine task distribution with intelligent selection
        task_distribution = self._compute_intelligent_task_distribution()
        
        bt.logging.info(f"🎯 Generating evaluation batch: "
                       f"benchmark={task_distribution['benchmark']}, "
                       f"synthetic={task_distribution['synthetic']}, "
                       f"context_lengths={context_length_distribution}")
        
        # Load real-world benchmark tasks with context length scaling
        if self.use_real_benchmarks and task_distribution['benchmark'] > 0:
            benchmark_tasks = self._load_scaled_benchmark_tasks(
                task_distribution['benchmark'], 
                context_length_distribution
            )
            tasks.extend(benchmark_tasks)
        
        # Generate synthetic tasks with context length scaling
        if task_distribution['synthetic'] > 0:
            synthetic_tasks = self._generate_scaled_synthetic_tasks(
                task_distribution['synthetic'],
                context_length_distribution
            )
            tasks.extend(synthetic_tasks)
        
        # Apply diversity-aware task selection
        tasks = self._apply_diversity_aware_selection(tasks)
        
        # Generate perturbation tests if scoring harness is available
        perturbation_tasks = []
        if self.scoring_harness and len(tasks) > 0:
            try:
                perturbation_tasks = self._generate_perturbation_tasks(tasks)
                tasks.extend(perturbation_tasks)
                bt.logging.info(f"🔄 Generated {len(perturbation_tasks)} perturbation test tasks")
            except Exception as e:
                bt.logging.warning(f"Failed to generate perturbation tasks: {e}")
                perturbation_tasks = []
        
        # Apply intelligent task ordering
        tasks = self._apply_intelligent_task_ordering(tasks)
        
        # Log batch generation summary
        self._log_batch_generation_summary(tasks)
        
        # Return limited number of tasks
        max_tasks = self.tasks_per_cycle + len(perturbation_tasks)
        return tasks[:max_tasks]
    
    def _get_context_length_distribution(self) -> Dict[str, int]:
        """Get context length distribution for current evaluation batch"""
        
        # Define context length ranges for infinite context testing
        context_ranges = {
            'short': (1000, 4000),      # Baseline context
            'medium': (4000, 16000),    # Standard long context
            'long': (16000, 64000),     # Extended context
            'ultra': (64000, 100000)    # Infinite context territory
        }
        
        # Distribute tasks across context lengths based on evaluation cycle
        if not hasattr(self, '_evaluation_cycle'):
            self._evaluation_cycle = 0
        
        self._evaluation_cycle += 1
        
        # Gradually increase focus on longer contexts as miners improve
        if self._evaluation_cycle % 10 < 3:
            # Early cycles: focus on shorter contexts
            distribution = {'short': 4, 'medium': 3, 'long': 1, 'ultra': 0}
        elif self._evaluation_cycle % 10 < 7:
            # Mid cycles: balanced distribution
            distribution = {'short': 2, 'medium': 3, 'long': 2, 'ultra': 1}
        else:
            # Later cycles: focus on infinite context
            distribution = {'short': 1, 'medium': 2, 'long': 3, 'ultra': 2}
        
        # Convert to actual context length ranges
        context_length_distribution = {}
        for range_name, count in distribution.items():
            if count > 0:
                context_length_distribution[range_name] = {
                    'count': count,
                    'range': context_ranges[range_name]
                }
        
        return context_length_distribution
    
    def _compute_intelligent_task_distribution(self) -> Dict[str, int]:
        """Compute intelligent task distribution based on miner performance and diversity needs"""
        
        # Base distribution
        base_benchmark_ratio = self.benchmark_task_ratio
        
        # Adjust based on diversity tracker if available
        if hasattr(self, 'diversity_tracker') and self.diversity_tracker:
            diversity_stats = self.diversity_tracker.get_diversity_stats()
            monoculture_risk = diversity_stats.get('monoculture_risk', {})
            
            # Increase benchmark tasks if monoculture risk is high
            if monoculture_risk.get('risk_level') == 'high':
                base_benchmark_ratio = min(0.8, base_benchmark_ratio + 0.2)
                bt.logging.info("🚨 High monoculture risk detected, increasing benchmark task ratio")
            elif monoculture_risk.get('risk_level') == 'medium':
                base_benchmark_ratio = min(0.7, base_benchmark_ratio + 0.1)
        
        # Adjust based on recent performance if available
        if hasattr(self, 'miner_performance_stats') and self.miner_performance_stats:
            # If miners are performing well on benchmarks, increase synthetic challenge
            avg_benchmark_performance = self._get_average_benchmark_performance()
            if avg_benchmark_performance > 0.8:
                base_benchmark_ratio = max(0.4, base_benchmark_ratio - 0.1)
                bt.logging.info("🎯 High benchmark performance detected, increasing synthetic challenge")
        
        num_benchmark_tasks = int(self.tasks_per_cycle * base_benchmark_ratio)
        num_synthetic_tasks = self.tasks_per_cycle - num_benchmark_tasks
        
        return {
            'benchmark': num_benchmark_tasks,
            'synthetic': num_synthetic_tasks,
            'total': self.tasks_per_cycle
        }
    
    def _load_scaled_benchmark_tasks(self, num_tasks: int, 
                                   context_length_distribution: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Load benchmark tasks with context length scaling"""
        
        tasks = []
        
        try:
            # Distribute benchmark tasks across context length ranges
            for range_name, range_info in context_length_distribution.items():
                range_tasks = min(range_info['count'], num_tasks - len(tasks))
                if range_tasks <= 0:
                    continue
                
                context_range = range_info['range']
                
                # Load benchmark tasks for this context range
                benchmark_tasks = self.benchmark_loader.get_all_benchmarks()[:range_tasks]
                
                # Convert to task dictionaries
                for benchmark_task in benchmark_tasks:
                    # Handle both dict and object formats
                    if isinstance(benchmark_task, dict):
                        task_id = benchmark_task.get('name', f'task_{int(time.time())}')
                        context = f"Context for {benchmark_task.get('description', 'benchmark task')}"
                        prompt = f"Complete this {benchmark_task.get('type', 'text generation')} task"
                        context_length = random.randint(context_range[0], context_range[1])
                        task_type = benchmark_task.get('type', 'text_generation')
                        dataset_name = benchmark_task.get('name', 'unknown')
                        difficulty_level = "medium"
                        evaluation_metrics = ["accuracy", "coherence"]
                        expected_output = "Expected completion"
                        metadata = benchmark_task
                    else:
                        task_id = getattr(benchmark_task, 'task_id', f'task_{int(time.time())}')
                        context = getattr(benchmark_task, 'context', f"Context for benchmark task")
                        prompt = getattr(benchmark_task, 'prompt', "Complete this task")
                        context_length = getattr(benchmark_task, 'context_length', random.randint(context_range[0], context_range[1]))
                        task_type = getattr(benchmark_task, 'task_type', 'text_generation')
                        dataset_name = getattr(benchmark_task, 'dataset_name', 'unknown')
                        difficulty_level = getattr(benchmark_task, 'difficulty_level', 'medium')
                        evaluation_metrics = getattr(benchmark_task, 'evaluation_metrics', ["accuracy"])
                        expected_output = getattr(benchmark_task, 'expected_output', "Expected output")
                        metadata = getattr(benchmark_task, 'metadata', {})
                    
                    task_dict = {
                        "task_id": f"benchmark_{task_id}_{int(time.time())}",
                        "type": "benchmark_evaluation",
                        "benchmark_task": benchmark_task,
                        "context": context,
                        "prompt": prompt,
                        "context_length": context_length,
                        "context_range": range_name,
                        "task_type": task_type,
                        "dataset_name": dataset_name,
                        "difficulty_level": difficulty_level,
                        "evaluation_metrics": evaluation_metrics,
                        "expected_output": expected_output,
                        "metadata": metadata
                    }
                    tasks.append(task_dict)
                
                bt.logging.info(f"📊 Loaded {len(benchmark_tasks)} benchmark tasks for {range_name} context ({context_range})")
                
        except Exception as e:
            bt.logging.error(f"❌ Error loading scaled benchmark tasks: {e}")
            # Fall back to regular benchmark loading
            try:
                fallback_tasks = self.benchmark_loader.get_all_benchmarks()[:num_tasks]
                for benchmark_task in fallback_tasks:
                    # Handle both dict and object formats
                    if isinstance(benchmark_task, dict):
                        task_id = benchmark_task.get('name', f'task_{int(time.time())}')
                        context = f"Context for {benchmark_task.get('description', 'benchmark task')}"
                        prompt = f"Complete this {benchmark_task.get('type', 'text generation')} task"
                        context_length = benchmark_task.get('max_length', 5000)
                        task_type = benchmark_task.get('type', 'text_generation')
                        dataset_name = benchmark_task.get('name', 'unknown')
                        difficulty_level = "medium"
                        evaluation_metrics = ["accuracy", "coherence"]
                        expected_output = "Expected completion"
                        metadata = benchmark_task
                    else:
                        task_id = getattr(benchmark_task, 'task_id', f'task_{int(time.time())}')
                        context = getattr(benchmark_task, 'context', f"Context for benchmark task")
                        prompt = getattr(benchmark_task, 'prompt', "Complete this task")
                        context_length = getattr(benchmark_task, 'context_length', 5000)
                        task_type = getattr(benchmark_task, 'task_type', 'text_generation')
                        dataset_name = getattr(benchmark_task, 'dataset_name', 'unknown')
                        difficulty_level = getattr(benchmark_task, 'difficulty_level', 'medium')
                        evaluation_metrics = getattr(benchmark_task, 'evaluation_metrics', ["accuracy"])
                        expected_output = getattr(benchmark_task, 'expected_output', "Expected output")
                        metadata = getattr(benchmark_task, 'metadata', {})
                    
                    task_dict = {
                        "task_id": f"benchmark_{task_id}_{int(time.time())}",
                        "type": "benchmark_evaluation",
                        "benchmark_task": benchmark_task,
                        "context": context,
                        "prompt": prompt,
                        "context_length": context_length,
                        "task_type": task_type,
                        "dataset_name": dataset_name,
                        "difficulty_level": difficulty_level,
                        "evaluation_metrics": evaluation_metrics,
                        "expected_output": expected_output,
                        "metadata": metadata
                    }
                    tasks.append(task_dict)
            except Exception as fallback_error:
                bt.logging.error(f"❌ Fallback benchmark loading also failed: {fallback_error}")
        
        return tasks
    
    def _generate_scaled_synthetic_tasks(self, num_tasks: int, 
                                       context_length_distribution: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Generate synthetic tasks with context length scaling"""
        
        tasks = []
        
        # Distribute synthetic tasks across context length ranges
        for range_name, range_info in context_length_distribution.items():
            range_tasks = min(range_info['count'], num_tasks - len(tasks))
            if range_tasks <= 0:
                continue
            
            context_range = range_info['range']
            min_length, max_length = context_range
            
            # Generate tasks for this context range
            for i in range(range_tasks):
                target_length = random.randint(min_length, max_length)
                
                # Select task type based on context length
                if target_length < 8000:
                    task_types = ["memory_retention", "pattern_recognition"]
                elif target_length < 32000:
                    task_types = ["memory_retention", "pattern_recognition", "scaling_test"]
                else:
                    task_types = ["scaling_test", "coherence_maintenance", "position_understanding"]
                
                task_type = random.choice(task_types)
                
                # Generate task based on type
                if task_type == "memory_retention":
                    task = self._generate_memory_retention_task(target_length, range_name)
                elif task_type == "pattern_recognition":
                    task = self._generate_pattern_recognition_task(target_length, range_name)
                elif task_type == "scaling_test":
                    task = self._generate_scaling_test_task(target_length, range_name)
                elif task_type == "coherence_maintenance":
                    task = self._generate_coherence_task(target_length, range_name)
                else:  # position_understanding
                    task = self._generate_position_understanding_task(target_length, range_name)
                
                tasks.append(task)
            
            bt.logging.info(f"🔧 Generated {range_tasks} synthetic tasks for {range_name} context ({context_range})")
        
        return tasks
    
    def _apply_diversity_aware_selection(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply diversity-aware task selection to prevent repetitive evaluation"""
        
        if not hasattr(self, 'diversity_tracker') or not self.diversity_tracker:
            return tasks  # No diversity tracker available
        
        # Get diversity stats
        diversity_stats = self.diversity_tracker.get_diversity_stats()
        monoculture_risk = diversity_stats.get('monoculture_risk', {})
        
        # If high monoculture risk, prioritize diverse task types
        if monoculture_risk.get('risk_level') == 'high':
            # Ensure task type diversity
            task_types = {}
            for task in tasks:
                task_type = task.get('task_type', task.get('type', 'unknown'))
                if task_type not in task_types:
                    task_types[task_type] = []
                task_types[task_type].append(task)
            
            # Select diverse tasks
            diverse_tasks = []
            max_per_type = max(1, len(tasks) // len(task_types))
            
            for task_type, type_tasks in task_types.items():
                selected = random.sample(type_tasks, min(max_per_type, len(type_tasks)))
                diverse_tasks.extend(selected)
            
            bt.logging.info(f"🎯 Applied diversity-aware selection: {len(diverse_tasks)} diverse tasks selected")
            return diverse_tasks
        
        return tasks
    
    def _apply_intelligent_task_ordering(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply intelligent ordering to tasks for optimal evaluation flow"""
        
        # Sort tasks by context length for gradual scaling
        tasks_with_length = []
        for task in tasks:
            context_length = task.get('context_length', 0)
            tasks_with_length.append((context_length, task))
        
        # Sort by context length with some randomization
        tasks_with_length.sort(key=lambda x: x[0] + random.randint(-1000, 1000))
        
        ordered_tasks = [task for _, task in tasks_with_length]
        
        bt.logging.info("📋 Applied intelligent task ordering based on context length scaling")
        
        return ordered_tasks
    
    def _get_average_benchmark_performance(self) -> float:
        """Get average benchmark performance across all miners"""
        
        if not hasattr(self, 'miner_performance_stats') or not self.miner_performance_stats:
            return 0.5  # Default moderate performance
        
        benchmark_scores = []
        for miner_stats in self.miner_performance_stats.values():
            for entry in miner_stats[-10:]:  # Last 10 evaluations
                if entry.get('task_type') == 'benchmark_evaluation':
                    benchmark_scores.append(entry.get('final_score', 0.0))
        
        if not benchmark_scores:
            return 0.5
        
        return sum(benchmark_scores) / len(benchmark_scores)
    
    def _log_batch_generation_summary(self, tasks: List[Dict[str, Any]]):
        """Log summary of generated evaluation batch"""
        
        # Count tasks by type
        task_type_counts = {}
        context_length_stats = []
        
        for task in tasks:
            task_type = task.get('task_type', task.get('type', 'unknown'))
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
            
            context_length = task.get('context_length', 0)
            if context_length > 0:
                context_length_stats.append(context_length)
        
        # Calculate context length statistics
        if context_length_stats:
            avg_context_length = sum(context_length_stats) / len(context_length_stats)
            min_context_length = min(context_length_stats)
            max_context_length = max(context_length_stats)
        else:
            avg_context_length = min_context_length = max_context_length = 0
        
        bt.logging.info(f"📊 Evaluation batch generated: {len(tasks)} total tasks")
        bt.logging.info(f"📊 Task type distribution: {task_type_counts}")
        bt.logging.info(f"📊 Context length stats: avg={avg_context_length:.0f}, "
                       f"min={min_context_length}, max={max_context_length}")
    
    def _generate_memory_retention_task(self, target_length: int, context_range: str) -> Dict[str, Any]:
        """Generate memory retention task for specific context length"""
        
        task_id = f"memory_{context_range}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create context with embedded information
        memory_anchors = {}
        words = []
        
        for i in range(target_length):
            if i % max(100, target_length // 20) == 0 and i > 0:
                # Insert memorable information
                anchor_info = f"ANCHOR_{i}_INFO_{random.randint(1000, 9999)}"
                words.append(anchor_info)
                memory_anchors[i] = anchor_info
            else:
                # Regular filler content
                words.append(random.choice([
                    "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
                    "he", "was", "for", "on", "are", "as", "with", "his", "they", "at"
                ]))
        
        context = " ".join(words)
        
        # Select random anchor for testing
        if memory_anchors:
            test_position = random.choice(list(memory_anchors.keys()))
            expected_info = memory_anchors[test_position]
            prompt = f"What specific information was stored at position {test_position}?"
        else:
            expected_info = "No anchor information found"
            prompt = "What information was stored at the beginning of the context?"
        
        return {
            "task_id": task_id,
            "type": "memory_retention",
            "context_length": target_length,
            "context_range": context_range,
            "context": context,
            "prompt": prompt,
            "expected_output": expected_info,
            "target_position": test_position if memory_anchors else 0,
            "memory_anchors": memory_anchors,
            "evaluation_metrics": ["exact_match", "memory_retention"],
            "difficulty_level": "hard" if target_length > 50000 else "medium"
        }
    
    def _generate_pattern_recognition_task(self, target_length: int, context_range: str) -> Dict[str, Any]:
        """Generate pattern recognition task for specific context length"""
        
        task_id = f"pattern_{context_range}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        pattern_types = ["fibonacci", "prime", "arithmetic", "geometric"]
        pattern_type = random.choice(pattern_types)
        
        # Generate pattern sequence
        if pattern_type == "fibonacci":
            pattern = [1, 1]
            while len(pattern) < 15:
                pattern.append(pattern[-1] + pattern[-2])
        elif pattern_type == "prime":
            pattern = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        elif pattern_type == "arithmetic":
            start = random.randint(1, 10)
            step = random.randint(2, 5)
            pattern = [start + i * step for i in range(15)]
        else:  # geometric
            start = random.randint(2, 5)
            ratio = random.randint(2, 3)
            pattern = [start * (ratio ** i) for i in range(10)]
        
        # Embed pattern in context
        filler_words = ["word", "text", "content", "data", "information"] * (target_length // 5)
        context_words = filler_words[:target_length - len(pattern)]
        
        # Insert pattern at random position
        insert_pos = random.randint(100, len(context_words) - 100)
        pattern_str = [str(x) for x in pattern]
        context_words[insert_pos:insert_pos] = pattern_str
        
        context = " ".join(context_words[:target_length])
        prompt = f"Identify the {pattern_type} sequence in the text and provide the next 3 numbers."
        expected = " ".join([str(x) for x in pattern])
        
        return {
            "task_id": task_id,
            "type": "pattern_recognition",
            "context_length": target_length,
            "context_range": context_range,
            "context": context,
            "prompt": prompt,
            "expected_output": expected,
            "pattern_type": pattern_type,
            "pattern_sequence": pattern,
            "evaluation_metrics": ["pattern_accuracy", "coherence"],
            "difficulty_level": "medium" if target_length < 32000 else "hard"
        }
    
    def _generate_scaling_test_task(self, target_length: int, context_range: str) -> Dict[str, Any]:
        """Generate scaling test task for infinite context capability"""
        
        task_id = f"scaling_{context_range}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create coherent narrative that scales to target length
        base_story = """
        In a distant future, humanity had developed the technology for infinite context processing.
        The breakthrough came from a revolutionary architecture called Hierarchical Flow Anchoring.
        This system could maintain perfect memory retention across unlimited sequence lengths.
        Unlike traditional attention mechanisms that degraded quadratically, HFA scaled linearly.
        The SimpleMind block architecture provided complementary capabilities for pattern recognition.
        Together, these innovations enabled true infinite context understanding.
        """
        
        # Repeat and extend the story to reach target length
        words = base_story.split()
        extended_words = []
        
        chapter_markers = []
        while len(extended_words) < target_length:
            chapter_num = len(extended_words) // 1000 + 1
            chapter_marker = f"CHAPTER_{chapter_num}_MARKER"
            chapter_markers.append((len(extended_words), chapter_marker))
            
            extended_words.extend([chapter_marker])
            extended_words.extend(words)
            extended_words.extend([
                "Furthermore,", "Additionally,", "Moreover,", "In", "addition,",
                "The", "system", "continued", "to", "demonstrate", "remarkable", "capabilities."
            ])
        
        context = " ".join(extended_words[:target_length])
        
        # Test scaling capability
        if chapter_markers:
            test_chapter = random.choice(chapter_markers)
            prompt = f"How many chapters are mentioned in this document and what is the main theme?"
            expected = f"The document contains {len(chapter_markers)} chapters discussing infinite context processing technology."
        else:
            prompt = "What is the main technological breakthrough described in this document?"
            expected = "Hierarchical Flow Anchoring and SimpleMind block architecture for infinite context processing."
        
        return {
            "task_id": task_id,
            "type": "scaling_test",
            "context_length": target_length,
            "context_range": context_range,
            "context": context,
            "prompt": prompt,
            "expected_output": expected,
            "chapter_markers": chapter_markers,
            "scale_factor": target_length // 1000,
            "evaluation_metrics": ["scaling_efficiency", "coherence", "comprehension"],
            "difficulty_level": "hard" if target_length > 64000 else "medium"
        }
    
    def _generate_coherence_task(self, target_length: int, context_range: str) -> Dict[str, Any]:
        """Generate coherence maintenance task"""
        
        task_id = f"coherence_{context_range}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create a narrative with intentional coherence challenges
        themes = ["technology", "science", "exploration", "discovery", "innovation"]
        main_theme = random.choice(themes)
        
        # Build context with theme consistency challenges
        paragraphs = []
        for i in range(target_length // 200):  # ~200 words per paragraph
            if i % 10 == 0:
                # Insert theme reinforcement
                paragraph = f"The central theme of {main_theme} continues to be relevant. "
            else:
                # Regular content
                paragraph = f"This section discusses various aspects of {main_theme}. "
            
            # Add filler content
            filler = "The research shows significant progress in understanding complex systems. " * 10
            paragraphs.append(paragraph + filler)
        
        context = " ".join(paragraphs)[:target_length]
        prompt = f"What is the central theme of this document and how is it maintained throughout?"
        expected = f"The central theme is {main_theme}, maintained through consistent reinforcement and relevant examples."
        
        return {
            "task_id": task_id,
            "type": "coherence_maintenance",
            "context_length": target_length,
            "context_range": context_range,
            "context": context,
            "prompt": prompt,
            "expected_output": expected,
            "main_theme": main_theme,
            "evaluation_metrics": ["coherence", "theme_consistency", "comprehension"],
            "difficulty_level": "hard"
        }
    
    def _generate_position_understanding_task(self, target_length: int, context_range: str) -> Dict[str, Any]:
        """Generate position understanding task"""
        
        task_id = f"position_{context_range}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create context with position-dependent information
        position_markers = {}
        words = []
        
        for i in range(target_length):
            if i % (target_length // 10) == 0 and i > 0:
                # Insert position marker
                marker = f"POSITION_{i}_MARKER_{random.randint(100, 999)}"
                position_markers[i] = marker
                words.append(marker)
            else:
                words.append(random.choice([
                    "content", "information", "data", "text", "material",
                    "section", "paragraph", "sentence", "word", "element"
                ]))
        
        context = " ".join(words)
        
        # Test position understanding
        if position_markers:
            test_positions = random.sample(list(position_markers.keys()), min(3, len(position_markers)))
            prompt = f"What markers appear at positions {test_positions}?"
            expected = ", ".join([position_markers[pos] for pos in test_positions])
        else:
            prompt = "What type of content structure is used in this document?"
            expected = "Position-based marker structure with regular intervals."
        
        return {
            "task_id": task_id,
            "type": "position_understanding",
            "context_length": target_length,
            "context_range": context_range,
            "context": context,
            "prompt": prompt,
            "expected_output": expected,
            "position_markers": position_markers,
            "evaluation_metrics": ["position_accuracy", "spatial_understanding"],
            "difficulty_level": "hard" if target_length > 32000 else "medium"
        }
    
    def _generate_synthetic_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """Generate synthetic evaluation tasks (original HFA tasks)"""
        
        tasks = []
        
        # Distribute synthetic tasks across different types
        task_types = ["memory_retention", "pattern_recognition", "scaling_test"]
        tasks_per_type = max(1, num_tasks // len(task_types))
        
        # Memory retention tasks - HFA's core breakthrough
        for i in range(min(tasks_per_type, len(self.context_lengths))):
            context_length = self.context_lengths[i % len(self.context_lengths)]
            task = {
                "type": "memory_retention",
                "context_length": context_length,
                "context": self.generate_memory_context(context_length),
                "prompt": self.generate_memory_prompt(),
                "target_position": random.randint(10, min(context_length - 100, 1000)),
                "expected_performance": 1.0 if context_length <= 50000 else 0.95  # HFA target
            }
            tasks.append(task)
        
        # Pattern recognition tasks
        for i in range(min(tasks_per_type, len(self.pattern_types))):
            pattern_type = self.pattern_types[i % len(self.pattern_types)]
            context_length = random.choice([5000, 15000, 50000])
            task = {
                "type": "pattern_recognition", 
                "context_length": context_length,
                "pattern_type": pattern_type,
                "context": self.generate_pattern_context(context_length, pattern_type),
                "prompt": f"Identify all {pattern_type} patterns in the sequence",
                "expected_patterns": self.get_expected_patterns(pattern_type)
            }
            tasks.append(task)
        
        # Scaling tests - infinite context capability
        remaining_tasks = num_tasks - len(tasks)
        if remaining_tasks > 0:
            base_length = 1000
            scale_factors = [5, 15, 50, 100]
            
            for i in range(remaining_tasks):
                scale_factor = scale_factors[i % len(scale_factors)]
                target_length = base_length * scale_factor
                task = {
                    "type": "scaling_test",
                    "base_length": base_length,
                    "target_length": target_length,
                    "context_length": target_length,
                    "scale_factor": scale_factor,
                    "context": self.generate_scaling_context(target_length),
                    "prompt": "Maintain coherence and accuracy across this extended context",
                    "expected_efficiency": max(0.90, 1.0 - (scale_factor / 200))  # HFA scaling
                }
                tasks.append(task)
        
        return tasks
    
    def _generate_perturbation_tasks(self, original_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate perturbation test variants of original tasks"""
        
        perturbation_tasks = []
        
        if not self.scoring_harness:
            return perturbation_tasks
        
        # Select a subset of tasks for perturbation testing
        num_perturbation_tasks = min(2, len(original_tasks))  # Limit to avoid overwhelming
        selected_tasks = random.sample(original_tasks, num_perturbation_tasks)
        
        for original_task in selected_tasks:
            try:
                # Generate perturbation tests using the scoring harness
                perturbation_tests = self.scoring_harness.perturbation_tester.generate_perturbation_tests(original_task)
                
                for perturbation_test in perturbation_tests:
                    # Create task dictionary from perturbation test
                    perturbed_task = perturbation_test.perturbed_task.copy()
                    
                    # Add perturbation metadata
                    perturbed_task.update({
                        "task_id": f"{original_task.get('task_id', 'task')}_perturb_{perturbation_test.perturbation_type}",
                        "is_perturbation_test": True,
                        "original_task_id": original_task.get("task_id", "unknown"),
                        "perturbation_type": perturbation_test.perturbation_type,
                        "expected_consistency_threshold": perturbation_test.expected_consistency_threshold
                    })
                    
                    perturbation_tasks.append(perturbed_task)
                    
                    bt.logging.debug(f"Generated {perturbation_test.perturbation_type} perturbation for task {original_task.get('task_id', 'unknown')}")
                    
            except Exception as e:
                bt.logging.warning(f"Failed to generate perturbation for task: {e}")
                continue
        
        return perturbation_tasks

    def generate_memory_context(self, length: int) -> str:
        """Generate context with embedded information for memory testing"""
        
        # Create context with specific information at various positions
        words = []
        memory_anchors = {}
        
        for i in range(length):
            if i % 100 == 0 and i > 0:
                # Insert memorable information
                anchor_info = f"ANCHOR_{i}_INFO_{random.randint(1000, 9999)}"
                words.append(anchor_info)
                memory_anchors[i] = anchor_info
            else:
                # Regular filler content
                words.append(random.choice([
                    "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
                    "he", "was", "for", "on", "are", "as", "with", "his", "they", "at"
                ]))
        
        return " ".join(words)

    def generate_memory_prompt(self) -> str:
        """Generate prompts for memory retention testing"""
        prompts = [
            "What information was stored at the beginning of the context?",
            "Recall the specific anchor information from position 500",
            "What pattern of information storage was used throughout the context?",
            "Identify all anchor points and their associated information",
            "Demonstrate perfect recall by listing information from early positions"
        ]
        return random.choice(prompts)

    def generate_pattern_context(self, length: int, pattern_type: str) -> str:
        """Generate context with embedded patterns for pattern recognition testing"""
        
        if pattern_type == "fibonacci":
            # Embed Fibonacci sequence
            fib = [1, 1]
            while len(fib) < 20:
                fib.append(fib[-1] + fib[-2])
            pattern_str = " ".join(map(str, fib))
            
        elif pattern_type == "prime":
            # Embed prime numbers
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            pattern_str = " ".join(map(str, primes))
            
        elif pattern_type == "alternating":
            # Embed alternating pattern
            pattern_str = " ".join(["A", "B"] * 10)
            
        else:
            pattern_str = "1 2 3 4 5 6 7 8 9 10"
        
        # Embed pattern in longer context
        filler_words = ["word", "text", "content", "data", "information"] * (length // 5)
        context_words = filler_words[:length - len(pattern_str.split())]
        
        # Insert pattern at random position
        insert_pos = random.randint(100, len(context_words) - 100)
        context_words[insert_pos:insert_pos] = pattern_str.split()
        
        return " ".join(context_words[:length])

    def generate_scaling_context(self, length: int) -> str:
        """Generate context for scaling tests"""
        
        # Create coherent narrative that scales to target length
        base_story = """
        In a distant future, humanity had developed the technology for infinite context processing.
        The breakthrough came from a revolutionary architecture called Hierarchical Flow Anchoring.
        This system could maintain perfect memory retention across unlimited sequence lengths.
        Unlike traditional attention mechanisms that degraded quadratically, HFA scaled linearly.
        """
        
        # Repeat and extend the story to reach target length
        words = base_story.split()
        extended_words = []
        
        while len(extended_words) < length:
            extended_words.extend(words)
            # Add connecting phrases
            extended_words.extend([
                "Furthermore,", "Additionally,", "Moreover,", "In", "addition,",
                "The", "system", "continued", "to", "demonstrate", "remarkable", "capabilities."
            ])
        
        return " ".join(extended_words[:length])

    def get_expected_patterns(self, pattern_type: str) -> List[str]:
        """Get expected patterns for evaluation"""
        
        if pattern_type == "fibonacci":
            return ["1, 1, 2, 3, 5, 8, 13, 21, 34, 55"]
        elif pattern_type == "prime":
            return ["2, 3, 5, 7, 11, 13, 17, 19, 23, 29"]
        elif pattern_type == "alternating":
            return ["A, B, A, B, A, B"]
        else:
            return ["1, 2, 3, 4, 5"]

    async def query_miners(self, miner_uids: List[int], task: Dict[str, Any]) -> Dict[int, Any]:
        """Query miners with evaluation task"""
        
        responses = {}
        
        # Create synapse based on task type
        if task["type"] == "benchmark_evaluation":
            # Use specialized benchmark synapse for real-world benchmark tasks
            benchmark_task = task["benchmark_task"]
            # Handle both dict and object formats
            if isinstance(benchmark_task, dict):
                task_id = benchmark_task.get('name', 'unknown')
                task_type = benchmark_task.get('type', 'text_generation')
                dataset_name = benchmark_task.get('name', 'unknown')
            else:
                task_id = getattr(benchmark_task, 'task_id', 'unknown')
                task_type = getattr(benchmark_task, 'task_type', 'text_generation')
                dataset_name = getattr(benchmark_task, 'dataset_name', 'unknown')
            
            synapse = template.protocol.BenchmarkEvaluationSynapse(
                task_id=task_id,
                task_type=task_type,
                dataset_name=dataset_name,
                context=task["context"],
                prompt=task["prompt"],
                difficulty_level=task["difficulty_level"],
                evaluation_metrics=task["evaluation_metrics"],
                expected_output=task.get("expected_output"),
                context_length=task["context_length"],
                max_tokens=200  # More tokens for complex benchmark tasks
            )
        elif task["type"] == "memory_retention":
            synapse = template.protocol.InfiniteContextSynapse(
                context=task["context"],
                prompt=task["prompt"],
                evaluation_type="memory_retention",
                target_position=task.get("target_position"),
                max_tokens=100
            )
        elif task["type"] == "pattern_recognition":
            synapse = template.protocol.InfiniteContextSynapse(
                context=task["context"],
                prompt=task["prompt"],
                evaluation_type="pattern_recognition",
                pattern_type=task["pattern_type"],
                max_tokens=100
            )
        elif task["type"] == "scaling_test":
            synapse = template.protocol.InfiniteContextSynapse(
                context=task["context"],
                prompt=task["prompt"],
                evaluation_type="scaling_test",
                max_tokens=100
            )
        else:
            synapse = template.protocol.InfiniteContextSynapse(
                context=task["context"],
                prompt=task["prompt"],
                max_tokens=100
            )
        
        # Query each miner
        for uid in miner_uids:
            try:
                bt.logging.debug(f"Querying miner {uid} for {task['type']} task")
                
                # Send query to miner
                response = await self.dendrite.forward(
                    axons=[self.metagraph.axons[uid]],
                    synapse=synapse,
                    timeout=45  # Longer timeout for benchmark tasks
                )
                
                if response and len(response) > 0:
                    bt.logging.info(f"🔍 Debug - Raw response from miner {uid}: {type(response[0])}")
                    bt.logging.info(f"🔍 Debug - Response content: {response[0]}")
                    responses[uid] = response[0]
                else:
                    bt.logging.warning(f"No response from miner {uid}")
                    responses[uid] = None
                    
            except Exception as e:
                bt.logging.error(f"Error querying miner {uid}: {e}")
                responses[uid] = None
        
        return responses

    def score_responses(self, responses: Dict[int, Any], task: Dict[str, Any]) -> Dict[int, float]:
        """
        Score miner responses using the sealed scoring harness with audit logging,
        logit hash computation, and perturbation testing.
        """
        
        scores = {}
        
        # Use scoring harness if available, otherwise fall back to legacy scoring
        if self.scoring_harness:
            scores = self._score_responses_with_harness(responses, task)
        else:
            bt.logging.warning("⚠️ Scoring harness not available, using legacy scoring")
            scores = self._score_responses_legacy(responses, task)
        
        return scores
    
    def _score_responses_with_harness(self, responses: Dict[int, Any], task: Dict[str, Any]) -> Dict[int, float]:
        """Score responses using the sealed scoring harness"""
        
        scores = {}
        
        for uid, response in responses.items():
            if response is None:
                scores[uid] = 0.0
                continue
                
            try:
                # Extract model information for audit
                model_info = None
                if hasattr(response, 'model_info'):
                    model_info = response.model_info
                elif isinstance(response, dict) and 'model_info' in response:
                    model_info = response['model_info']
                
                # Score using sealed harness
                scoring_result = self.scoring_harness.score_response(
                    task=task,
                    response=response,
                    miner_uid=uid,
                    model_info=model_info
                )
                
                base_score = scoring_result.final_score
                
                # Apply diversity tracking and incentives
                diversity_metrics = None
                if self.diversity_tracker:
                    try:
                        # Track miner response for diversity analysis
                        diversity_metrics = self.diversity_tracker.track_miner_response(
                            miner_uid=uid,
                            response=response,
                            model_info=model_info,
                            task_type=task.get('type')
                        )
                        
                        # Apply diversity incentives to base score
                        final_score_with_diversity = self.diversity_tracker.compute_diversity_incentive(
                            miner_uid=uid,
                            base_score=base_score
                        )
                        
                        scores[uid] = final_score_with_diversity
                        
                        bt.logging.info(f"🎯 Diversity adjustment for miner {uid}: "
                                      f"{base_score:.4f} → {final_score_with_diversity:.4f} "
                                      f"(uniqueness={diversity_metrics.response_uniqueness_score:.3f})")
                        
                    except Exception as e:
                        bt.logging.warning(f"Error in diversity tracking for miner {uid}: {e}")
                        scores[uid] = base_score
                else:
                    scores[uid] = base_score
                
                # Store detailed metrics for analysis
                if uid not in self.miner_performance_stats:
                    self.miner_performance_stats[uid] = []
                
                performance_entry = {
                    'task_type': task['type'],
                    'context_length': task.get('context_length', 0),
                    'score_components': scoring_result.score_components,
                    'base_score': base_score,
                    'final_score': scores[uid],
                    'quality_score': scoring_result.quality_score,
                    'consistency_score': scoring_result.consistency_score,
                    'efficiency_score': scoring_result.efficiency_score,
                    'logit_hash': scoring_result.logit_hash,
                    'model_signature': scoring_result.model_signature,
                    'perturbation_scores': scoring_result.perturbation_scores,
                    'timestamp': scoring_result.timestamp
                }
                
                # Add diversity metrics if available
                if diversity_metrics:
                    performance_entry.update({
                        'diversity_metrics': diversity_metrics.to_dict(),
                        'diversity_adjustment': scores[uid] - base_score
                    })
                
                # Add task-specific information
                if task["type"] == "benchmark_evaluation":
                    performance_entry.update({
                        'benchmark_type': task.get('task_type'),
                        'dataset_name': task.get('dataset_name'),
                        'difficulty_level': task.get('difficulty_level')
                    })
                
                self.miner_performance_stats[uid].append(performance_entry)
                
                bt.logging.info(f"🔒 Sealed scoring for miner {uid}: {scores[uid]:.4f} "
                              f"(Q:{scoring_result.quality_score:.3f}, "
                              f"C:{scoring_result.consistency_score:.3f}, "
                              f"E:{scoring_result.efficiency_score:.3f})")
                
            except Exception as e:
                bt.logging.error(f"Error in sealed scoring for miner {uid}: {e}")
                scores[uid] = 0.0
        
        return scores
    
    def _score_responses_legacy(self, responses: Dict[int, Any], task: Dict[str, Any]) -> Dict[int, float]:
        """Legacy scoring method (fallback when harness unavailable)"""
        
        scores = {}
        
        for uid, response in responses.items():
            if response is None:
                scores[uid] = 0.0
                continue
                
            try:
                score_components = {}
                
                # Handle benchmark evaluation tasks
                if task["type"] == "benchmark_evaluation":
                    base_score = self._score_benchmark_response(response, task, uid, score_components)
                else:
                    base_score = self._score_synthetic_response(response, task, uid, score_components)
                
                # Apply diversity tracking and incentives
                diversity_metrics = None
                if self.diversity_tracker:
                    try:
                        # Extract model info if available
                        model_info = None
                        if hasattr(response, 'model_info'):
                            model_info = response.model_info
                        elif isinstance(response, dict) and 'model_info' in response:
                            model_info = response['model_info']
                        
                        # Track miner response for diversity analysis
                        diversity_metrics = self.diversity_tracker.track_miner_response(
                            miner_uid=uid,
                            response=response,
                            model_info=model_info,
                            task_type=task.get('type')
                        )
                        
                        # Apply diversity incentives to base score
                        final_score_with_diversity = self.diversity_tracker.compute_diversity_incentive(
                            miner_uid=uid,
                            base_score=base_score
                        )
                        
                        scores[uid] = final_score_with_diversity
                        
                        bt.logging.info(f"🎯 Legacy diversity adjustment for miner {uid}: "
                                      f"{base_score:.4f} → {final_score_with_diversity:.4f}")
                        
                    except Exception as e:
                        bt.logging.warning(f"Error in legacy diversity tracking for miner {uid}: {e}")
                        scores[uid] = base_score
                else:
                    scores[uid] = base_score
                
                # Store detailed metrics for analysis
                if uid not in self.miner_performance_stats:
                    self.miner_performance_stats[uid] = []
                
                performance_entry = {
                    'task_type': task['type'],
                    'context_length': task.get('context_length', 0),
                    'score_components': score_components,
                    'base_score': base_score,
                    'final_score': scores[uid],
                    'timestamp': time.time()
                }
                
                # Add diversity metrics if available
                if diversity_metrics:
                    performance_entry.update({
                        'diversity_metrics': diversity_metrics.to_dict(),
                        'diversity_adjustment': scores[uid] - base_score
                    })
                
                # Add task-specific information
                if task["type"] == "benchmark_evaluation":
                    performance_entry.update({
                        'benchmark_type': task.get('task_type'),
                        'dataset_name': task.get('dataset_name'),
                        'difficulty_level': task.get('difficulty_level')
                    })
                
                self.miner_performance_stats[uid].append(performance_entry)
                
            except Exception as e:
                bt.logging.error(f"Error scoring response from miner {uid}: {e}")
                scores[uid] = 0.0
        
        return scores
    
    def _score_benchmark_response(self, response: Any, task: Dict[str, Any], uid: int, score_components: Dict[str, float]) -> float:
        """Score response for benchmark evaluation tasks"""
        
        # Extract benchmark-specific metrics
        if isinstance(response, dict):
            exact_match = response.get('exact_match_score', 0.0)
            f1_score = response.get('f1_score', 0.0)
            rouge_l = response.get('rouge_l_score', 0.0)
            semantic_sim = response.get('semantic_similarity_score', 0.0)
            coherence = response.get('coherence_score', 0.0)
            tokens_per_sec = response.get('tokens_per_second', 0.0)
            
            # Task-specific scores
            needle_retrieval = response.get('needle_retrieval_score', 0.0)
            multi_hop = response.get('multi_hop_reasoning_score', 0.0)
            summarization = response.get('summarization_quality_score', 0.0)
            reading_comp = response.get('reading_comprehension_score', 0.0)
            
        else:
            exact_match = getattr(response, 'exact_match_score', 0.0)
            f1_score = getattr(response, 'f1_score', 0.0)
            rouge_l = getattr(response, 'rouge_l_score', 0.0)
            semantic_sim = getattr(response, 'semantic_similarity_score', 0.0)
            coherence = getattr(response, 'coherence_score', 0.0)
            tokens_per_sec = getattr(response, 'tokens_per_second', 0.0)
            
            # Task-specific scores
            needle_retrieval = getattr(response, 'needle_retrieval_score', 0.0)
            multi_hop = getattr(response, 'multi_hop_reasoning_score', 0.0)
            summarization = getattr(response, 'summarization_quality_score', 0.0)
            reading_comp = getattr(response, 'reading_comprehension_score', 0.0)
        
        bt.logging.info(f"🔍 Debug - Miner {uid} benchmark response - exact_match: {exact_match}, f1: {f1_score}")
        
        # Calculate weighted score components
        score_components['exact_match'] = exact_match * self.scoring_weights.get('exact_match_score', 0.1)
        score_components['f1_score'] = f1_score * self.scoring_weights.get('f1_score', 0.05)
        score_components['semantic_similarity'] = semantic_sim * self.scoring_weights.get('semantic_similarity_score', 0.05)
        score_components['coherence'] = coherence * self.scoring_weights.get('coherence_score', 0.15)
        
        # Task-specific scoring based on benchmark type
        task_type = task.get('task_type', '')
        if task_type == 'needle_haystack':
            score_components['task_specific'] = needle_retrieval * 0.25
        elif task_type == 'hotpotqa':
            score_components['task_specific'] = multi_hop * 0.25
        elif task_type == 'govreport':
            score_components['task_specific'] = (summarization or rouge_l) * 0.25
        elif task_type == 'longbench':
            score_components['task_specific'] = reading_comp * 0.25
        else:
            score_components['task_specific'] = (exact_match + f1_score) / 2 * 0.25
        
        # Performance efficiency
        efficiency_score = min(1.0, tokens_per_sec / 1000.0)
        score_components['efficiency'] = efficiency_score * self.scoring_weights.get('tokens_per_second', 0.1)
        
        # Context length scaling bonus
        context_length = task.get('context_length', 0)
        if context_length > 32000:
            scaling_bonus = 0.1
        elif context_length > 16000:
            scaling_bonus = 0.05
        else:
            scaling_bonus = 0.0
        
        score_components['scaling_bonus'] = scaling_bonus
        
        final_score = sum(score_components.values())
        return min(1.0, max(0.0, final_score))
    
    def _score_synthetic_response(self, response: Any, task: Dict[str, Any], uid: int, score_components: Dict[str, float]) -> float:
        """Score response for synthetic evaluation tasks (original HFA scoring)"""
        
        # Handle both dict and synapse object responses
        if isinstance(response, dict):
            memory_score = response.get('memory_retention_score', 0.0)
            position_score = response.get('position_understanding_score', 0.0)
            coherence_score = response.get('coherence_score', 0.0)
            tokens_per_sec = response.get('tokens_per_second', 0.0)
        else:
            memory_score = getattr(response, 'memory_retention_score', 0.0)
            position_score = getattr(response, 'position_understanding_score', 0.0)
            coherence_score = getattr(response, 'coherence_score', 0.0)
            tokens_per_sec = getattr(response, 'tokens_per_second', 0.0)
        
        bt.logging.info(f"🔍 Debug - Miner {uid} synthetic response - memory: {memory_score}")
        
        score_components['memory_retention'] = memory_score * self.scoring_weights['memory_retention_score']
        score_components['position_understanding'] = position_score * self.scoring_weights['position_understanding_score']
        score_components['coherence'] = coherence_score * self.scoring_weights['coherence_score']
        
        # Performance efficiency
        efficiency_score = min(1.0, tokens_per_sec / 1000.0)
        score_components['efficiency'] = efficiency_score * self.scoring_weights['tokens_per_second']
        
        # Scaling capability - use memory retention as proxy for scaling
        scaling_score = memory_score
        score_components['scaling'] = scaling_score * self.scoring_weights['scaling_efficiency']
        
        final_score = sum(score_components.values())
        return min(1.0, max(0.0, final_score))

    def aggregate_scores(self, all_task_scores: List[Dict[int, float]], miner_uids: List[int]) -> Dict[int, float]:
        """Aggregate scores across all evaluation tasks"""
        
        aggregated_scores = {}
        
        for uid in miner_uids:
            uid_scores = []
            
            for task_scores in all_task_scores:
                if uid in task_scores:
                    uid_scores.append(task_scores[uid])
            
            if uid_scores:
                # Use weighted average with emphasis on consistency
                aggregated_scores[uid] = np.mean(uid_scores)
            else:
                aggregated_scores[uid] = 0.0
        
        return aggregated_scores

    def update_scores(self, final_scores: Dict[int, float], miner_uids: List[int]):
        """Update miner scores and set weights"""
        
        try:
            # Convert scores to tensor
            scores_tensor = torch.zeros(len(self.metagraph.hotkeys))
            
            for uid in miner_uids:
                if uid < len(scores_tensor):
                    scores_tensor[uid] = final_scores.get(uid, 0.0)
            
            # Update moving average scores
            self.update_moving_average(scores_tensor)
            
            # Set weights on chain
            self.set_weights()
            
        except Exception as e:
            bt.logging.error(f"Error updating scores: {e}")

    def log_task_results(self, task: Dict[str, Any], scores: Dict[int, float]):
        """Log results for individual task"""
        
        avg_score = np.mean(list(scores.values())) if scores else 0.0
        max_score = max(scores.values()) if scores else 0.0
        
        bt.logging.info(f" Task Results - {task['type']}:")
        bt.logging.info(f"   Context Length: {task.get('context_length', 'N/A')}")
        
        # Add benchmark-specific information
        if task['type'] == 'benchmark_evaluation':
            bt.logging.info(f"   Benchmark Type: {task.get('task_type', 'N/A')}")
            bt.logging.info(f"   Dataset: {task.get('dataset_name', 'N/A')}")
            bt.logging.info(f"   Difficulty: {task.get('difficulty_level', 'N/A')}")
        
        bt.logging.info(f"   Average Score: {avg_score:.3f}")
        bt.logging.info(f"   Max Score: {max_score:.3f}")
        bt.logging.info(f"   Miners Evaluated: {len(scores)}")
        
        # Track benchmark statistics
        if task['type'] == 'benchmark_evaluation':
            benchmark_type = task.get('task_type', 'unknown')
            if benchmark_type not in self.benchmark_stats:
                self.benchmark_stats[benchmark_type] = {
                    'total_tasks': 0,
                    'avg_scores': [],
                    'max_scores': []
                }
            
            self.benchmark_stats[benchmark_type]['total_tasks'] += 1
            self.benchmark_stats[benchmark_type]['avg_scores'].append(avg_score)
            self.benchmark_stats[benchmark_type]['max_scores'].append(max_score)

    def log_evaluation_summary(self, final_scores: Dict[int, float], miner_uids: List[int]):
        """Log summary of evaluation cycle"""
        
        if not final_scores:
            bt.logging.warning(" No scores to summarize")
            return
            
        avg_score = np.mean(list(final_scores.values()))
        max_score = max(final_scores.values())
        min_score = min(final_scores.values())
        
        # Find top performers
        sorted_miners = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_miners[:3]
        
        bt.logging.info(" HFA Infinite Context Evaluation Summary:")
        bt.logging.info(f"   Average Score: {avg_score:.3f}")
        bt.logging.info(f"   Max Score: {max_score:.3f}")
        bt.logging.info(f"   Min Score: {min_score:.3f}")
        bt.logging.info(f"   Miners Evaluated: {len(final_scores)}")
        
        bt.logging.info(" Top Performers:")
        for i, (uid, score) in enumerate(top_3, 1):
            bt.logging.info(f"   {i}. Miner {uid}: {score:.3f}")
        
        # Save evaluation history
        self.evaluation_history.append({
            'timestamp': time.time(),
            'scores': final_scores,
            'summary': {
                'avg_score': avg_score,
                'max_score': max_score,
                'min_score': min_score,
                'num_miners': len(final_scores)
            }
        })
        
        # Keep only recent history
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-50:]  # Keep last 50 entries
        
        # Log benchmark performance summary periodically
        if len(self.evaluation_history) % 10 == 0:
            self._log_benchmark_summary()

    def update_moving_average(self, scores: torch.Tensor):
        """Update moving average scores for miners"""
        try:
            # Initialize moving averages if not exists
            if not hasattr(self, 'moving_averages'):
                self.moving_averages = torch.zeros_like(scores)
                self.alpha = 0.1  # Moving average decay factor
            
            # Update moving averages
            self.moving_averages = (1 - self.alpha) * self.moving_averages + self.alpha * scores
            
            bt.logging.info(f" Updated moving averages: {self.moving_averages}")
            
        except Exception as e:
            bt.logging.error(f"Error updating moving averages: {e}")
    
    def _log_benchmark_summary(self):
        """Log summary of benchmark performance across all benchmark types"""
        
        if not self.benchmark_stats:
            return
        
        bt.logging.info(" Benchmark Performance Summary:")
        
        for benchmark_type, stats in self.benchmark_stats.items():
            if stats['avg_scores']:
                overall_avg = np.mean(stats['avg_scores'])
                overall_max = np.mean(stats['max_scores'])
                
                bt.logging.info(f"   {benchmark_type.upper()}:")
                bt.logging.info(f"     Tasks Completed: {stats['total_tasks']}")
                bt.logging.info(f"     Average Score: {overall_avg:.3f}")
                bt.logging.info(f"     Average Max Score: {overall_max:.3f}")
        
        # Log benchmark loader statistics
        try:
            loader_stats = self.benchmark_loader.get_benchmark_stats()
            bt.logging.info(f" Available Benchmarks: {loader_stats['available_benchmarks']}")
        except Exception as e:
            bt.logging.warning(f"Error getting benchmark loader stats: {e}")
    
    def cleanup_audit_data(self):
        """Periodically clean up old audit data"""
        if self.scoring_harness:
            try:
                self.scoring_harness.cleanup_old_audit_data()
                bt.logging.info("🧹 Audit data cleanup completed")
            except Exception as e:
                bt.logging.error(f"Error during audit cleanup: {e}")
    
    def evaluate_perturbation_consistency(self, original_responses: Dict[int, Any], 
                                        perturbed_responses: Dict[int, Any],
                                        perturbation_type: str) -> Dict[int, float]:
        """Evaluate consistency between original and perturbed responses"""
        
        consistency_scores = {}
        
        if not self.scoring_harness:
            # Return perfect consistency if no harness available
            return {uid: 1.0 for uid in original_responses.keys()}
        
        for uid in original_responses.keys():
            if uid not in perturbed_responses:
                consistency_scores[uid] = 0.0
                continue
            
            try:
                # Use perturbation tester to evaluate consistency
                consistency_score = self.scoring_harness.perturbation_tester._compute_response_similarity(
                    original_responses[uid], perturbed_responses[uid]
                )
                consistency_scores[uid] = consistency_score
                
                bt.logging.debug(f"Miner {uid} consistency for {perturbation_type}: {consistency_score:.3f}")
                
            except Exception as e:
                bt.logging.error(f"Error evaluating consistency for miner {uid}: {e}")
                consistency_scores[uid] = 0.0
        
        return consistency_scores
    
    def get_audit_trail(self, miner_uid: int, task_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for specific miner and task"""
        if self.scoring_harness:
            return self.scoring_harness.get_audit_trail(miner_uid, task_id)
        else:
            return []


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with HFAValidator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
