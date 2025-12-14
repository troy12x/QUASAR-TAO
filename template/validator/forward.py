# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 HFA Research Team

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import random
import asyncio
import numpy as np
from typing import List, Dict, Any
import bittensor as bt

from template.protocol import InfiniteContextSynapse
from template.validator.reward import get_rewards, calculate_incentive_distribution
from template.utils.uids import get_random_uids


async def forward(self):
    """
    HFA Infinite Context Validator Forward Pass
    
    This function implements the core evaluation loop for the HFA infinite context subnet.
    It generates diverse infinite context challenges, queries miners, evaluates responses
    using HFA-specific metrics, and distributes rewards based on breakthrough performance.
    
    The evaluation focuses on:
    - Memory Retention: Perfect recall across ultra-long sequences (100% target)
    - Pattern Recognition: Complex pattern detection in extended contexts  
    - Scaling Performance: Maintaining quality as context length increases
    - Position Understanding: Superior position sensitivity (224% improvement)
    - Coherence Maintenance: Semantic consistency over infinite sequences
    
    Args:
        self: Validator neuron instance with metagraph, dendrite, and configuration
    """
    
    bt.logging.info(" Starting HFA infinite context evaluation cycle...")
    
    try:
        # Select miners for evaluation - prioritize active miners
        miner_uids = get_hfa_evaluation_miners(self)
        if not miner_uids:
            bt.logging.warning(" No miners available for HFA evaluation")
            await asyncio.sleep(30)
            return
            
        bt.logging.info(f" Evaluating {len(miner_uids)} miners on infinite context capabilities")
        
        # Generate HFA-specific evaluation tasks
        evaluation_tasks = generate_hfa_evaluation_tasks(self)
        
        # Execute evaluation tasks
        all_responses = []
        all_tasks = []
        
        for task_idx, task in enumerate(evaluation_tasks):
            bt.logging.info(f" Task {task_idx + 1}/{len(evaluation_tasks)}: {task['type']} (context: {task.get('context_length', 'N/A')})")
            
            # Create synapse for this task
            synapse = create_hfa_synapse(task)
            
            # Query miners
            responses = await self.dendrite(
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=synapse,
                deserialize=True,
                timeout=45  # Longer timeout for complex infinite context tasks
            )
            
            all_responses.extend(responses)
            all_tasks.extend([task] * len(responses))
            
            # Log task completion
            valid_responses = sum(1 for r in responses if r is not None)
            bt.logging.info(f" Task completed: {valid_responses}/{len(responses)} valid responses")
            
            # Brief pause between tasks to avoid overwhelming miners
            await asyncio.sleep(3)
        
        # Calculate rewards using HFA-specific metrics
        bt.logging.info(" Calculating HFA infinite context rewards...")
        rewards = get_rewards(self, tasks=all_tasks, responses=all_responses)
        
        # Aggregate rewards per miner
        miner_rewards = aggregate_miner_rewards(rewards, miner_uids, len(evaluation_tasks))
        
        # Apply incentive distribution mechanism
        if hasattr(self, 'miner_performance_stats'):
            adjusted_rewards = calculate_incentive_distribution(
                miner_rewards, 
                self.miner_performance_stats,
                consistency_weight=0.25
            )
        else:
            adjusted_rewards = miner_rewards
        
        # Update scores and set weights
        bt.logging.info(" Updating miner scores and setting weights...")
        self.update_scores(adjusted_rewards, miner_uids)
        
        # Log evaluation summary
        log_hfa_evaluation_summary(miner_rewards, adjusted_rewards, miner_uids, evaluation_tasks)
        
        # Store evaluation history for analysis
        store_evaluation_history(self, miner_uids, miner_rewards, evaluation_tasks)
        
    except Exception as e:
        bt.logging.error(f" Error in HFA validator forward pass: {e}")
        bt.logging.error(f"Exception details: {type(e).__name__}: {str(e)}")
    
    # Wait before next evaluation cycle
    bt.logging.info(" Waiting for next evaluation cycle...")
    await asyncio.sleep(90)  # 90 second cycle for thorough evaluation


def get_hfa_evaluation_miners(self, max_miners: int = 16) -> List[int]:
    """
    Select miners for HFA infinite context evaluation.
    
    Prioritizes active miners and ensures diverse evaluation coverage.
    """
    try:
        # Get all potential miners (non-validators)
        all_miners = []
        for uid in range(len(self.metagraph.hotkeys)):
            if not self.metagraph.validator_permit[uid]:
                all_miners.append(uid)
        
        # Limit to reasonable number for evaluation efficiency
        if len(all_miners) > max_miners:
            # Randomly sample to ensure fairness
            selected_miners = random.sample(all_miners, max_miners)
        else:
            selected_miners = all_miners
        
        return selected_miners
        
    except Exception as e:
        bt.logging.error(f"Error selecting miners: {e}")
        return []


def generate_hfa_evaluation_tasks(self) -> List[Dict[str, Any]]:
    """
    Generate diverse HFA infinite context evaluation tasks.
    
    Creates tasks that test the breakthrough capabilities of HFA:
    - Perfect memory retention across ultra-long sequences
    - Superior position understanding (224% improvement)
    - Linear scaling vs quadratic degradation
    - Complex pattern recognition in extended contexts
    """
    
    tasks = []
    
    # Context lengths for infinite context testing
    context_lengths = [1000, 5000, 15000, 50000, 100000]
    
    # Memory retention tasks - HFA's core breakthrough
    for context_length in context_lengths[:3]:  # Limit for cycle efficiency
        task = {
            "type": "memory_retention",
            "context_length": context_length,
            "context": generate_memory_test_context(context_length),
            "prompt": "Recall the specific information from early positions in the context",
            "target_position": random.randint(50, min(context_length // 10, 500)),
            "expected_performance": 1.0 if context_length <= 50000 else 0.95
        }
        tasks.append(task)
    
    # Pattern recognition tasks
    pattern_types = ["fibonacci", "prime", "alternating"]
    for pattern_type in pattern_types[:2]:  # Limit for efficiency
        context_length = random.choice([5000, 15000, 50000])
        task = {
            "type": "pattern_recognition",
            "context_length": context_length,
            "pattern_type": pattern_type,
            "context": generate_pattern_test_context(context_length, pattern_type),
            "prompt": f"Identify and continue the {pattern_type} pattern embedded in the context",
            "expected_patterns": get_expected_pattern_results(pattern_type)
        }
        tasks.append(task)
    
    # Scaling efficiency tests
    for scale_factor in [10, 50]:  # Test significant scaling
        base_length = 1000
        target_length = base_length * scale_factor
        task = {
            "type": "scaling_test",
            "base_length": base_length,
            "target_length": target_length,
            "scale_factor": scale_factor,
            "context": generate_scaling_test_context(target_length),
            "prompt": "Maintain coherence and demonstrate understanding across this extended context",
            "expected_efficiency": max(0.85, 1.0 - (scale_factor / 100))
        }
        tasks.append(task)
    
    return tasks


def generate_memory_test_context(length: int) -> str:
    """Generate context with embedded memory anchors for testing perfect recall."""
    
    words = []
    anchor_positions = []
    
    for i in range(length):
        if i % 200 == 0 and i > 0:
            # Insert memory anchor
            anchor_id = random.randint(10000, 99999)
            anchor_text = f"MEMORY_ANCHOR_{anchor_id}_POSITION_{i}"
            words.append(anchor_text)
            anchor_positions.append((i, anchor_id))
        else:
            # Regular filler content
            words.append(random.choice([
                "context", "information", "data", "sequence", "processing",
                "memory", "attention", "neural", "network", "model",
                "the", "and", "to", "of", "in", "for", "with", "on"
            ]))
    
    return " ".join(words)


def generate_pattern_test_context(length: int, pattern_type: str) -> str:
    """Generate context with embedded patterns for recognition testing."""
    
    # Generate the specific pattern
    if pattern_type == "fibonacci":
        pattern = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    elif pattern_type == "prime":
        pattern = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    elif pattern_type == "alternating":
        pattern = ["ALPHA", "BETA"] * 8
    else:
        pattern = list(range(1, 13))
    
    # Create filler content
    filler_words = ["word", "text", "content", "data", "information", "context"] * (length // 6)
    
    # Insert pattern at strategic positions
    pattern_str = " ".join(map(str, pattern))
    insert_position = random.randint(100, min(length - 200, 1000))
    
    context_words = filler_words[:length - len(pattern_str.split())]
    context_words[insert_position:insert_position] = pattern_str.split()
    
    return " ".join(context_words[:length])


def generate_scaling_test_context(length: int) -> str:
    """Generate coherent context for scaling efficiency testing."""
    
    base_narrative = """
    The Hierarchical Flow Anchoring architecture represents a breakthrough in infinite context processing.
    Unlike traditional attention mechanisms that suffer from quadratic complexity, HFA achieves linear scaling.
    This revolutionary approach enables perfect memory retention across unlimited sequence lengths.
    The system maintains 100% accuracy even as context extends to hundreds of thousands of tokens.
    Position understanding improves by 224% compared to baseline transformer architectures.
    """
    
    # Extend narrative to target length while maintaining coherence
    words = base_narrative.split()
    extended_words = []
    
    section_connectors = [
        "Furthermore, the architecture demonstrates",
        "Additionally, this breakthrough enables",
        "Moreover, the system exhibits",
        "In practical applications, HFA shows",
        "Research indicates that this approach",
        "Experimental results confirm that"
    ]
    
    while len(extended_words) < length:
        extended_words.extend(words)
        if len(extended_words) < length - 20:
            connector = random.choice(section_connectors)
            extended_words.extend(connector.split())
    
    return " ".join(extended_words[:length])


def get_expected_pattern_results(pattern_type: str) -> List[str]:
    """Get expected results for pattern recognition tasks."""
    
    if pattern_type == "fibonacci":
        return ["233", "377", "610"]  # Next numbers in sequence
    elif pattern_type == "prime":
        return ["53", "59", "61"]  # Next prime numbers
    elif pattern_type == "alternating":
        return ["ALPHA", "BETA", "ALPHA"]
    else:
        return ["13", "14", "15"]


def create_hfa_synapse(task: Dict[str, Any]) -> InfiniteContextSynapse:
    """Create HFA-specific synapse for evaluation task."""
    
    synapse = InfiniteContextSynapse(
        context=task["context"],
        prompt=task["prompt"],
        evaluation_type=task["type"],
        max_tokens=150,
        context_length=task.get("context_length", len(task["context"].split()))
    )
    
    # Add task-specific parameters
    if task["type"] == "memory_retention":
        synapse.target_position = task.get("target_position")
    elif task["type"] == "pattern_recognition":
        synapse.pattern_type = task.get("pattern_type")
    elif task["type"] == "scaling_test":
        synapse.scale_factor = task.get("scale_factor")
    
    return synapse


def aggregate_miner_rewards(rewards: np.ndarray, miner_uids: List[int], num_tasks: int) -> np.ndarray:
    """Aggregate rewards across multiple tasks for each miner."""
    
    if len(rewards) == 0:
        return np.zeros(len(miner_uids))
    
    # Reshape rewards to [num_miners, num_tasks]
    num_miners = len(miner_uids)
    
    if len(rewards) != num_miners * num_tasks:
        bt.logging.warning(f"Reward array length mismatch: {len(rewards)} vs expected {num_miners * num_tasks}")
        # Pad or truncate as needed
        expected_length = num_miners * num_tasks
        if len(rewards) < expected_length:
            rewards = np.pad(rewards, (0, expected_length - len(rewards)))
        else:
            rewards = rewards[:expected_length]
    
    try:
        reward_matrix = rewards.reshape(num_tasks, num_miners)
        # Average across tasks for each miner
        miner_rewards = np.mean(reward_matrix, axis=0)
    except ValueError:
        bt.logging.error("Failed to reshape rewards, using simple averaging")
        # Fallback: simple averaging
        miner_rewards = np.zeros(num_miners)
        for i in range(num_miners):
            start_idx = i * num_tasks
            end_idx = start_idx + num_tasks
            if end_idx <= len(rewards):
                miner_rewards[i] = np.mean(rewards[start_idx:end_idx])
    
    return miner_rewards


def log_hfa_evaluation_summary(
    original_rewards: np.ndarray, 
    adjusted_rewards: np.ndarray, 
    miner_uids: List[int], 
    tasks: List[Dict[str, Any]]
):
    """Log comprehensive evaluation summary."""
    
    if len(original_rewards) == 0:
        bt.logging.warning(" No rewards to summarize")
        return
    
    bt.logging.info(" HFA Infinite Context Evaluation Summary:")
    bt.logging.info(f"   Tasks Completed: {len(tasks)}")
    bt.logging.info(f"   Miners Evaluated: {len(miner_uids)}")
    bt.logging.info(f"   Average Reward: {np.mean(original_rewards):.3f}")
    bt.logging.info(f"   Max Reward: {np.max(original_rewards):.3f}")
    bt.logging.info(f"   Min Reward: {np.min(original_rewards):.3f}")
    bt.logging.info(f"   Reward Std Dev: {np.std(original_rewards):.3f}")
    
    # Top performers
    top_indices = np.argsort(adjusted_rewards)[-3:][::-1]
    bt.logging.info(" Top Performers (Adjusted):")
    for i, idx in enumerate(top_indices, 1):
        if idx < len(miner_uids):
            uid = miner_uids[idx]
            score = adjusted_rewards[idx]
            bt.logging.info(f"   {i}. Miner {uid}: {score:.3f}")
    
    # Task type breakdown
    task_types = {}
    for task in tasks:
        task_type = task['type']
        task_types[task_type] = task_types.get(task_type, 0) + 1
    
    bt.logging.info(" Task Breakdown:")
    for task_type, count in task_types.items():
        bt.logging.info(f"   {task_type}: {count} tasks")


def store_evaluation_history(self, miner_uids: List[int], rewards: np.ndarray, tasks: List[Dict[str, Any]]):
    """Store evaluation history for performance tracking and analysis."""
    
    try:
        if not hasattr(self, 'evaluation_history'):
            self.evaluation_history = []
        
        evaluation_record = {
            'timestamp': time.time(),
            'miner_uids': miner_uids,
            'rewards': rewards.tolist() if len(rewards) > 0 else [],
            'num_tasks': len(tasks),
            'task_types': [task['type'] for task in tasks],
            'avg_reward': float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
            'max_reward': float(np.max(rewards)) if len(rewards) > 0 else 0.0
        }
        
        self.evaluation_history.append(evaluation_record)
        
        # Keep only recent history (last 50 evaluations)
        if len(self.evaluation_history) > 50:
            self.evaluation_history = self.evaluation_history[-50:]
            
    except Exception as e:
        bt.logging.error(f"Error storing evaluation history: {e}")
