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

import numpy as np
import math
from typing import List, Dict, Any, Optional
import bittensor as bt


def reward_infinite_context_response(
    task: Dict[str, Any], 
    response: Any, 
    expected_performance: float = 1.0
) -> float:
    """
    Reward miner response for infinite context tasks based on HFA breakthrough capabilities.
    
    This function evaluates responses using HFA-specific metrics:
    - Memory Retention: Perfect recall across ultra-long sequences (100% target)
    - Position Understanding: Superior position sensitivity (224% improvement)
    - Coherence Maintenance: Semantic consistency over infinite sequences
    - Scaling Efficiency: Linear scaling vs quadratic degradation
    - Pattern Recognition: Complex pattern detection in extended contexts
    
    Args:
        task: The evaluation task containing context and expected results
        response: The miner's response with performance metrics
        expected_performance: Expected performance baseline for the task
        
    Returns:
        float: Reward score between 0.0 and 1.0
    """
    
    if response is None:
        bt.logging.debug("No response received, returning 0 reward")
        return 0.0
    
    try:
        # Extract metrics from response
        if hasattr(response, 'deserialize'):
            metrics = response.deserialize()
        else:
            metrics = response if isinstance(response, dict) else {}
        
        # HFA-specific scoring weights based on breakthrough capabilities
        scoring_weights = {
            "memory_retention_score": 0.35,      # Core HFA breakthrough - perfect memory
            "position_understanding_score": 0.25, # 224% improvement over baselines
            "coherence_score": 0.20,             # Consistency over long sequences
            "tokens_per_second": 0.10,           # Efficiency matters for real deployment
            "scaling_efficiency": 0.10           # Infinite context scaling capability
        }
        
        reward_components = {}
        
        # 1. Memory Retention Score (Core HFA Capability)
        memory_score = metrics.get('memory_retention_score', 0.0)
        # HFA targets 100% memory retention - reward exponentially for high scores
        memory_reward = math.pow(memory_score, 2) if memory_score > 0.8 else memory_score * 0.5
        reward_components['memory_retention'] = memory_reward * scoring_weights['memory_retention_score']
        
        # 2. Position Understanding Score (224% HFA Improvement)
        position_score = metrics.get('position_understanding_score', 0.0)
        # Reward superior position sensitivity
        position_reward = min(1.0, position_score * 1.2)  # Boost for HFA's 224% improvement
        reward_components['position_understanding'] = position_reward * scoring_weights['position_understanding_score']
        
        # 3. Coherence Score (Long Sequence Consistency)
        coherence_score = metrics.get('coherence_score', 0.0)
        # Penalize coherence drops in long contexts
        context_length = task.get('context_length', 1000)
        coherence_penalty = max(0.8, 1.0 - (context_length / 100000))  # Penalty for very long contexts
        coherence_reward = coherence_score * coherence_penalty
        reward_components['coherence'] = coherence_reward * scoring_weights['coherence_score']
        
        # 4. Performance Efficiency
        tokens_per_sec = metrics.get('tokens_per_second', 0.0)
        # Normalize efficiency score - reward fast processing
        efficiency_score = min(1.0, tokens_per_sec / 1000.0)  # 1000 tokens/sec = full score
        reward_components['efficiency'] = efficiency_score * scoring_weights['tokens_per_second']
        
        # 5. Scaling Efficiency (Infinite Context Capability)
        scaling_score = metrics.get('scaling_efficiency', 0.0)
        # Reward linear scaling behavior vs quadratic degradation
        scale_factor = task.get('scale_factor', 1)
        if scale_factor > 1:
            # Bonus for maintaining performance at scale
            scaling_bonus = 1.0 + (0.1 * math.log(scale_factor))
            scaling_reward = scaling_score * scaling_bonus
        else:
            scaling_reward = scaling_score
        reward_components['scaling'] = min(1.0, scaling_reward) * scoring_weights['scaling_efficiency']
        
        # Calculate final composite reward
        final_reward = sum(reward_components.values())
        
        # Apply task-specific bonuses
        task_type = task.get('type', 'general')
        
        # Bonus for exceptional performance on challenging tasks
        if task_type == 'memory_retention' and memory_score > 0.95:
            final_reward *= 1.1  # 10% bonus for near-perfect memory
        elif task_type == 'scaling_test' and scaling_score > 0.9:
            final_reward *= 1.15  # 15% bonus for excellent scaling
        elif task_type == 'pattern_recognition' and coherence_score > 0.9:
            final_reward *= 1.05  # 5% bonus for pattern mastery
        
        # Ensure reward is within bounds
        final_reward = min(1.0, max(0.0, final_reward))
        
        # Log detailed scoring for analysis
        bt.logging.debug(f"HFA Reward Breakdown for {task_type}:")
        bt.logging.debug(f"  Memory: {reward_components.get('memory_retention', 0):.3f}")
        bt.logging.debug(f"  Position: {reward_components.get('position_understanding', 0):.3f}")
        bt.logging.debug(f"  Coherence: {reward_components.get('coherence', 0):.3f}")
        bt.logging.debug(f"  Efficiency: {reward_components.get('efficiency', 0):.3f}")
        bt.logging.debug(f"  Scaling: {reward_components.get('scaling', 0):.3f}")
        bt.logging.debug(f"  Final Reward: {final_reward:.3f}")
        
        return final_reward
        
    except Exception as e:
        bt.logging.error(f"Error calculating reward: {e}")
        return 0.0


def get_rewards(
    self,
    tasks: List[Dict[str, Any]],
    responses: List[Any],
) -> np.ndarray:
    """
    Calculate rewards for multiple HFA infinite context evaluation responses.
    
    This function processes responses from miners for various infinite context tasks
    and returns normalized rewards based on HFA breakthrough performance metrics.
    
    Args:
        self: Validator instance
        tasks: List of evaluation tasks sent to miners
        responses: List of responses from miners
        
    Returns:
        np.ndarray: Array of reward scores for each response
    """
    
    if not responses:
        bt.logging.warning("No responses to evaluate")
        return np.array([])
    
    rewards = []
    
    for i, response in enumerate(responses):
        try:
            # Get corresponding task if available
            task = tasks[i] if i < len(tasks) else {"type": "general", "context_length": 1000}
            
            # Calculate reward for this response
            reward_score = reward_infinite_context_response(task, response)
            rewards.append(reward_score)
            
            bt.logging.debug(f"Response {i}: Task={task.get('type', 'unknown')}, Reward={reward_score:.3f}")
            
        except Exception as e:
            bt.logging.error(f"Error processing response {i}: {e}")
            rewards.append(0.0)
    
    rewards_array = np.array(rewards)
    
    # Log reward statistics
    if len(rewards_array) > 0:
        bt.logging.info(f"HFA Reward Statistics:")
        bt.logging.info(f"  Mean: {np.mean(rewards_array):.3f}")
        bt.logging.info(f"  Max: {np.max(rewards_array):.3f}")
        bt.logging.info(f"  Min: {np.min(rewards_array):.3f}")
        bt.logging.info(f"  Std: {np.std(rewards_array):.3f}")
    
    return rewards_array


def calculate_incentive_distribution(
    scores: np.ndarray,
    performance_history: Dict[int, List[float]],
    consistency_weight: float = 0.3
) -> np.ndarray:
    """
    Calculate TAO incentive distribution based on HFA infinite context performance.
    
    This function implements the incentive mechanism that rewards miners for:
    - Sustained high performance on infinite context tasks
    - Consistency across different evaluation types
    - Innovation in attention mechanisms and memory retention
    - Efficient scaling to ultra-long contexts
    
    Args:
        scores: Current evaluation scores for miners
        performance_history: Historical performance data for consistency evaluation
        consistency_weight: Weight for consistency bonus (0.0 to 1.0)
        
    Returns:
        np.ndarray: Adjusted scores for TAO distribution
    """
    
    if len(scores) == 0:
        return scores
    
    adjusted_scores = scores.copy()
    
    # Apply consistency bonus for sustained performance
    for uid, history in performance_history.items():
        if uid < len(adjusted_scores) and len(history) >= 5:
            # Calculate consistency score (lower variance = higher consistency)
            recent_scores = history[-10:]  # Last 10 evaluations
            consistency_score = 1.0 - min(1.0, np.std(recent_scores))
            
            # Apply consistency bonus
            consistency_bonus = consistency_score * consistency_weight
            adjusted_scores[uid] = min(1.0, adjusted_scores[uid] * (1.0 + consistency_bonus))
    
    # Normalize to ensure fair distribution
    if np.sum(adjusted_scores) > 0:
        adjusted_scores = adjusted_scores / np.sum(adjusted_scores) * len(adjusted_scores)
    
    return adjusted_scores
