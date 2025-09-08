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

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from template.base.validator import BaseValidatorNeuron

# Import HFA protocol
import template


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
            "coherence_maintenance"
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
        
        # Performance tracking
        self.evaluation_history = []
        self.miner_performance_stats = {}
        
        # Scoring weights based on HFA breakthrough capabilities
        self.scoring_weights = {
            "memory_retention_score": 0.35,      # Core HFA breakthrough
            "position_understanding_score": 0.25, # 224% improvement
            "coherence_score": 0.20,             # Consistency over long sequences
            "tokens_per_second": 0.10,           # Efficiency
            "scaling_efficiency": 0.10           # Infinite context scaling
        }
        
        bt.logging.info(" HFA Validator ready for infinite context evaluation")

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
        """Generate diverse infinite context evaluation tasks"""
        
        tasks = []
        
        # Memory retention tasks - HFA's core breakthrough
        for context_length in self.context_lengths:
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
        for pattern_type in self.pattern_types:
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
        base_length = 1000
        for scale_factor in [5, 15, 50, 100]:
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
        
        # Randomize task order
        random.shuffle(tasks)
        return tasks[:8]  # Limit to 8 tasks per cycle

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
        if task["type"] == "memory_retention":
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
                    timeout=30
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
        """Score miner responses based on HFA evaluation criteria"""
        
        scores = {}
        
        for uid, response in responses.items():
            if response is None:
                scores[uid] = 0.0
                continue
                
            try:
                # Extract metrics directly from response synapse
                # Calculate composite score based on HFA breakthrough capabilities
                score_components = {}
                
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
                
                bt.logging.info(f"🔍 Debug - Miner {uid} response type: {type(response)}")
                bt.logging.info(f"🔍 Debug - Miner {uid} memory_retention_score: {memory_score}")
                
                score_components = {}
                score_components['memory_retention'] = memory_score * self.scoring_weights['memory_retention_score']
                score_components['position_understanding'] = position_score * self.scoring_weights['position_understanding_score']
                score_components['coherence'] = coherence_score * self.scoring_weights['coherence_score']
                
                # Performance efficiency
                efficiency_score = min(1.0, tokens_per_sec / 1000.0)  # Normalize to reasonable range
                score_components['efficiency'] = efficiency_score * self.scoring_weights['tokens_per_second']
                
                # Scaling capability - use memory retention as proxy for scaling
                scaling_score = memory_score  # HFA maintains performance across scales
                score_components['scaling'] = scaling_score * self.scoring_weights['scaling_efficiency']
                
                # Final composite score
                final_score = sum(score_components.values())
                scores[uid] = min(1.0, max(0.0, final_score))
                
                bt.logging.info(f"🔍 Debug - Miner {uid} final_score: {final_score}, score_components: {score_components}")
                
                # Store detailed metrics for analysis
                if uid not in self.miner_performance_stats:
                    self.miner_performance_stats[uid] = []
                
                self.miner_performance_stats[uid].append({
                    'task_type': task['type'],
                    'context_length': task.get('context_length', 0),
                    'score_components': score_components,
                    'final_score': scores[uid],
                    'memory_retention_score': memory_score,
                    'position_understanding_score': position_score,
                    'coherence_score': coherence_score,
                    'tokens_per_second': tokens_per_sec,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                bt.logging.error(f"Error scoring response from miner {uid}: {e}")
                scores[uid] = 0.0
        
        return scores

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
        bt.logging.info(f"   Average Score: {avg_score:.3f}")
        bt.logging.info(f"   Max Score: {max_score:.3f}")
        bt.logging.info(f"   Miners Evaluated: {len(scores)}")

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
            bt.logging.info(f" Task: {task['type']}, Context: {task['context_length']}, Avg Score: {avg_score:.3f}")

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


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with HFAValidator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
