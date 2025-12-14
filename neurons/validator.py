# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 QUASAR-TAO Team

import os
import time
import asyncio
import torch
import bittensor as bt
import random
import sys
import traceback
import wandb
import argparse
import math
from typing import List, Dict, Union

# Add the parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import template
from template.base.validator import BaseValidatorNeuron
from template.benchmarks.benchmark_loader import BenchmarkLoader
from template.benchmarks.metrics import dataset2metric

# --- Constants ---
WANDB_PROJECT = "quasar-long-context-subnet"
# WANDB_ENTITY will be auto-detected from logged-in user

# Penalties
PENALTY_NO_RESPONSE = 0.0
PENALTY_FAKE = -0.5 # Serious penalty for fake data if detected
TIMEOUT_PENALTY = 0.0

# Context Buckets (for nuanced scoring)
# Context Buckets (Granular & Scaled)
BUCKETS = {
    "32k": (0, 32000),
    "50k": (32000, 50000),
    "64k": (50000, 64000),
    "124k": (64000, 124000),
    "256k": (124000, 256000),
    "456k": (256000, 456000),
    "653k": (456000, 653000),
    "1m": (653000, 1000000),
    "1.5m": (1000000, 1500000),
    "2m": (1500000, 2000000),
    "infinity": (2000000, float('inf'))
}

# Non-Linear Reward Multipliers
REWARD_MULTIPLIERS = {
    "32k": 1.0,
    "50k": 1.5,
    "64k": 2.0,
    "124k": 3.0,
    "256k": 4.5,
    "456k": 6.0,
    "653k": 8.0,
    "1m": 12.0,
    "1.5m": 16.0,
    "2m": 20.0,
    "infinity": 25.0
}

class Validator(BaseValidatorNeuron):
    """
    Professional Validator for QUASAR-TAO Long Context Subnet.
    """

    def __init__(self, config=None):
        # Initialize tracking variables BEFORE super().__init__() 
        # because base class calls save_state() during sync()
        self.difficulty_level = "medium"
        self.bucket_scores = {}  # Will be properly initialized after metagraph exists
        
        super(Validator, self).__init__(config=config)
        bt.logging.info("🚀 Initializing Professional Long Context Validator...")
        
        # 1. Load Benchmarks
        self.benchmark_loader = BenchmarkLoader(config={
            'longbench': {
                'enabled_tasks': ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', 'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa']
            }
        })
        
        # Now properly initialize bucket_scores with correct size
        self.bucket_scores = {b: torch.zeros(self.metagraph.n) for b in BUCKETS}
        
        # 2. State Management
        self.load_state()
        
        # 3. WandB Init
        self.init_wandb()
        
        # 4. Concurrency Control
        self.semaphore = asyncio.Semaphore(10) # Limit concurrent heavy ops if needed

    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            if self.config.wandb.off:
                self.wandb_run = None
                return
            
            # Auto-detect entity from logged-in user
            try:
                wandb_entity = wandb.api.default_entity
            except:
                wandb_entity = None  # Let WandB use default
                
            run_name = f"validator-{self.uid}-{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            self.wandb_run = wandb.init(
                name=run_name,
                project=WANDB_PROJECT,
                entity=wandb_entity,  # Auto-detected or None for default
                config={
                    "uid": self.uid,
                    "hotkey": self.wallet.hotkey.ss58_address,
                    "version": template.__version__,
                },
                allow_val_change=True
            )
            bt.logging.success(f"✅ WandB initialized: {run_name} (entity: {wandb_entity or 'default'})")
        except Exception as e:
            bt.logging.warning(f"❌ Failed to init WandB: {e}. Running without it.")
            self.wandb_run = None

    def save_state(self):
        """Save validator state to disk."""
        try:
            state = {
                "step": self.step,
                "scores": self.scores,
                "difficulty_level": self.difficulty_level,
                "bucket_scores": self.bucket_scores # Persist bucket stats
            }
            torch.save(state, self.config.neuron.full_path + "/state.pt")
            bt.logging.info("💾 State saved.")
        except Exception as e:
            bt.logging.error(f"❌ Failed to save state: {e}")

    def load_state(self):
        """Load validator state from disk."""
        try:
            self.difficulty_level = "medium"
            self.bucket_scores = {k: torch.zeros(self.metagraph.n, device=self.device) for k in BUCKETS.keys()}
            self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)
            self.consecutive_failures = torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)
            self.moving_avg_decay = 0.9
            
            state_path = self.config.neuron.full_path + "/state.pt"
            if os.path.exists(state_path):
                state = torch.load(state_path, weights_only=False)
                self.step = state.get("step", 0)
                self.scores = state.get("scores", self.scores).to(self.device)
                self.difficulty_level = state.get("difficulty_level", "medium")
                loaded_buckets = state.get("bucket_scores", {})
                for k, v in loaded_buckets.items():
                    self.bucket_scores[k] = v.to(self.device)
                
                bt.logging.success("💾 State loaded successfully.")
        except Exception as e:
            bt.logging.warning(f"⚠️ Failed to load state (starting fresh): {e}")

    async def forward(self):
        """Main Validation Loop"""
        try:
            async with self.semaphore: # Limit concurrency
                # 1. Task Gen
                task = self.get_task()
                if not task: return

                # 2. Context Bucket logic
                bucket_name = self.get_bucket(task.context_length)
                bt.logging.info(f"📋 Task: {task.task_id} [{bucket_name}] | Len: {task.context_length}")

                # 3. Query
                miner_uids = self.get_random_miners(self.config.neuron.sample_size)
                synapse = template.protocol.BenchmarkEvaluationSynapse(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    dataset_name=task.dataset_name,
                    context=task.context,
                    prompt=task.prompt,
                    difficulty_level=task.difficulty_level
                )
                
                responses = await self.dendrite(
                    axons=[self.metagraph.axons[uid] for uid in miner_uids],
                    synapse=synapse,
                    deserialize=True,
                    timeout=max(10, task.context_length / 200) # Robust timeout
                )

                # 4. Scoring
                rewards = self.score_responses(task, responses, miner_uids)
                
                # Log rewards and accuracy
                if rewards:
                    avg_reward = sum(rewards) / len(rewards)
                    avg_accuracy = sum(self.last_raw_accuracies) / len(self.last_raw_accuracies) if hasattr(self, 'last_raw_accuracies') and self.last_raw_accuracies else 0
                    bt.logging.info(f"💰 Rewards: {avg_reward:.4f} | 🎯 Accuracy: {avg_accuracy:.4f} | Top5: {[f'{r:.3f}' for r in rewards[:5]]}")
                
                # 5. Update Weights (Per Bucket)
                self.update_scores(rewards, miner_uids, bucket_name)
                
                # 6. Log
                self.log_metrics(rewards, miner_uids, bucket_name, task)

                # 7. Persistence
                if self.step % 50 == 0:
                    self.save_state()

        except Exception as e:
            bt.logging.error(f"❌ Error in forward: {e}")
            traceback.print_exc()

    def get_task(self):
        # ... logic from previous step, utilizing BenchmarkLoader adaptive ...
        target_diff = self.difficulty_level
        if random.random() < 0.1: target_diff = random.choice(["easy", "medium", "hard", "extreme"])
        tasks = self.benchmark_loader.load_benchmark_tasks(1, difficulty=target_diff)
        return tasks[0] if tasks else None

    def get_bucket(self, length: int) -> str:
        for name, (low, high) in BUCKETS.items():
            if low <= length < high:
                return name
        return "infinity"

    def get_random_miners(self, k: int):
        uids = [uid for uid in range(self.metagraph.n) if self.metagraph.axons[uid].is_serving]
        return random.sample(uids, min(k, len(uids))) if uids else []

    def score_responses(self, task, responses, miner_uids):
        rewards = []
        metric_fn = dataset2metric.get(task.dataset_name, dataset2metric['narrativeqa'])
        
        raw_scores = []
        raw_accuracies = []  # Track raw accuracy before multipliers
        
        for uid, response in zip(miner_uids, responses):
            if not response or not response.response:
                # Penalty Logic: No Response
                self.consecutive_failures[uid] += 1
                failure_penalty = PENALTY_NO_RESPONSE - (0.1 * self.consecutive_failures[uid]) # Escalating penalty
                raw_scores.append(failure_penalty)
                continue
            
            # Reset failures if successful response
            self.consecutive_failures[uid] = 0
            
            try:
                if task.all_classes:
                    score = metric_fn(response.response, task.expected_output, all_classes=task.all_classes)
                else:
                    score = metric_fn(response.response, task.expected_output)
                
                # Track raw accuracy
                raw_accuracies.append(score)
                
                # Apply Non-Linear Multiplier based on context length
                bucket = self.get_bucket(task.context_length)
                multiplier = REWARD_MULTIPLIERS.get(bucket, 1.0)
                final_score = score * multiplier
                
                raw_scores.append(final_score)
            except Exception as e:
                bt.logging.error(f"Scoring error for {uid}: {e}")
                raw_scores.append(0.0)

        # Store raw accuracy for logging
        self.last_raw_accuracies = raw_accuracies
        
        # Use RAW SCORES directly (like Omegalabs)
        # This makes rewards EXACTLY proportional to accuracy
        # 10% accuracy = 0.1 reward, 90% accuracy = 0.9 reward
        if not raw_scores:
            return []
            
        rewards_tensor = torch.tensor(raw_scores, dtype=torch.float32)
        
        # Only clip to valid range [0, 1]
        # Context-length multipliers already applied, so just ensure valid range
        rewards_tensor = torch.clamp(rewards_tensor, 0.0, 1.0)

        return rewards_tensor.tolist()

    def update_scores(self, rewards, miner_uids, bucket_name):
        if not rewards: return
        
        r_tensor = torch.tensor(rewards, device=self.device)
        uids_tensor = torch.tensor(miner_uids, device=self.device)
        
        # Update bucket specific score
        self.bucket_scores[bucket_name].scatter_(0, uids_tensor, r_tensor * 0.1 + self.bucket_scores[bucket_name][uids_tensor] * 0.9)
        
        # Composite global score (weighted sum of buckets? or max? or avg?)
        # Strategy: Sum of buckets allows specific miners to specialize or be generalists.
        # Simple Sum for now.
        composite = torch.zeros_like(self.scores)
        for b_name in BUCKETS:
            composite += self.bucket_scores[b_name]
            
        # Normalize
        if composite.max() > 0:
            composite = composite / composite.max()
            
        self.scores = composite
        bt.logging.info(f"� Global Scores Updated. Top: {torch.topk(self.scores, 5).values}")

    def log_metrics(self, rewards, miner_uids, bucket_name, task):
        if self.config.wandb.off or not hasattr(self, 'wandb_run') or self.wandb_run is None:
            return
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        avg_accuracy = sum(self.last_raw_accuracies) / len(self.last_raw_accuracies) if hasattr(self, 'last_raw_accuracies') and self.last_raw_accuracies else 0
        
        wandb.log({
            "step": self.step,
            f"rewards/{bucket_name}": avg_reward,
            f"accuracy/{bucket_name}": avg_accuracy,
            "avg_accuracy": avg_accuracy,
            "context_length": task.context_length,
            "global_difficulty": self.difficulty_level
        })


# Entry point
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... step: {validator.step}")
            time.sleep(60)

