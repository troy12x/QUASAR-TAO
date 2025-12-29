# The MIT License (MIT)
# Copyright © 2026 SILX INC

import os
import time
import asyncio
import torch
import bittensor as bt
import random
import sys
import traceback
import numpy as np
import wandb
import argparse
import math
import hashlib
import requests
from typing import List, Dict, Union, Tuple

# Add the parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import quasar
from quasar.base.validator import BaseValidatorNeuron
from quasar.benchmarks.benchmark_loader import BenchmarkLoader
from quasar.benchmarks.metrics import dataset2metric

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
    '32k': (0, 32_000),      # < 32k
    '128k': (32_000, 128_000),
    '500k': (128_000, 500_000),
    '1M': (500_000, 1_000_000),
    '2M': (1_000_000, 2_000_000),
}

# Non-linear Reward Multipliers
REWARD_MULTIPLIERS = {
    '32k': 0.1,    # Heavy penalty for short context (User requested ~0.1 for <100k)
    '128k': 0.1,   # Still penalize < 128k heavily
    '500k': 1.0,   # Baseline for "Long Context"
    '1M': 1.5,     # Reward scaling
    '2M': 2.0      # Maximum reward
}

class Validator(BaseValidatorNeuron):
    """
    Professional Validator for SILX INC Quasar Long Context Subnet.
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
            'mrcr': {
                'enabled': True,
                'n_needles_range': [2, 4, 8]
            }
        })

        self.api_root = getattr(self.config, 'api_root', "http://localhost:8000")
        bt.logging.info(f"🌐 Validator API Root: {self.api_root}")
        
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
                    "version": quasar.__version__,
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
                
                # Handle potential numpy vs torch conflict for scores
                loaded_scores = state.get("scores", self.scores)
                if isinstance(loaded_scores, np.ndarray):
                    self.scores = torch.from_numpy(loaded_scores).float().to(self.device)
                else:
                    self.scores = loaded_scores.to(self.device)
                    
                self.difficulty_level = state.get("difficulty_level", "medium")
                loaded_buckets = state.get("bucket_scores", {})
                for k, v in loaded_buckets.items():
                    if isinstance(v, np.ndarray):
                        self.bucket_scores[k] = torch.from_numpy(v).float().to(self.device)
                    else:
                        self.bucket_scores[k] = v.to(self.device)
                
                bt.logging.success("💾 State loaded successfully.")
        except Exception as e:
            bt.logging.warning(f"⚠️ Failed to load state (starting fresh): {e}")

    def _get_signature(self) -> str:
        """Sign the hotkey address to authenticate with the API."""
        hotkey = self.wallet.hotkey.ss58_address
        return f"0x{self.wallet.hotkey.sign(hotkey).hex()}"

    def _call_api(self, endpoint: str, method: str = "GET", data: dict = None, params: dict = None) -> Union[dict, None]:
        """Helper to call the Validator API with authentication headers."""
        url = f"{self.api_root}/{endpoint.lstrip('/')}"
        headers = {
            "Hotkey": self.wallet.hotkey.ss58_address,
            "Signature": self._get_signature()
        }
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=120)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=120)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            bt.logging.error(f"❌ API Call Error ({endpoint}): {e}")
            return None

    async def forward(self):
        """Main Validation Loop"""
        bt.logging.info("➡️ Entering forward loop...")
        try:
            async with self.semaphore: # Limit concurrency
                # 1. Task Gen (Fetch from API)
                max_ctx = getattr(self.config.neuron, 'max_context_length', None)
                api_task = self._call_api("get_task", params={"max_context_length": max_ctx})
                
                if not api_task:
                    print("⚠️ Failed to fetch task from API, falling back to local generation...")
                    task = self.get_task()
                else:
                    # Adapt API task to the local Task object format
                    # (Assuming the API returns a compatible structure or we wrap it)
                    from types import SimpleNamespace
                    # Map 'id' from API to 'task_id' for validator compatibility
                    if 'id' in api_task and 'task_id' not in api_task:
                        api_task['task_id'] = api_task['id']
                    task = SimpleNamespace(**api_task)
                    if not hasattr(task, 'context'):
                        print("⚠️ API task missing details, using local generator with API task_id...")
                        local_task = self.get_task()
                        if local_task:
                            local_task.task_id = task.id
                            task = local_task

                if not task: 
                    print("⚠️ No task generated, skipping step.")
                    return

                # 2. Context Bucket logic
                bucket_name = self.get_bucket(task.context_length)
                print(f"📋 Task: {task.task_id} [{bucket_name}] | Len: {task.context_length}")

                # 3. Query
                miner_uids = self.get_random_miners(self.config.neuron.sample_size)
                synapse = quasar.protocol.BenchmarkEvaluationSynapse(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    dataset_name=task.dataset_name,
                    # Light Synapse: Omit context and prompt to reduce bandwidth.
                    # Miners will fetch these details from the Validator API.
                    context=None,
                    prompt=None,
                    difficulty_level=task.difficulty_level,
                    evaluation_metrics=None # Trigger post_init logic correctly
                )
                
                print(f"📡 [VALIDATOR] Sending Light Synapse (context=None, prompt=None) to {len(miner_uids)} miners...")
                print(f"📊 [VALIDATOR] Synapse ID: {synapse.task_id} | Task Type: {synapse.task_type}")
                
                try:
                    responses = await self.dendrite(
                        axons=[self.metagraph.axons[uid] for uid in miner_uids],
                        synapse=synapse,
                        deserialize=False, # We want the full synapse object for metadata and .response
                        timeout=3600 # 1 hour timeout to allow for very long generation
                    )
                except Exception as e:
                    print(f"❌ [VALIDATOR] Dendrite error details: {type(e).__name__} - {str(e)}")
                    if "ServerDisconnectedError" in str(e) or "Disconnected" in str(e):
                        bt.logging.warning(f"📡 Miner connection dropped (ServerDisconnectedError). Likely due to extreme context length.")
                    else:
                        bt.logging.error(f"❌ Dendrite error: {e}")
                    return # Skip scoring for this failed round

                # 4. Scoring
                rewards = self.score_responses(task, responses, miner_uids)
                
                # Log individual metrics for transparency
                if rewards:
                    print("\n📊 --- Miner Performance Breakdown ---")
                    for i, (uid, reward, resp) in enumerate(zip(miner_uids, rewards, responses)):
                        acc = self.last_raw_accuracies[i] if i < len(self.last_raw_accuracies) else 0.0
                        ptime = getattr(resp, 'processing_time', 0.0)
                        status = getattr(resp.dendrite, 'status_message', 'N/A')
                        actual_resp = getattr(resp, 'response', 'N/A')
                        # Ensure all values are safe for formatting
                        r_val = float(reward) if reward is not None else 0.0
                        acc_val = float(acc) if acc is not None else 0.0
                        ptime_val = float(ptime) if ptime is not None else 0.0
                        
                        print(f"UID {uid:3} | Reward: {r_val:.4f} | Accuracy: {acc_val:.4f} | Time: {ptime_val:.2f}s | {status}")
                        
                        # Show extracted answer
                        extracted, method = self._extract_answer(actual_resp)
                        print(f"        └─ Expected: {task.expected_output} | Extracted: {extracted} ({method})")
                    
                    avg_reward = sum(rewards) / len(rewards)
                    avg_accuracy = sum(self.last_raw_accuracies) / len(self.last_raw_accuracies) if hasattr(self, 'last_raw_accuracies') and self.last_raw_accuracies else 0
                    bt.logging.info(f"💰 Rewards: {avg_reward:.4f} | 🎯 Accuracy: {avg_accuracy:.4f} | Top5: {[f'{r:.3f}' for r in rewards[:5]]}")
                    print("---------------------------------------\n")
                else:
                    print("⚠️ No rewards calculated (no miners or all failed).")
                
                # 5. Report Results to API (API performs Authoritative Scoring)
                for i, uid in enumerate(miner_uids):
                    response = responses[i]
                    resp_text = getattr(response, 'response', "") or ""
                    
                    report_data = {
                        "task_id": task.task_id,
                        "miner_hotkey": self.metagraph.hotkeys[uid],
                        "miner_uid": int(uid),
                        "response_text": resp_text,
                        "all_classes": getattr(task, 'all_classes', None)
                    }
                    self._call_api("report_result", method="POST", data=report_data)
                
                # 6. Sync Authoritative Scores from API for Weight Setting
                self.sync_scores_from_api()

                # 7. Log
                # Note: self.last_raw_accuracies might not be accurately populated locally anymore if we move scoring entirely to API.
                # For now, we continue to compute it locally for internal logging if needed, or rely on API returns.
                self.log_metrics(rewards, miner_uids, bucket_name, task, responses)

                # 8. Persistence
                if self.step % 50 == 0:
                    self.save_state()

        except Exception as e:
            bt.logging.error(f"❌ Error in forward: {e}")
            traceback.print_exc()

    def sync_scores_from_api(self):
        """Fetch authoritative scores from the Validator API and update local copies."""
        scores_data = self._call_api("get_scores")
        if scores_data:
            # Update self.scores based on hotkeys
            new_scores = torch.zeros(self.metagraph.n, device=self.device)
            for item in scores_data:
                hotkey = item['hotkey']
                score = item['score']
                if hotkey in self.metagraph.hotkeys:
                    uid = self.metagraph.hotkeys.index(hotkey)
                    new_scores[uid] = score
            self.scores = new_scores
            bt.logging.info("✅ Synced authoritative scores from Validator API.")

    def get_task(self):
        # ... logic from previous step, utilizing BenchmarkLoader adaptive ...
        target_diff = self.difficulty_level
        if random.random() < 0.1: target_diff = random.choice(["easy", "medium", "hard", "extreme"])
        bt.logging.info(f"🧠 Loading MRCR benchmark task (difficulty: {target_diff})...")
        max_ctx = getattr(self.config.neuron, 'max_context_length', None)
        # Force 'mrcr' type for the loader
        tasks = self.benchmark_loader.load_benchmark_tasks(1, benchmark_types=['longbench'], difficulty=target_diff, max_context_length=max_ctx)
        if not tasks:
            bt.logging.error("❌ Failed to load any tasks from BenchmarkLoader!")
            return None
            
        task = tasks[0]
        
        # [Solvability Check]
        # Verify the task result is logically consistent
        # This is a sanity check to prevent "broken" tasks from punishing miners.
        # We re-calculate the expected result from the task metadata (if available) or trust the loader.
        # For V4, we trust the loader's deterministic generation, but we log the invariant.
        
        if "quasar_execution" in task.dataset_name:
            try:
                # Basic type check
                val = float(task.expected_output)
                bt.logging.info(f"✅ Task {task.task_id} solvability check passed. Target: {val}")
            except Exception as e:
                bt.logging.error(f"❌ Task {task.task_id} failed solvability check: {e}")
                return None
                
        return task

    def get_bucket(self, length: int) -> str:
        for name, (low, high) in BUCKETS.items():
            if low <= length < high:
                return name
        return "infinity"

    def get_random_miners(self, k: int):
        uids = [uid for uid in range(self.metagraph.n) if self.metagraph.axons[uid].is_serving and uid != self.uid]
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
                raw_accuracies.append(0.0)
                continue
            
            # Reset failures if successful response
            self.consecutive_failures[uid] = 0
            
            try:
                if task.dataset_name in ["quasar_execution_v1", "quasar_execution_v3"]:
                    # Execution Scoring: Exact numeric match (with small tolerance)
                    try:
                        miner_val, method = self._extract_answer(response.response)
                        target_val = float(task.expected_output)
                        
                        # Tolerance of 0.1% or 0.01 absolute
                        if miner_val is not None and math.isclose(miner_val, target_val, rel_tol=1e-3, abs_tol=0.01):
                            score = 1.0
                        else:
                            score = 0.0
                    except:
                        score = 0.0
                elif task.dataset_name == "grid_search": # Example placeholder
                    pass
                elif task.all_classes:
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
                raw_accuracies.append(0.0)

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

    def _extract_answer(self, text: str) -> Tuple[Union[float, None], str]:
        """Extract numeric answer from text, handling \\boxed{} and CoT. Returns (value, method)."""
        if text is None:
            return None, "none(empty)"
            
        try:
            import re
            text = text.strip()
            # 1. Try \\boxed{val}
            boxed_match = re.search(r"\\boxed\{([0-9\.]+)\}", text)
            if boxed_match:
                return float(boxed_match.group(1)), "boxed"
            
            # 2. Try finding the LAST number in the text
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            if nums:
                return float(nums[-1]), f"last_num({nums[-1]})"
                
            return None, "none"
        except Exception as e:
            bt.logging.error(f"❌ Answer extraction error: {e}")
            if text:
                bt.logging.error(f"   Text snippet: {text[:200]}...")
            return None, f"error({str(e)[:50]})"

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
        k_val = min(5, len(self.scores))
        if k_val > 0:
            top_vals = torch.topk(self.scores, k_val).values
            bt.logging.info(f" Global Scores Updated. Top: {top_vals}")
        else:
            bt.logging.info(" Global Scores Updated. No miners scored.")

    def log_metrics(self, rewards, miner_uids, bucket_name, task, responses=None):
        if self.config.wandb.off or not hasattr(self, 'wandb_run') or self.wandb_run is None:
            return
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        avg_accuracy = sum(self.last_raw_accuracies) / len(self.last_raw_accuracies) if hasattr(self, 'last_raw_accuracies') and self.last_raw_accuracies else 0
        
        metrics = {
            "step": self.step,
            f"rewards/{bucket_name}": avg_reward,
            f"accuracy/{bucket_name}": avg_accuracy,
            "avg_accuracy": avg_accuracy,
            "context_length": task.context_length,
            "global_difficulty": self.difficulty_level
        }
        
        # Log individual miner metrics to tables/plots if provided
        if responses:
            for i, uid in enumerate(miner_uids):
                metrics[f"miner/{uid}/reward"] = rewards[i]
                metrics[f"miner/{uid}/accuracy"] = self.last_raw_accuracies[i] if i < len(self.last_raw_accuracies) else 0.0
                metrics[f"miner/{uid}/time"] = getattr(responses[i], 'processing_time', 0.0)

        wandb.log(metrics)


# Entry point
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... step: {validator.step}")
            time.sleep(60)

