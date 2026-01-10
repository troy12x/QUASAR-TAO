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
from quasar.utils.system_metrics import get_system_metrics

# --- Constants ---
WANDB_PROJECT = "quasar-long-context-subnet"
# WANDB_ENTITY will be auto-detected from logged-in user

# Penalties
PENALTY_NO_RESPONSE = 0.0
PENALTY_FAKE = -0.5 # Serious penalty for fake data if detected
TIMEOUT_PENALTY = 0.0

# League Configuration (100k to 1M in 100k increments)
LEAGUES = ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "1M"]
LEAGUE_MULTIPLIERS = {
    "100k": 0.5,
    "200k": 0.75,
    "300k": 1.0,
    "400k": 1.25,
    "500k": 1.5,
    "600k": 1.75,
    "700k": 2.0,
    "800k": 2.25,
    "900k": 2.5,
    "1M": 3.0
}

def get_league(context_length: int) -> str:
    """Determine league based on context length."""
    for i, league in enumerate(LEAGUES):
        max_tokens = (i + 1) * 100_000
        if context_length <= max_tokens:
            return league
    return "1M"  # Fallback to highest league

class Validator(BaseValidatorNeuron):
    """
    Professional Validator for SILX INC Quasar Long Context Subnet.
    """

    def __init__(self, config=None):
        # Initialize tracking variables BEFORE super().__init__() 
        # because base class calls save_state() during sync()
        self.difficulty_level = "medium"
        self.bucket_scores = {} 
        self.cumulative_reward = 0.0
        self.cumulative_accuracy = 0.0
        self.bucket_cumulative_rewards = {} # Empty until buckets exist
        
        # Task progress tracking
        self.tasks_completed = 0
        self.total_tasks = 0
        self.current_benchmark = None
        
        # Active miners tracking for handshake
        self.active_miners = {}
        
        # Latency tracking
        self.task_latencies = {}
        
        super(Validator, self).__init__(config=config)
        bt.logging.info("🚀 Initializing Professional Long Context Validator...")
        
        # 1. Load Benchmarks
        self.benchmark_loader = BenchmarkLoader(config={
            'mrcr': {'enabled': True, 'n_needles_range': [2, 4, 8]},
            'longbench': {'enabled': True}
        })

        self.api_root = getattr(self.config, 'api_root', "https://quasar-subnet.onrender.com")
        if "localhost" in self.api_root or "127.0.0.1" in self.api_root:
            self.api_root = "https://quasar-subnet.onrender.com"
        print(f"📡 [VALIDATOR] Active API Root: {self.api_root}")
        bt.logging.info(f"🌐 Validator API Root: {self.api_root}")
        
        # Now properly initialize bucket_scores with correct size
        self.bucket_scores = {b: torch.zeros(self.metagraph.n) for b in LEAGUES}
        self.bucket_cumulative_rewards = {b: 0.0 for b in LEAGUES}
        
        # 2. State Management
        self.load_state()
        
        # 3. WandB Init
        self.init_wandb()
        
        # 4. Concurrency Control
        self._semaphore = None

    @property
    def semaphore(self):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(10)
        return self._semaphore

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
            if self.wandb_run:
                print(f"📊 [WANDB] View charts at: {self.wandb_run.get_url()}")
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
                "bucket_scores": self.bucket_scores, # Persist bucket stats
                "cumulative_reward": self.cumulative_reward,
                "cumulative_accuracy": self.cumulative_accuracy,
                "bucket_cumulative_rewards": self.bucket_cumulative_rewards
            }
            torch.save(state, self.config.neuron.full_path + "/state.pt")
            bt.logging.info("💾 State saved.")
        except Exception as e:
            bt.logging.error(f"❌ Failed to save state: {e}")

    def load_state(self):
        """Load validator state from disk."""
        try:
            self.difficulty_level = "medium"
            self.bucket_scores = {k: torch.zeros(self.metagraph.n, device=self.device) for k in LEAGUES}
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

                self.cumulative_reward = state.get("cumulative_reward", 0.0)
                self.cumulative_accuracy = state.get("cumulative_accuracy", 0.0)
                # Safe merge for bucket cumulative rewards
                loaded_bucket_rewards = state.get("bucket_cumulative_rewards", {})
                for k, v in loaded_bucket_rewards.items():
                    if k in self.bucket_cumulative_rewards:
                        self.bucket_cumulative_rewards[k] = v

                self.difficulty_level = state.get("difficulty_level", "medium")
                loaded_buckets = state.get("bucket_scores", {})
                for k, v in loaded_buckets.items():
                    if k in self.bucket_scores:
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

    async def run_handshake_phase(self, round_id: str) -> None:
        """Check miner liveness before dispatch."""
        
        # Get all miners from metagraph
        all_uids = [uid for uid in range(len(self.metagraph.axons)) if self.metagraph.axons[uid].is_serving and uid != self.uid]
        
        if not all_uids:
            bt.logging.warning("No serving miners found for handshake")
            self.active_miners[round_id] = []
            return
        
        # Send handshake to all miners
        handshake = quasar.protocol.StartRoundSynapse(
            round_id=round_id,
            timestamp=int(time.time())
        )
        
        bt.logging.info(f"🤝 Sending handshake to {len(all_uids)} miners...")
        
        try:
            responses = await self.dendrite(
                axons=[self.metagraph.axons[uid] for uid in all_uids],
                synapse=handshake,
                deserialize=False,
                timeout=10
            )
        except Exception as e:
            bt.logging.error(f"Handshake failed: {e}")
            self.active_miners[round_id] = []
            return
        
        # Track which miners are ready
        self.active_miners[round_id] = []
        
        for uid, response in zip(all_uids, responses):
            if response and response.is_ready:
                self.active_miners[round_id].append(uid)
                bt.logging.info(f"  Miner {uid} is ready (capacity={response.available_capacity}, version={response.miner_version})")
            else:
                bt.logging.warning(f"  Miner {uid} is not ready or offline")
        
        bt.logging.success(f"✅ Handshake complete: {len(self.active_miners[round_id])}/{len(all_uids)} miners ready")

    async def send_feedback(self, round_id: str, task_id: str, scores: dict, latencies: dict) -> None:
        """Send feedback to miners after evaluation."""
        
        active_uids = self.active_miners.get(round_id, [])
        
        if not active_uids:
            return
        
        for uid in active_uids:
            score = scores.get(uid, 0.0)
            latency = latencies.get(uid, 0.0)
            
            feedback = quasar.protocol.TaskFeedbackSynapse(
                round_id=round_id,
                task_id=task_id,
                score=score,
                latency_seconds=latency,
                feedback_text=f"Your score: {score:.4f}",
                suggestions="Keep improving!" if score < 0.5 else "Great job!"
            )
            
            try:
                response = await self.dendrite(
                    axons=[self.metagraph.axons[uid]],
                    synapse=feedback,
                    deserialize=False,
                    timeout=10
                )
                
                if response and response.acknowledged:
                    bt.logging.info(f"  Feedback sent to miner {uid}: score={score:.4f}")
            except Exception as e:
                bt.logging.warning(f"Failed to send feedback to miner {uid}: {e}")

    async def send_cleanup(self, round_id: str, task_id: str, validation_results: dict) -> None:
        """Send cleanup signal to miners."""
        
        active_uids = self.active_miners.get(round_id, [])
        
        if not active_uids:
            return
        
        for uid in active_uids:
            # Get validation result for this miner
            uid_result = validation_results.get(uid, {"score": 0.0})
            
            cleanup = quasar.protocol.TaskCleanupSynapse(
                task_id=task_id,
                validation_response=uid_result
            )
            
            try:
                response = await self.dendrite(
                    axons=[self.metagraph.axons[uid]],
                    synapse=cleanup,
                    deserialize=False,
                    timeout=10
                )
                
                if response and response.acknowledged:
                    if response.cleanup_ok:
                        bt.logging.info(f"  Cleanup ok for miner {uid}")
                    else:
                        bt.logging.warning(f"  Cleanup failed for miner {uid}: {response.error_message}")
            except Exception as e:
                bt.logging.warning(f"Failed to send cleanup to miner {uid}: {e}")

    async def forward(self):
        """Main Validation Loop"""
        bt.logging.info("➡️ Entering forward loop...")
        bt.logging.info("\n" + "═"*50)
        bt.logging.info("➡️  [VALIDATOR] STARTING NEW FORWARD CYCLE...")
        bt.logging.info("═"*50)
        
        # Generate round ID for this cycle
        round_id = f"round_{self.step}_{int(time.time())}"
        
        try:
            async with self.semaphore: # Limit concurrency
                # Phase 1: Handshake - Check miner liveness
                await self.run_handshake_phase(round_id)
                
                # Check if we have active miners
                active_uids = self.active_miners.get(round_id, [])
                if not active_uids:
                    bt.logging.warning("No active miners after handshake, skipping round")
                    return
                
                # 2. Task Gen (Fetch from API)
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
                    bt.logging.warning("⚠️ No task generated, skipping step.")
                    print("⚠️  [VALIDATOR] Task Generation Failed!")
                    return

                # 3. Context Bucket logic
                bucket_name = self.get_bucket(task.context_length)
                
                # Track benchmark progress
                dataset_name = getattr(task, 'dataset_name', 'unknown')
                if self.current_benchmark != dataset_name:
                    self.current_benchmark = dataset_name
                    self.tasks_completed = 0
                    # Get total tasks for this benchmark
                    try:
                        if hasattr(self.benchmark_loader, 'benchmarks') and dataset_name in self.benchmark_loader.benchmarks:
                            self.total_tasks = len(self.benchmark_loader.benchmarks[dataset_name])
                        else:
                            self.total_tasks = 100  # Default estimate
                    except:
                        self.total_tasks = 100
                
                self.tasks_completed += 1
                
                # Calculate progress percentage
                progress_pct = (self.tasks_completed / self.total_tasks * 100) if self.total_tasks > 0 else 0
                progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
                
                bt.logging.success(f"📋 Task Loaded: {task.task_id} [{bucket_name}] | Length: {task.context_length}")
                print(f"📈 Progress: [{progress_bar}] {self.tasks_completed}/{self.total_tasks} ({progress_pct:.1f}%) - {dataset_name}")

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
                
                # Cooldown/Stagger (Prevents ServerDisconnectedError on local systems)
                await asyncio.sleep(5) 
                
                print(f"📡 [VALIDATOR] Sending Light Synapse (context=None, prompt=None) to {len(miner_uids)} miners...")
                print(f"📊 [VALIDATOR] Synapse ID: {synapse.task_id} | Task Type: {synapse.task_type}")
                try:
                    # Dynamic Timeout: 90s floor + 90s per 100k tokens
                    # Aligned with Omega Labs standard floor. Safety fallback to 90.
                    timeout_floor = getattr(self.config.neuron, 'timeout_floor', 90)
                    if timeout_floor is None: timeout_floor = 90
                    
                    ctx_len = getattr(task, 'context_length', 0) or 0
                    timeout = timeout_floor + (ctx_len // 100_000) * 90
                    timeout = min(max(timeout, timeout_floor), 1800) # Clamp between floor and 30m
                    
                    bt.logging.info(f"⏳ [VALIDATOR] Querying miners (Timeout: {timeout}s)...")
                    responses = await self.dendrite(
                        axons=[self.metagraph.axons[uid] for uid in miner_uids],
                        synapse=synapse,
                        deserialize=False, 
                        timeout=timeout
                    )
                    bt.logging.success(f"📥 [VALIDATOR] Received {len([r for r in responses if r.response])} valid responses.")
                except Exception as e:
                    print(f"❌ [VALIDATOR] Dendrite error details: {type(e).__name__} - {str(e)}")
                    if "ServerDisconnectedError" in str(e) or "Disconnected" in str(e):
                        bt.logging.warning(f"📡 Miner connection dropped (ServerDisconnectedError). Likely due to extreme context length.")
                    else:
                        bt.logging.error(f"❌ Dendrite error: {e}")
                    return # Skip scoring for this failed round

                # 4. Scoring
                bt.logging.info("🧮 Starting scoring for received responses...")
                rewards = self.score_responses(task, responses, miner_uids)
                
                # Log individual metrics for transparency
                if rewards:
                    total_reward = sum(rewards)
                    avg_reward = total_reward / len(rewards)
                    avg_accuracy = sum(self.last_raw_accuracies) / len(self.last_raw_accuracies) if hasattr(self, 'last_raw_accuracies') and self.last_raw_accuracies else 0
                    
                    # Log Summary Table
                    print("\n" + "="*80)
                    print(f"📊 STEP {self.step} SUMMARY | Project: {WANDB_PROJECT}")
                    print("="*80)
                    
                    # Validator Info
                    v_incentive = float(self.metagraph.I[self.uid]) if hasattr(self.metagraph, 'I') else 0.0
                    v_emission = float(self.metagraph.E[self.uid]) if hasattr(self.metagraph, 'E') else 0.0
                    v_vtrust = float(self.metagraph.V[self.uid]) if hasattr(self.metagraph, 'V') else 0.0
                    v_stake = float(self.metagraph.S[self.uid]) if hasattr(self.metagraph, 'S') else 0.0
                    
                    print(f"👤 VALIDATOR: UID {self.uid} | Stake: {v_stake:.2f} τ | VTrust: {v_vtrust:.4f} | Incentive: {v_incentive:.4f} | Emission: {v_emission:.4f} τ")
                    print("-" * 80)
                    print(f"{'UID':<5} | {'Reward':<10} | {'Accuracy':<10} | {'Method':<15} | {'Time(s)':<8} | {'Status'}")
                    print("-" * 80)
                    for i, uid in enumerate(miner_uids):
                        # Force all values to be safe for formatting (avoid NoneType errors)
                        r_val = float(rewards[i]) if rewards[i] is not None else 0.0
                        acc_val = float(self.last_raw_accuracies[i]) if i < len(self.last_raw_accuracies) and self.last_raw_accuracies[i] is not None else 0.0
                        method = self.last_scoring_methods[i] if hasattr(self, 'last_scoring_methods') and i < len(self.last_scoring_methods) and self.last_scoring_methods[i] is not None else "unknown"
                        ptime = float(getattr(responses[i], 'processing_time', 0.0) or 0.0)
                        status = getattr(responses[i].dendrite, 'status_message', 'N/A')
                        
                        print(f"{uid:<5} | {r_val:<10.4f} | {acc_val:<10.3f} | {method:<15} | {ptime:<8.2f} | {status}")
                    
                    sys_m = get_system_metrics()
                    print("-" * 80)
                    print(f"💰 STEP REWARD: {total_reward:.4f} | 🎯 STEP ACC: {avg_accuracy:.4f} | 💾 RAM: {sys_m.get('system/ram_percent', 0):.1f}%")
                    print("="*80 + "\n")
                    
                    bt.logging.info(f"💰 Total Reward: {total_reward:.4f} | Avg Reward: {avg_reward:.4f} | 🎯 Avg Accuracy: {avg_accuracy:.4f} | Top5: {[f'{r:.3f}' for r in rewards[:5]]}")
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
                
                # 7. Send feedback to miners
                scores_dict = {uid: rewards[i] if i < len(rewards) else 0.0 for i, uid in enumerate(miner_uids)}
                latencies_dict = {uid: float(getattr(responses[i], 'processing_time', 0.0) or 0.0) if i < len(responses) else 0.0 for i, uid in enumerate(miner_uids)}
                await self.send_feedback(round_id, task.task_id, scores_dict, latencies_dict)
                
                # 8. Send cleanup to miners
                validation_results = {uid: {"score": scores_dict.get(uid, 0.0)} for uid in miner_uids}
                await self.send_cleanup(round_id, task.task_id, validation_results)

                # 9. Log
                # Note: self.last_raw_accuracies might not be accurately populated locally anymore if we move scoring entirely to API.
                # For now, we continue to compute it locally for internal logging if needed, or rely on API returns.
                self.log_metrics(rewards, miner_uids, bucket_name, task, responses)

                # 10. Persistence
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
        """Determine league based on context length."""
        return get_league(length)

    def get_random_miners(self, k: int):
        uids = [uid for uid in range(self.metagraph.n) if self.metagraph.axons[uid].is_serving and uid != self.uid]
        bt.logging.info(f"🔍 Found {len(uids)} serving miners in metagraph.")
        
        if not uids:
            return []
            
        selected = random.sample(uids, min(k, len(uids)))
        bt.logging.info(f"🎯 Selected UIDs for query: {selected}")
        return selected

    def score_responses(self, task, responses, miner_uids):
        rewards = []
        metric_fn = dataset2metric.get(task.dataset_name, dataset2metric['narrativeqa'])
        
        raw_scores = []
        raw_accuracies = []  # Track raw accuracy before multipliers
        methods = []         # Track method (symbolic vs numeric)
        
        for uid, response in zip(miner_uids, responses):
            if not response or not response.response:
                # Penalty Logic: No Response
                self.consecutive_failures[uid] += 1
                failure_penalty = PENALTY_NO_RESPONSE - (0.1 * self.consecutive_failures[uid]) # Escalating penalty
                raw_scores.append(failure_penalty)
                raw_accuracies.append(0.0)
                methods.append("no_response")
                continue
            
            # Reset failures if successful response
            self.consecutive_failures[uid] = 0
            
            try:
                if task.dataset_name in ["quasar_execution_v1", "quasar_execution_v3"]:
                    # --- Advanced Math Scoring ---
                    try:
                        miner_val_raw, method = self._extract_answer(response.response)
                        try:
                            target_val = float(task.expected_output)
                        except:
                            target_val = None

                        # Attempt Symbolic Verification with math-verify
                        try:
                            from math_verify import parse, verify
                            m_expr = parse(response.response)
                            t_expr = parse(task.expected_output)
                            
                            if m_expr and t_expr:
                                if verify(m_expr, t_expr):
                                    score = 1.0
                                    method = "symbolic"
                                else:
                                    if miner_val_raw is not None and target_val is not None:
                                        error = abs(miner_val_raw - target_val)
                                        denom = max(abs(target_val), 1e-9)
                                        rel_error = error / denom
                                        # Reciprocal Decay with 0.1 floor for any attempt
                                        score = max(0.1, 1.0 / (1.0 + rel_error))
                                        method = "numeric(rel_error)"
                                    else:
                                        score = 0.0
                            elif miner_val_raw is not None and target_val is not None:
                                # Fallback to standard numeric
                                error = abs(miner_val_raw - target_val)
                                denom = max(abs(target_val), 1e-9)
                                rel_error = error / denom
                                score = max(0.1, 1.0 / (1.0 + rel_error))
                                method = "numeric(rel_error)"
                            else:
                                score = 0.0
                        except Exception as e:
                            # Basic Reciprocal Decay Fallback
                            if miner_val_raw is not None and target_val is not None:
                                error = abs(miner_val_raw - target_val)
                                denom = max(abs(target_val), 1e-9)
                                rel_error = error / denom
                                score = max(0.1, 1.0 / (1.0 + rel_error))
                                method = "numeric(rel_error)"
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
                
                # Apply League Multiplier based on context length
                league = self.get_bucket(task.context_length)
                multiplier = LEAGUE_MULTIPLIERS.get(league, 1.0)
                final_score = score * multiplier
                
                raw_scores.append(final_score)
                methods.append(method)
            except Exception as e:
                bt.logging.error(f"Scoring error for {uid}: {e}")
                raw_scores.append(0.0)
                raw_accuracies.append(0.0)
                methods.append("error")

        # Store for logging (Ensure these are cleared per step)
        self.last_raw_accuracies = raw_accuracies
        self.last_scoring_methods = methods
        
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
        """Extract numeric answer from text, handling <think> blocks and \\boxed{}."""
        if text is None:
            return None, "none(empty)"
            
        try:
            import re
            text = text.strip()
            
            # 1. Strip <think> blocks if present
            # This prevents picking up reasoning steps as the final answer
            text_without_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            search_text = text_without_think if text_without_think else text
            
            # 2. Try \\boxed{val}
            # Handle possible spaces, LaTeX artifacts, and commas
            boxed_match = re.search(r"\\boxed\{\s*([-+]?[0-9\.,]+)\s*\}", search_text)
            if boxed_match:
                val_str = boxed_match.group(1).replace(',', '') 
                return float(val_str), "boxed"
            
            # 3. Try finding the LAST number in the text (prioritize text outside thinking blocks)
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", search_text)
            if nums:
                return float(nums[-1]), f"last_num({nums[-1]})"
            
            # 4. Fallback search (find ANY number in the entire text if nothing else worked)
            all_nums = re.findall(r"[-+]?[0-9\.]+", text)
            if all_nums:
                return float(all_nums[-1]), f"global_last({all_nums[-1]})"
                
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
        
        # Composite global score (weighted sum of leagues? or max? or avg?)
        # Strategy: Sum of leagues allows specific miners to specialize or be generalists.
        # Simple Sum for now.
        composite = torch.zeros_like(self.scores)
        for b_name in LEAGUES:
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
        
        # Update Cumulative Metrics
        step_total_reward = sum(rewards) if rewards else 0
        step_total_accuracy = sum(self.last_raw_accuracies) if hasattr(self, 'last_raw_accuracies') and self.last_raw_accuracies else 0
        
        self.cumulative_reward += step_total_reward
        self.cumulative_accuracy += step_total_accuracy
        self.bucket_cumulative_rewards[bucket_name] += step_total_reward

        # Core Global Metrics (These will always show up in main charts)
        metrics = {
            "reward": avg_reward,
            "accuracy": avg_accuracy,
            "total_reward": step_total_reward,
            "total_accuracy": step_total_accuracy,
            "cumulative_reward": self.cumulative_reward,
            "cumulative_accuracy": self.cumulative_accuracy,
            "ema_score_avg": self.scores.mean().item() if isinstance(self.scores, torch.Tensor) else float(self.scores.mean()),
            "step": self.step,
            "context_length": task.context_length,
        }
        
        # Add System Metrics
        metrics.update(get_system_metrics())
        
        # Add Validator Stats
        metrics.update({
            "validator/stake": float(self.metagraph.S[self.uid]) if hasattr(self.metagraph, 'S') else 0.0,
            "validator/vtrust": float(self.metagraph.V[self.uid]) if hasattr(self.metagraph, 'V') else 0.0,
            "validator/incentive": float(self.metagraph.I[self.uid]) if hasattr(self.metagraph, 'I') else 0.0,
            "validator/emission": float(self.metagraph.E[self.uid]) if hasattr(self.metagraph, 'E') else 0.0,
        })
        
        # Per-Bucket Metrics
        metrics[f"rewards/{bucket_name}"] = avg_reward
        metrics[f"accuracy/{bucket_name}"] = avg_accuracy
        metrics[f"total_reward/{bucket_name}"] = step_total_reward
        metrics[f"cumulative_reward/{bucket_name}"] = self.bucket_cumulative_rewards[bucket_name]
        
        # Log individual miner metrics
        if responses:
            for i, uid in enumerate(miner_uids):
                metrics[f"miner_reward/uid_{uid}"] = rewards[i]
                metrics[f"miner_accuracy/uid_{uid}"] = self.last_raw_accuracies[i] if i < len(self.last_raw_accuracies) else 0.0
                metrics[f"miner_time/uid_{uid}"] = getattr(responses[i], 'processing_time', 0.0)

        # Use commit=True to ensure it's sent immediately and increment wandb step
        wandb.log(metrics, step=self.step)
        bt.logging.debug(f"Logged to WandB for step {self.step}")


# Entry point
if __name__ == "__main__":
    validator = Validator()
    validator.run()

