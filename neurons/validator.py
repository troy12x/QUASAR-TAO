# The MIT License (MIT)
# Copyright © 2026 SILX INC

import os
import sys
import time
import asyncio
import json
import subprocess
import tempfile
import uuid
import torch
import numpy as np
import bittensor as bt
import random
import traceback
import requests
from typing import List, Dict, Union, Optional

# Add the parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import quasar
from quasar.base.validator import BaseValidatorNeuron

# --- Constants ---
CHALLENGE_URL = os.getenv("CHALLENGE_URL", "http://localhost:8080")
VALIDATOR_API_URL = os.getenv("VALIDATOR_API_URL", "https://quasar-subnet.onrender.com")
TIMEOUT_SECS = 300
EVALUATION_DELAY = 3.0  # Delay between miner evaluations (seconds)
EXECUTION_TIMEOUT = 30  # Docker execution timeout
CODE_RUNNER_PORT = 8765

class DockerExecutor:
    """Executes code in Docker containers locally."""
    
    def __init__(self):
        self.runner_container_id = None
    
    def execute_code(self, code: str, function_name: str, test_input: Any) -> Dict:
        """Execute code in a Docker container."""
        print(f"[DOCKER] Executing {function_name} with input {test_input}", flush=True)
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            code_file = f.name
        
        container_name = f"quasar-runner-{uuid.uuid4().hex[:8]}"
        
        try:
            # Create container
            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "-p", f"{CODE_RUNNER_PORT}:{CODE_RUNNER_PORT}",
                "-v", f"{code_file}:/app/code.py:ro",
                "-w", "/app",
                "python:3.11-slim",
                "tail", "-f", "/dev/null"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return {"success": False, "error": f"Failed to create container: {result.stderr}"}
            
            container_id = result.stdout.strip()
            
            # Start code runner HTTP server
            exec_cmd = [
                "docker", "exec", "-d", container_id,
                "python", "-c",
                "import sys; sys.path.insert(0, '/app'); "
                "exec(open('/app/code.py').read(), globals()); "
                "from code_runner import app; "
                "import uvicorn; "
                "uvicorn.run(app, host='0.0.0.0', port=8765)"
            ]
            
            subprocess.run(exec_cmd, capture_output=True)
            
            # Wait for health check
            health_url = f"http://localhost:{CODE_RUNNER_PORT}/health"
            for i in range(50):
                try:
                    response = requests.get(health_url, timeout=1)
                    if response.status_code == 200:
                        break
                except:
                    time.sleep(0.1)
            else:
                self._cleanup(container_id)
                return {"success": False, "error": "Runner container failed to start"}
            
            # Send execute request
            execute_url = f"http://localhost:{CODE_RUNNER_PORT}/execute"
            response = requests.post(
                execute_url,
                json={"code": "", "function_name": function_name, "test_input": test_input},
                timeout=EXECUTION_TIMEOUT
            )
            
            result_data = response.json()
            
            # Cleanup
            self._cleanup(container_id)
            
            return result_data
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            os.unlink(code_file)
    
    def _cleanup(self, container_id: str):
        """Stop and remove container."""
        try:
            subprocess.run(["docker", "stop", container_id], capture_output=True)
            subprocess.run(["docker", "rm", container_id], capture_output=True)
        except:
            pass

class Validator(BaseValidatorNeuron):
    """
    Simplified Validator for QUASAR-SUBNET.
    Evaluates miners by calling the challenge container.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("🚀 Initializing QUASAR Validator...")
        
        # Initialize Docker executor for local execution
        self.docker_executor = DockerExecutor()
        bt.logging.info("🐳 Docker executor initialized (local execution)")
        
        # Initialize scores
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)
        self.load_state()
        
        # Cache for API scores (refreshed each cycle)
        self.api_scores_cache = {}
        self.last_api_fetch = 0
        self.api_cache_ttl = 60  # Cache for 60 seconds

        self.api_league = os.getenv("VALIDATOR_LEAGUE")
        self.api_model_name = os.getenv("VALIDATOR_MODEL_NAME")
        
        bt.logging.info(f"📡 Validator API URL: {VALIDATOR_API_URL}")

    def fetch_api_scores(self) -> Dict[str, float]:
        """Fetch all scores from validator_api and cache them."""
        current_time = time.time()
        
        # Return cached scores if still valid
        if current_time - self.last_api_fetch < self.api_cache_ttl:
            print(f"[VALIDATOR] Using cached scores (age={current_time - self.last_api_fetch:.1f}s)", flush=True)
            return self.api_scores_cache
        
        print(f"[VALIDATOR] Fetching fresh scores from API...", flush=True)
        
        try:
            params = {}
            if self.api_league:
                params["league"] = self.api_league
            if self.api_model_name:
                params["model_name"] = self.api_model_name

            response = requests.get(
                f"{VALIDATOR_API_URL}/get_scores",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            scores = response.json()
            self.api_scores_cache = {s.get("hotkey"): s.get("score", 0.0) for s in scores}
            self.last_api_fetch = current_time
            
            print(f"[VALIDATOR] Cached {len(self.api_scores_cache)} miner scores", flush=True)
            return self.api_scores_cache
            
        except Exception as e:
            print(f"[VALIDATOR] ⚠️ Failed to fetch scores: {e}", flush=True)
            bt.logging.warning(f"Failed to fetch scores from API: {e}")
            return self.api_scores_cache  # Return stale cache if fetch fails

    def evaluate_submissions_locally(self):
        """Fetch pending submissions from API and evaluate them locally using Docker."""
        print(f"[VALIDATOR] 🐳 Evaluating submissions locally with Docker...", flush=True)
        
        try:
            # Fetch pending submissions from API
            response = requests.get(
                f"{VALIDATOR_API_URL}/get_pending_submissions",
                params={"limit": 10},
                timeout=30
            )
            response.raise_for_status()
            
            submissions = response.json()
            print(f"[VALIDATOR] Found {len(submissions)} pending submissions", flush=True)
            
            for submission in submissions:
                result_id = submission["id"]
                task_id = submission["task_id"]
                code = submission["response_text"]
                miner_hotkey = submission["miner_hotkey"]
                
                print(f"[VALIDATOR] Evaluating submission {result_id[:8]}...", flush=True)
                
                # Fetch task details to get test cases
                task_response = requests.get(
                    f"{VALIDATOR_API_URL}/get_task/{task_id}",
                    timeout=30
                )
                
                if task_response.status_code != 200:
                    print(f"[VALIDATOR] ⚠️ Failed to fetch task {task_id}", flush=True)
                    continue
                
                task_data = task_response.json()
                test_cases = task_data.get("test_cases", [])
                
                if not test_cases:
                    print(f"[VALIDATOR] ⚠️ No test cases for task {task_id}", flush=True)
                    continue
                
                # Evaluate code against test cases using Docker
                passed = 0
                total = len(test_cases)
                
                for test_case in test_cases:
                    test_input = test_case.get("input_code", "")
                    expected_output = test_case.get("expected_output")
                    
                    # Execute in Docker
                    result = self.docker_executor.execute_code(
                        code=code,
                        function_name="solve",
                        test_input=test_input
                    )
                    
                    if result.get("success") and str(result.get("output")) == str(expected_output):
                        passed += 1
                
                # Calculate score
                score = passed / total if total > 0 else 0.0
                print(f"[VALIDATOR] Score: {passed}/{total} = {score:.4f}", flush=True)
                
                # Update score in API
                update_response = requests.post(
                    f"{VALIDATOR_API_URL}/update_score",
                    json={"result_id": result_id, "score": score},
                    timeout=30
                )
                update_response.raise_for_status()
                
                print(f"[VALIDATOR] ✅ Updated score for {miner_hotkey[:12]}...", flush=True)
                
        except Exception as e:
            print(f"[VALIDATOR] ⚠️ Failed to evaluate submissions: {e}", flush=True)
            bt.logging.warning(f"Failed to evaluate submissions: {e}")

    def load_state(self):
        """Load validator state from disk."""
        try:
            state_path = self.config.neuron.full_path + "/state.pt"
            if os.path.exists(state_path):
                state = torch.load(state_path, weights_only=False)
                self.step = state.get("step", 0)
                scores = state.get("scores", self.scores)
                # Convert numpy array to torch tensor if needed
                if isinstance(scores, np.ndarray):
                    scores = torch.from_numpy(scores).float()
                self.scores = scores.to(self.device)
                bt.logging.success("💾 State loaded successfully.")
        except Exception as e:
            bt.logging.warning(f"⚠️ Failed to load state (starting fresh): {e}")

    def save_state(self):
        """Save validator state to disk."""
        try:
            state = {
                "step": self.step,
                "scores": self.scores,
            }
            torch.save(state, self.config.neuron.full_path + "/state.pt")
            bt.logging.info("💾 State saved.")
        except Exception as e:
            bt.logging.error(f"❌ Failed to save state: {e}")

    async def forward(self):
        """Main validation loop - evaluate submissions locally with Docker."""
        print("[VALIDATOR] ➡️ Starting validation cycle...", flush=True)
        bt.logging.info("➡️ Starting validation cycle...")

        try:
            # Step 1: Evaluate pending submissions locally with Docker
            print("[VALIDATOR] 🐳 Step 1: Evaluating pending submissions locally...", flush=True)
            self.evaluate_submissions_locally()
            
            # Step 2: Fetch scores from API
            print("[VALIDATOR] 📊 Step 2: Fetching scores from API...", flush=True)
            api_scores = self.fetch_api_scores()

            # Get serving miners
            print("[VALIDATOR] Scanning metagraph for serving miners...", flush=True)
            all_miner_uids = [
                uid for uid in range(self.metagraph.n)
                if self.metagraph.axons[uid].is_serving and uid != self.uid
            ]

            if not all_miner_uids:
                print("[VALIDATOR] ⚠️ No serving miners found, skipping round", flush=True)
                bt.logging.warning("No serving miners found, skipping round")
                return

            print(f"[VALIDATOR] 🎯 Found {len(all_miner_uids)} serving miners in metagraph", flush=True)

            # Filter to only miners that have submitted to the API
            miner_uids = []
            for uid in all_miner_uids:
                hotkey = self.metagraph.hotkeys[uid]
                if hotkey in api_scores:
                    miner_uids.append(uid)

            if not miner_uids:
                print("[VALIDATOR] ⚠️ No miners have submitted to API yet, skipping round", flush=True)
                bt.logging.warning("No miners have submitted to API yet, skipping round")
                return

            print(f"[VALIDATOR] 🎯 Evaluating {len(miner_uids)} miners with API submissions: {miner_uids}", flush=True)
            bt.logging.info(f"🎯 Evaluating {len(miner_uids)} miners with API submissions")

            # Evaluate each miner using API scores
            rewards = []
            for i, uid in enumerate(miner_uids):
                miner_hotkey = self.metagraph.hotkeys[uid]
                score = api_scores.get(miner_hotkey, 0.0)
                rewards.append(score)
                
                # Update local scores
                self.scores[uid] = score
                
                print(f"[VALIDATOR]   Miner {uid} ({miner_hotkey[:8]}...): score={score:.4f}", flush=True)
                bt.logging.info(f"  Miner {uid} ({miner_hotkey[:8]}...): score={score:.4f}")
                
                # Delay between evaluations (except for last miner)
                if i < len(miner_uids) - 1:
                    time.sleep(EVALUATION_DELAY)

            # Log summary
            avg_score = sum(rewards) / len(rewards) if rewards else 0.0
            print(f"[VALIDATOR] ✅ Evaluation complete: avg_score={avg_score:.4f}", flush=True)
            bt.logging.success(f"✅ Evaluation complete: avg_score={avg_score:.4f}")

            # Submit weights to Bittensor
            print(f"[VALIDATOR] Submitting weights for {len(miner_uids)} miners...", flush=True)
            await self.submit_weights(miner_uids)
            
            # Save state periodically
            if self.step % 10 == 0:
                print(f"[VALIDATOR] Saving state at step {self.step}...", flush=True)
                self.save_state()
                
        except Exception as e:
            print(f"[VALIDATOR] ❌ Error in forward: {e}", flush=True)
            bt.logging.error(f"❌ Error in forward: {e}")
            traceback.print_exc()

    async def submit_weights(self, miner_uids: List[int]):
        """Submit weights to Bittensor based on challenge container scores."""
        
        # Create weight vector (all miners get 0, evaluated miners get their scores)
        weights = torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)
        
        for uid in miner_uids:
            weights[uid] = self.scores[uid]
        
        # Normalize weights to sum to 1.0
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # If all scores are 0, distribute evenly
            if len(miner_uids) > 0:
                weights[miner_uids] = 1.0 / len(miner_uids)
        
        # Convert to u16 format for Bittensor (0-65535)
        weights_u16 = (weights * 65535).to(torch.uint16)
        
        try:
            # Submit weights to Bittensor
            self.set_weights(
                weights=weights_u16,
                poll_key="quasar_subnet"
            )
            
            bt.logging.success(f"✅ Weights submitted to Bittensor")
            bt.logging.info(f"   Top miners: {[(uid, float(weights[uid])) for uid in sorted(miner_uids, key=lambda u: weights[u], reverse=True)[:5]]}")
            
        except Exception as e:
            bt.logging.error(f"❌ Failed to submit weights: {e}")

    async def evaluate_miner(self, uid: int, miner_hotkey: str) -> float:
        """Evaluate a miner by querying validator_api for their scores."""
        
        print(f"[VALIDATOR]   Miner uid={uid} hotkey={miner_hotkey[:12]}...", flush=True)
        
        # Get scores from cache (fetches from API if cache is stale)
        scores = self.fetch_api_scores()
        
        # Find the score for this specific miner
        miner_score = scores.get(miner_hotkey, 0.0)
        
        print(f"[VALIDATOR]   Score: {miner_score:.4f}", flush=True)
        bt.logging.info(f"    Miner {uid} score: {miner_score:.4f}")
        return miner_score


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=24, help="Subnet netuid")
    parser.add_argument("--wallet.name", type=str, default="validator", help="Wallet name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", help="Wallet hotkey")
    parser.add_argument("--subtensor.network", type=str, default="finney", help="Bittensor network")
    parser.add_argument("--neuron.sample_size", type=int, default=10, help="Number of miners to evaluate")
    parser.add_argument("--neuron.timeout", type=int, default=60, help="Timeout in seconds")
    parser.add_argument("--neuron.vpermit_tao_limit", type=int, default=4096, help="VPermit TAO limit")
    parser.add_argument("--logging.debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Build config properly - bt.Config expects an ArgumentParser
    config = bt.Config(parser)
    
    # Run validator
    validator = Validator(config=config)
    
    print("[VALIDATOR] Starting validator loop...", flush=True)
    bt.logging.info("🚀 Starting validator loop...")
    validator.run()

