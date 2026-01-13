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
import ast
from typing import List, Dict, Union, Optional, Any

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
        self.runner_port = None
        self.runner_code = None
        self._cleanup_old_containers()
    
    def _cleanup_old_containers(self):
        """Clean up any old quasar-runner containers."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=quasar-runner", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            for name in result.stdout.strip().split('\n'):
                if name:
                    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        except:
            pass
    
    def _get_available_port(self):
        """Find an available port."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def _cleanup(self, container_id: str):
        """Stop and remove a container."""
        try:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=10)
        except:
            pass
    
    def start_container(self, code: str, function_name: str) -> Dict:
        """Start a Docker container with the code loaded. Returns container info."""
        print(f"[DOCKER] Starting container for {function_name}...", flush=True)
        
        # Get path to code_runner.py
        challenge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "challenge")
        code_runner_path = os.path.join(challenge_dir, "code_runner.py")
        
        if not os.path.exists(code_runner_path):
            return {"success": False, "error": f"code_runner.py not found at {code_runner_path}"}
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            code_file = f.name
        
        container_name = f"quasar-runner-{uuid.uuid4().hex[:8]}"
        
        # Get a random available port
        port = self._get_available_port()
        
        try:
            # Create container with code and code_runner mounted
            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "-p", f"{port}:{port}",
                "-v", f"{code_file}:/app/miner_code.py:ro",
                "-v", f"{code_runner_path}:/app/code_runner.py:ro",
                "-w", "/app",
                "python:3.11-slim",
                "sh", "-c",
                f"pip install fastapi uvicorn pydantic -q && python code_runner.py {port}"
            ]
            
            print(f"[DOCKER] Starting container on port {port}", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"[DOCKER] Failed to create container: {result.stderr}", flush=True)
                return {"success": False, "error": f"Failed to create container: {result.stderr}"}
            
            container_id = result.stdout.strip()
            print(f"[DOCKER] Container started: {container_id}", flush=True)
            
            # Wait for health check
            health_url = f"http://localhost:{port}/health"
            print(f"[DOCKER] Waiting for health check at {health_url}...", flush=True)
            
            # Give container time to install dependencies and start server
            time.sleep(5)
            
            for i in range(50):
                try:
                    response = requests.get(health_url, timeout=1)
                    if response.status_code == 200:
                        print(f"[DOCKER] Health check passed after {i*0.1}s", flush=True)
                        break
                except Exception as e:
                    if i % 10 == 0:
                        print(f"[DOCKER] Health check attempt {i+1}/50: {e}", flush=True)
                    time.sleep(0.1)
            else:
                print(f"[DOCKER] Health check failed, container logs:", flush=True)
                logs = subprocess.run(["docker", "logs", container_id], capture_output=True, text=True)
                print(f"[DOCKER] {logs.stdout}", flush=True)
                self._cleanup(container_id)
                return {"success": False, "error": "Runner container failed to start"}
            
            # Store container info for reuse
            self.runner_container_id = container_id
            self.runner_port = port
            self.runner_code = code
            
            return {"success": True, "container_id": container_id, "port": port}
            
        except Exception as e:
            print(f"[DOCKER] Exception: {traceback.format_exc()}", flush=True)
            return {"success": False, "error": str(e)}
    
    def execute_in_container(self, function_name: str, test_input: Any) -> Dict:
        """Execute a test case in the running container."""
        if not self.runner_container_id or not self.runner_port:
            return {"success": False, "error": "No running container. Call start_container first."}
        
        print(f"[DOCKER] Executing {function_name} with input {test_input}", flush=True)
        
        try:
            execute_url = f"http://localhost:{self.runner_port}/execute"
            response = requests.post(
                execute_url,
                json={"code": self.runner_code, "function_name": function_name, "test_input": test_input},
                timeout=EXECUTION_TIMEOUT
            )
            
            result_data = response.json()
            print(f"[DOCKER] Execution result: {result_data}", flush=True)
            
            return result_data
            
        except Exception as e:
            print(f"[DOCKER] Exception: {traceback.format_exc()}", flush=True)
            return {"success": False, "error": str(e)}
    
    def stop_container(self):
        """Stop and remove the running container."""
        if self.runner_container_id:
            print(f"[DOCKER] Stopping container {self.runner_container_id}...", flush=True)
            self._cleanup(self.runner_container_id)
            self.runner_container_id = None
            self.runner_port = None
            self.runner_code = None
    
    def execute_code(self, code: str, function_name: str, test_input: Any) -> Dict:
        """Execute code in a Docker container (legacy method for single execution)."""
        # Start container
        start_result = self.start_container(code, function_name)
        if not start_result.get("success"):
            return start_result
        
        # Execute
        result = self.execute_in_container(function_name, test_input)
        
        # Stop container
        self.stop_container()
        
        # Cleanup temp file
        try:
            os.unlink(code_file)
        except:
            pass
        
        return result

class Validator(BaseValidatorNeuron):
    """
    Simplified Validator for QUASAR-SUBNET.
    Evaluates miners by calling the challenge container.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("🚀 Initializing QUASAR Validator...")
        
        # Set polling interval from config (default 5 minutes = 300 seconds)
        polling_interval = getattr(config.neuron, 'polling_interval', 300)
        if hasattr(self, 'neuron'):
            self.neuron.polling_interval_seconds = polling_interval
        elif hasattr(self, '_polling_interval_seconds'):
            self._polling_interval_seconds = polling_interval
        bt.logging.info(f"⏱️ Polling interval: {polling_interval}s ({polling_interval/60:.1f} minutes)")
        
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

    def _infer_function_name(self, code: str) -> str:
        try:
            tree = ast.parse(code or "")
            fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            if "solve" in fn_names:
                return "solve"
            for preferred in ("main", "solution"):
                if preferred in fn_names:
                    return preferred
            if len(fn_names) == 1:
                return fn_names[0]
            if len(fn_names) > 1:
                return fn_names[0]
        except Exception:
            pass
        return "solve"

    def evaluate_submissions_locally(self) -> Dict[str, float]:
        """Fetch pending submissions from API and evaluate them locally using Docker.
        
        Returns:
            Dictionary mapping miner_hotkey to score (0.0 to 1.0).
        """
        print(f"[VALIDATOR] 🐳 Evaluating submissions locally with Docker...", flush=True)
        bt.logging.info("🐳 Evaluating submissions locally with Docker...")
        
        evaluated_scores = {}  # hotkey -> score
        
        try:
            # Fetch pending submissions from API
            print(f"[VALIDATOR] Fetching from: {VALIDATOR_API_URL}/get_pending_submissions", flush=True)
            response = requests.get(
                f"{VALIDATOR_API_URL}/get_pending_submissions",
                params={"limit": 10},
                timeout=30
            )
            print(f"[VALIDATOR] Response status: {response.status_code}", flush=True)
            response.raise_for_status()
            
            submissions = response.json()
            print(f"[VALIDATOR] Found {len(submissions)} pending submissions", flush=True)
            bt.logging.info(f"Found {len(submissions)} pending submissions")
            
            if not submissions:
                print("[VALIDATOR] No pending submissions to evaluate", flush=True)
                return
            
            for submission in submissions:
                result_id = str(submission["id"])
                task_id = submission["task_id"]
                code = submission["response_text"]
                miner_hotkey = submission["miner_hotkey"]
                
                function_name = self._infer_function_name(code)
                print(f"[VALIDATOR] Evaluating submission {result_id[:8]} (fn={function_name})...", flush=True)
                
                # Fetch task details to get test cases
                task_response = requests.get(
                    f"{VALIDATOR_API_URL}/get_task/{task_id}",
                    timeout=30
                )
                
                if task_response.status_code != 200:
                    print(f"[VALIDATOR] ⚠️ Failed to fetch task {task_id}, skipping submission", flush=True)
                    continue
                
                task_data = task_response.json()
                test_cases = task_data.get("test_cases", [])
                
                if not isinstance(test_cases, list) or not test_cases:
                    print(f"[VALIDATOR] ⚠️ No test cases for task {task_id}, skipping submission", flush=True)
                    continue
                
                # Evaluate code against test cases using Docker
                passed = 0
                total = len(test_cases)
                docker_failed = False
                
                # Start container once for all test cases
                print(f"[VALIDATOR] Starting container for {total} test cases...", flush=True)
                start_result = self.docker_executor.start_container(code, function_name)
                if not start_result.get("success"):
                    print(f"[VALIDATOR] ⚠️ Failed to start container: {start_result.get('error')}, skipping submission", flush=True)
                    continue
                
                try:
                    for test_case in test_cases:
                        test_input = test_case.get("input_code", "")
                        expected_output = test_case.get("expected_output")
                        
                        # Execute in running container
                        result = self.docker_executor.execute_in_container(
                            function_name=function_name,
                            test_input=test_input
                        )
                        
                        if not result.get("success"):
                            print(f"[VALIDATOR] ⚠️ Docker execution failed: {result.get('error')}, skipping submission", flush=True)
                            docker_failed = True
                            break
                        
                        if str(result.get("output")) == str(expected_output):
                            passed += 1
                finally:
                    # Stop container after all test cases
                    self.docker_executor.stop_container()
                
                # Skip if Docker failed
                if docker_failed:
                    print(f"[VALIDATOR] Skipping submission {result_id[:8]} due to Docker failure", flush=True)
                    continue
                
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
                evaluated_scores[miner_hotkey] = score
            
            if evaluated_scores:
                print(f"[VALIDATOR] ✅ Evaluated {len(evaluated_scores)} submissions", flush=True)
            else:
                print(f"[VALIDATOR] ⚠️ No submissions were evaluated", flush=True)
                
        except Exception as e:
            print(f"[VALIDATOR] ⚠️ Failed to evaluate submissions: {e}", flush=True)
            bt.logging.warning(f"Failed to evaluate submissions: {e}")
        
        return evaluated_scores

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
        """Main validation loop - evaluate pending submissions locally with Docker and submit weights."""
        print("[VALIDATOR] ➡️ Starting validation cycle...", flush=True)
        bt.logging.info("➡️ Starting validation cycle...")

        try:
            # Evaluate pending submissions locally with Docker
            print("[VALIDATOR] 🐳 Evaluating pending submissions locally...", flush=True)
            evaluated_scores = self.evaluate_submissions_locally()
            
            # If no submissions were evaluated, wait before next cycle
            if not evaluated_scores:
                print("[VALIDATOR] ⚠️ No pending submissions to evaluate, waiting 5 minutes...", flush=True)
                bt.logging.info("No pending submissions, waiting 5 minutes...")
                
                # Get polling interval from config
                polling_interval = getattr(self.neuron, 'polling_interval_seconds', 300) if hasattr(self, 'neuron') else 300
                time.sleep(polling_interval)
                return
            
            # Evaluation complete - scores are already updated in API
            # Other validators will fetch weights from /get_weights and submit to chain
            print(f"[VALIDATOR] ✅ Evaluation complete: {len(evaluated_scores)} submissions evaluated", flush=True)
            print(f"[VALIDATOR] 📊 Scores updated in API - validators can fetch weights from /get_weights", flush=True)
            bt.logging.success(f"✅ Evaluation complete: {len(evaluated_scores)} submissions")
            
            for hotkey, score in evaluated_scores.items():
                print(f"[VALIDATOR]   {hotkey[:12]}...: score={score:.4f}", flush=True)
                
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
            # Set weights on self (Bittensor reads from self.weights)
            self.weights = weights_u16
            
            # Submit weights to Bittensor (no arguments needed)
            self.set_weights()
            
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
    parser.add_argument("--neuron.polling_interval", type=int, default=300, help="Polling interval in seconds (default: 300 = 5 minutes)")
    parser.add_argument("--logging.debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Build config properly - bt.Config expects an ArgumentParser
    config = bt.Config(parser)
    
    # Run validator
    validator = Validator(config=config)
    
    print("[VALIDATOR] Starting validator loop...", flush=True)
    bt.logging.info("🚀 Starting validator loop...")
    validator.run()

