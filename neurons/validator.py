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
import shutil
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

class PerformanceValidator:
    """Validates miner performance claims by cloning repos and running tests."""
    
    def __init__(self):
        self.validator_api_url = VALIDATOR_API_URL
        self.temp_dir = tempfile.mkdtemp(prefix="quasar_validator_")
        print(f"[VALIDATOR] Initialized with temp dir: {self.temp_dir}")
    
    def fetch_pending_submissions(self, limit: int = 10) -> List[Dict]:
        """Fetch pending submissions from validator API."""
        try:
            response = requests.get(
                f"{self.validator_api_url}/get_submission_stats",
                params={"limit": limit},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Filter for submissions that haven't been validated yet
            # For now, we'll return all recent submissions
            return data.get("recent_submissions", [])
        except Exception as e:
            print(f"[VALIDATOR] Error fetching submissions: {e}")
            return []
    
    def clone_miner_repo(self, fork_url: str) -> str:
        """Clone miner's fork repository to temporary directory."""
        repo_name = fork_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(self.temp_dir, repo_name)
        
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        
        print(f"[VALIDATOR] Cloning repo: {fork_url}")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", fork_url, repo_path],
                check=True,
                capture_output=True,
                timeout=120
            )
            print(f"[VALIDATOR] Repo cloned to: {repo_path}")
            return repo_path
        except subprocess.TimeoutExpired:
            print(f"[VALIDATOR] Clone timeout for {fork_url}")
            raise
        except subprocess.CalledProcessError as e:
            print(f"[VALIDATOR] Clone failed: {e.stderr}")
            raise
    
    def run_performance_test(self, repo_path: str, sequence_length: int = 100000) -> float:
        """Run performance test on cloned repository."""
        test_script = os.path.join(repo_path, "test_quasar_attention.py")
        
        if not os.path.exists(test_script):
            print(f"[VALIDATOR] Test script not found: {test_script}")
            return 0.0
        
        print(f"[VALIDATOR] Running performance test...")
        
        # Create temporary test script with target sequence length
        temp_test_script = os.path.join(repo_path, "test_temp.py")
        with open(temp_test_script, 'w') as f:
            f.write(f"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable verbose logging for Triton kernel compilation
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["TRITON_PRINT_DEBUG"] = "1"

import torch
from fla.layers.quasar import QuasarAttention

def test_quasar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 1
    seq_len = {sequence_length}
    hidden_size = 512
    head_dim = 64
    num_heads = 8
    
    quasar = QuasarAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=True,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Warmup
    for _ in range(3):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    num_runs = 10
    start = time.time()
    
    for _ in range(num_runs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    tokens_per_sec = (batch_size * seq_len * num_runs) / elapsed
    
    print(f"RESULT: {{tokens_per_sec:.2f}}")
    return tokens_per_sec

if __name__ == "__main__":
    tps = test_quasar()
    print(f"Tokens/sec: {{tps:.2f}}")
""")
        
        try:
            result = subprocess.run(
                [sys.executable, temp_test_script],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output = result.stdout + result.stderr
            
            # Extract tokens/sec from output
            for line in output.split('\n'):
                if "RESULT:" in line:
                    tokens_per_sec = float(line.split("RESULT:")[1].strip())
                    print(f"[VALIDATOR] Test result: {tokens_per_sec:.2f} tokens/sec")
                    return tokens_per_sec
            
            print(f"[VALIDATOR] Could not parse test results: {output}")
            return 0.0
            
        except subprocess.TimeoutExpired:
            print(f"[VALIDATOR] Test timed out (300s)")
            return 0.0
        except Exception as e:
            print(f"[VALIDATOR] Test failed: {e}")
            return 0.0
        finally:
            # Clean up temp test script
            if os.path.exists(temp_test_script):
                os.remove(temp_test_script)
    
    def verify_performance(self, claimed: float, actual: float, tolerance: float = 0.1) -> bool:
        """Verify if actual performance is close to claimed performance."""
        if actual <= 0:
            return False
        
        # Calculate percentage difference
        diff = abs(claimed - actual) / claimed
        is_valid = diff <= tolerance
        
        print(f"[VALIDATOR] Performance verification:")
        print(f"  Claimed: {claimed:.2f} tokens/sec")
        print(f"  Actual: {actual:.2f} tokens/sec")
        print(f"  Difference: {diff:.2%}")
        print(f"  Valid: {is_valid}")
        
        return is_valid
    
    def validate_submission(self, submission: Dict) -> Dict:
        """Validate a single submission."""
        fork_url = submission.get("fork_url")
        commit_hash = submission.get("commit_hash")
        claimed_performance = submission.get("tokens_per_sec")
        target_sequence_length = submission.get("target_sequence_length", 100000)
        
        print(f"\n[VALIDATOR] Validating submission: {submission.get('id')}")
        print(f"  Fork URL: {fork_url}")
        print(f"  Commit: {commit_hash}")
        print(f"  Claimed performance: {claimed_performance:.2f} tokens/sec")
        
        try:
            # Clone the repository
            repo_path = self.clone_miner_repo(fork_url)
            
            # Run performance test
            actual_performance = self.run_performance_test(repo_path, target_sequence_length)
            
            # Verify performance
            is_valid = self.verify_performance(claimed_performance, actual_performance)
            
            # Clean up
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
            
            return {
                "submission_id": submission.get("id"),
                "miner_hotkey": submission.get("miner_hotkey"),
                "claimed_performance": claimed_performance,
                "actual_performance": actual_performance,
                "is_valid": is_valid,
                "fork_url": fork_url,
                "commit_hash": commit_hash
            }
            
        except Exception as e:
            print(f"[VALIDATOR] Validation failed: {e}")
            traceback.print_exc()
            return {
                "submission_id": submission.get("id"),
                "miner_hotkey": submission.get("miner_hotkey"),
                "claimed_performance": claimed_performance,
                "actual_performance": 0.0,
                "is_valid": False,
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"[VALIDATOR] Cleaned up temp dir: {self.temp_dir}")


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
        
        # Initialize PerformanceValidator for speed optimization validation
        self.performance_validator = PerformanceValidator()
        bt.logging.info("⚡ Performance validator initialized")
        
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

    def evaluate_performance_submissions(self) -> Dict[str, float]:
        """Evaluate performance submissions by cloning repos and running tests.
        
        Returns:
            Dictionary mapping miner_hotkey to score (0.0 to 1.0).
        """
        print(f"[VALIDATOR] ⚡ Evaluating performance submissions...", flush=True)
        bt.logging.info("⚡ Evaluating performance submissions...")
        
        evaluated_scores = {}  # hotkey -> score
        
        try:
            # Fetch pending submissions from API
            submissions = self.performance_validator.fetch_pending_submissions(limit=5)
            
            if not submissions:
                print("[VALIDATOR] No performance submissions to evaluate", flush=True)
                return evaluated_scores
            
            print(f"[VALIDATOR] Found {len(submissions)} performance submissions", flush=True)
            
            for submission in submissions:
                # Validate the submission
                result = self.performance_validator.validate_submission(submission)
                
                miner_hotkey = result.get("miner_hotkey")
                is_valid = result.get("is_valid", False)
                actual_performance = result.get("actual_performance", 0.0)
                
                # Score based on validity and performance
                if is_valid:
                    # Normalize performance to score (higher is better)
                    # Assuming max reasonable performance is around 200,000 tokens/sec
                    score = min(actual_performance / 200000.0, 1.0)
                    print(f"[VALIDATOR] ✅ Valid submission from {miner_hotkey[:12]}... - Score: {score:.4f}", flush=True)
                    evaluated_scores[miner_hotkey] = score
                else:
                    print(f"[VALIDATOR] ❌ Invalid submission from {miner_hotkey[:12]}...", flush=True)
                    evaluated_scores[miner_hotkey] = 0.0
            
            if evaluated_scores:
                print(f"[VALIDATOR] ✅ Evaluated {len(evaluated_scores)} performance submissions", flush=True)
                
        except Exception as e:
            print(f"[VALIDATOR] ⚠️ Failed to evaluate performance submissions: {e}", flush=True)
            bt.logging.warning(f"Failed to evaluate performance submissions: {e}")
            traceback.print_exc()
        
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

