# The MIT License (MIT)
# Copyright 2026 SILX INC

import os
import sys
import time
import asyncio
import subprocess
import tempfile
import torch
import numpy as np
import bittensor as bt
import traceback
import requests
import shutil
import json
from typing import List, Dict, Optional

# Add the parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import quasar
from quasar.base.validator import BaseValidatorNeuron

# --- Constants ---
VALIDATOR_API_URL = os.getenv("VALIDATOR_API_URL", "https://quasar-subnet.onrender.com")

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
    
    def checkout_commit(self, repo_path: str, commit_hash: str) -> None:
        try:
            subprocess.run(
                ["git", "checkout", commit_hash],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", commit_hash],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            subprocess.run(
                ["git", "checkout", commit_hash],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

    def run_performance_test(self, repo_path: str, sequence_length: int) -> Dict[str, float]:
        """Run performance test on cloned repository.

        Returns:
            Dict with keys: tokens_per_sec, vram_mb
        """
        print(f"[VALIDATOR] Running performance test (seq_len={sequence_length})...")
        
        # Create temporary test script with target sequence length
        temp_test_script = os.path.join(repo_path, f"test_temp_{sequence_length}.py")
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
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

    vram_bytes = 0
    if device.type == "cuda":
        vram_bytes = torch.cuda.max_memory_allocated()
    vram_mb = vram_bytes / (1024 * 1024)
    
    print(f"RESULT: {{tokens_per_sec:.2f}}")
    print(f"VRAM_MB: {{vram_mb:.2f}}")
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
            
            tokens_per_sec = 0.0
            vram_mb = 0.0
            for line in output.split('\n'):
                if "RESULT:" in line:
                    tokens_per_sec = float(line.split("RESULT:")[1].strip())
                if "VRAM_MB:" in line:
                    vram_mb = float(line.split("VRAM_MB:")[1].strip())
            
            if tokens_per_sec > 0:
                print(f"[VALIDATOR] Test result: {tokens_per_sec:.2f} tokens/sec | VRAM: {vram_mb:.2f} MB")
                return {"tokens_per_sec": tokens_per_sec, "vram_mb": vram_mb}
            
            print(f"[VALIDATOR] Could not parse test results: {output}")
            return {"tokens_per_sec": 0.0, "vram_mb": 0.0}
            
        except subprocess.TimeoutExpired:
            print(f"[VALIDATOR] Test timed out (300s)")
            return {"tokens_per_sec": 0.0, "vram_mb": 0.0}
        except Exception as e:
            print(f"[VALIDATOR] Test failed: {e}")
            return {"tokens_per_sec": 0.0, "vram_mb": 0.0}
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
        claimed_benchmarks_json = submission.get("benchmarks")

        # Parse claimed benchmarks if available
        claimed_benchmarks = {}
        if claimed_benchmarks_json:
            try:
                claimed_benchmarks = json.loads(claimed_benchmarks_json)
            except Exception as e:
                print(f"[VALIDATOR] Failed to parse benchmarks: {e}")

        print(f"\n[VALIDATOR] Validating submission: {submission.get('id')}")
        print(f"  Fork URL: {fork_url}")
        print(f"  Commit: {commit_hash}")
        print(f"  Claimed performance: {claimed_performance:.2f} tokens/sec @ seq_len={target_sequence_length}")
        if claimed_benchmarks:
            print(f"  Claimed benchmarks:")
            for seq_len, metrics in claimed_benchmarks.items():
                print(f"    {seq_len}: {metrics.get('tokens_per_sec', 0):.2f} tokens/sec | VRAM: {metrics.get('vram_mb', 0):.2f} MB")

        try:
            # Clone the repository
            repo_path = self.clone_miner_repo(fork_url)

            if commit_hash:
                self.checkout_commit(repo_path, commit_hash)

            # Run benchmarks for all reported sequence lengths
            seq_lengths_to_test = sorted(set([512, 1024, 2048, int(target_sequence_length)]))
            if claimed_benchmarks:
                seq_lengths_to_test = sorted(set(list(claimed_benchmarks.keys()) + [int(target_sequence_length)]))

            results_by_seq_len: Dict[int, Dict[str, float]] = {}
            for seq_len in seq_lengths_to_test:
                results_by_seq_len[seq_len] = self.run_performance_test(repo_path, seq_len)

            target_results = results_by_seq_len.get(int(target_sequence_length), {"tokens_per_sec": 0.0, "vram_mb": 0.0})
            actual_performance = float(target_results.get("tokens_per_sec", 0.0))

            # Calculate score: higher actual = higher rewards, lower actual = zero
            # If actual >= claimed * 0.9, give full reward (10% tolerance)
            # If actual < claimed * 0.9, give zero reward
            tolerance = 0.9  # 90% of claimed
            score = 0.0
            if actual_performance >= claimed_performance * tolerance:
                # Bonus for exceeding claimed performance
                score = 1.0 + (actual_performance - claimed_performance) / claimed_performance
            else:
                # Below tolerance, zero reward
                score = 0.0

            print(f"[VALIDATOR] Performance verification:")
            print(f"  Claimed: {claimed_performance:.2f} tokens/sec @ seq_len={target_sequence_length}")
            print(f"  Actual: {actual_performance:.2f} tokens/sec @ seq_len={target_sequence_length}")
            print(f"  Difference: {(actual_performance - claimed_performance) / claimed_performance * 100:.2f}%")
            print(f"  Score: {score:.4f} (higher actual = higher rewards)")

            # Compare all reported sequence lengths
            print(f"[VALIDATOR] Benchmark comparison:")
            for seq_len in sorted(claimed_benchmarks.keys()) if claimed_benchmarks else []:
                claimed = claimed_benchmarks.get(seq_len, {}).get("tokens_per_sec", 0)
                actual = results_by_seq_len.get(seq_len, {}).get("tokens_per_sec", 0)
                diff = (actual - claimed) / claimed * 100 if claimed > 0 else 0
                print(f"  {seq_len}: claimed={claimed:.2f}, actual={actual:.2f}, diff={diff:.2f}%")

            # Clean up
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)

            return {
                "submission_id": submission.get("id"),
                "miner_hotkey": submission.get("miner_hotkey"),
                "claimed_performance": claimed_performance,
                "actual_performance": actual_performance,
                "results_by_seq_len": results_by_seq_len,
                "score": score,
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
                "score": 0.0,
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
        
        # Initialize PerformanceValidator for speed optimization validation
        self.performance_validator = PerformanceValidator()
        bt.logging.info("⚡ Performance validator initialized")
        
        # Initialize scores
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)
        self.load_state()
        
        bt.logging.info(f"📡 Validator API URL: {VALIDATOR_API_URL}")

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
                # Skip already validated submissions
                if submission.get("validated", False):
                    continue

                # Validate the submission
                result = self.performance_validator.validate_submission(submission)

                miner_hotkey = result.get("miner_hotkey")
                score = result.get("score", 0.0)

                # Use the score from validate_submission (already calculated)
                # Normalize to 0-1 range (assuming max reasonable score is around 2.0)
                normalized_score = min(score / 2.0, 1.0)

                if score > 0:
                    print(f"[VALIDATOR] ✅ Valid submission from {miner_hotkey[:12]}... - Score: {score:.4f} (normalized: {normalized_score:.4f})", flush=True)
                    evaluated_scores[miner_hotkey] = normalized_score
                else:
                    print(f"[VALIDATOR] ❌ Invalid submission from {miner_hotkey[:12]}...", flush=True)
                    evaluated_scores[miner_hotkey] = 0.0

                # Mark submission as validated in API
                submission_id = submission.get("id")
                if submission_id:
                    try:
                        requests.post(
                            f"{VALIDATOR_API_URL}/mark_validated",
                            json={"submission_id": submission_id},
                            timeout=30
                        )
                    except Exception as e:
                        print(f"[VALIDATOR] Failed to mark submission as validated: {e}", flush=True)

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
        """Main validation loop - validate speed submissions by cloning miner repos and running performance tests."""
        print("[VALIDATOR] ➡️ Starting validation cycle...", flush=True)
        bt.logging.info("➡️ Starting validation cycle...")

        try:
            # Evaluate performance submissions
            print("[VALIDATOR] ⚡ Evaluating performance submissions...", flush=True)
            evaluated_scores = self.evaluate_performance_submissions()
            
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

