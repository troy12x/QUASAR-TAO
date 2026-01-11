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
import requests
from typing import List, Dict, Union

# Add the parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import quasar
from quasar.base.validator import BaseValidatorNeuron

# --- Constants ---
CHALLENGE_URL = os.getenv("CHALLENGE_URL", "http://localhost:8080")
TIMEOUT_SECS = 300

class Validator(BaseValidatorNeuron):
    """
    Simplified Validator for QUASAR-SUBNET.
    Evaluates miners by calling the challenge container.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("🚀 Initializing QUASAR Validator...")
        
        # Initialize scores
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)
        self.load_state()
        
        bt.logging.info(f"📡 Challenge URL: {CHALLENGE_URL}")

    def load_state(self):
        """Load validator state from disk."""
        try:
            state_path = self.config.neuron.full_path + "/state.pt"
            if os.path.exists(state_path):
                state = torch.load(state_path, weights_only=False)
                self.step = state.get("step", 0)
                self.scores = state.get("scores", self.scores).to(self.device)
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
        """Main validation loop - call challenge container to evaluate miners."""
        bt.logging.info("➡️ Starting validation cycle...")
        
        try:
            # Get serving miners
            miner_uids = [
                uid for uid in range(self.metagraph.n)
                if self.metagraph.axons[uid].is_serving and uid != self.uid
            ]
            
            if not miner_uids:
                bt.logging.warning("No serving miners found, skipping round")
                return
            
            bt.logging.info(f"🎯 Found {len(miner_uids)} serving miners")
            
            # Evaluate each miner by calling challenge container
            rewards = []
            for uid in miner_uids:
                miner_hotkey = self.metagraph.hotkeys[uid]
                score = await self.evaluate_miner(uid, miner_hotkey)
                rewards.append(score)
                
                # Update local scores
                self.scores[uid] = score
                
                bt.logging.info(f"  Miner {uid} ({miner_hotkey[:8]}...): score={score:.4f}")
            
            # Log summary
            avg_score = sum(rewards) / len(rewards) if rewards else 0.0
            bt.logging.success(f"✅ Evaluation complete: avg_score={avg_score:.4f}")
            
            # Submit weights to Bittensor
            await self.submit_weights(miner_uids)
            
            # Save state periodically
            if self.step % 10 == 0:
                self.save_state()
                
        except Exception as e:
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
        """Evaluate a miner by calling the challenge container."""
        
        # Get miner axon endpoint
        axon = self.metagraph.axons[uid]
        miner_endpoint = f"http://{axon.ip}:{axon.port}"
        
        # Prepare evaluation request
        request_data = {
            "request_id": f"req_{self.step}_{uid}_{int(time.time())}",
            "submission_id": f"sub_{uid}_{int(time.time())}",
            "participant_id": miner_hotkey,
            "data": {
                "miner_endpoint": miner_endpoint,
                "miner_uid": uid,
            },
            "metadata": {
                "validator_hotkey": self.wallet.hotkey.ss58_address,
                "validator_uid": self.uid,
            },
            "epoch": self.step,
            "deadline": int(time.time()) + TIMEOUT_SECS,
        }
        
        try:
            # Call challenge container
            response = requests.post(
                f"{CHALLENGE_URL}/evaluate",
                json=request_data,
                timeout=TIMEOUT_SECS
            )
            response.raise_for_status()
            
            result = response.json()
            score = result.get("score", 0.0)
            
            bt.logging.info(f"    Challenge response: success={result.get('success')}, score={score:.4f}")
            return score
            
        except requests.exceptions.Timeout:
            bt.logging.warning(f"    Timeout evaluating miner {uid}")
            return 0.0
        except requests.exceptions.RequestException as e:
            bt.logging.warning(f"    Error evaluating miner {uid}: {e}")
            return 0.0
        except Exception as e:
            bt.logging.error(f"    Unexpected error evaluating miner {uid}: {e}")
            return 0.0


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
    
    # Build config
    config = bt.config(args)
    
    # Run validator
    validator = Validator(config=config)
    
    bt.logging.info("🚀 Starting validator loop...")
    asyncio.run(validator.run())

