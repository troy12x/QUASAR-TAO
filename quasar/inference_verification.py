# The MIT License (MIT)
# Copyright 2026 SILX INC
#
# Inference Verification Module for QUASAR-SUBNET
# Based on const's qllm/quasar.py architecture
#
# This module implements:
# - Reference model for logit comparison
# - Logit verification (cosine similarity + max absolute diff)
# - Commit-reveal mechanism for Docker image submissions
# - Throughput-based scoring with verification gate

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QUASAR - INFERENCE VERIFICATION                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MECHANISM                                                                   ║
║  ─────────                                                                   ║
║  Miners submit Docker images exposing an inference server. Validators run   ║
║  containers, measuring generation throughput while verifying correctness    ║
║  by comparing logits at a random decode step.                               ║
║                                                                              ║
║  Container interface: inference(prompt, gen_len, logits_at_step)            ║
║  → Returns: {tokens, captured_logits, elapsed_sec}                          ║
║                                                                              ║
║  Verification: cosine similarity + max absolute diff on captured logits     ║
║  Scoring: throughput (tok/sec) if verified, infinity if failed              ║
║  Leader: epsilon-dominance, winner-take-all weights                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import asyncio
import random
import numpy as np
import torch
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ CONFIGURATION                                                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class InferenceVerificationConfig:
    """Configuration for inference verification."""
    
    # Network configuration
    netuid: int = int(os.environ.get("NETUID", 24))
    
    # Reference model (Qwen/Qwen2.5-0.5B-Instruct as specified)
    reference_model: str = os.environ.get("REFERENCE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    
    # Inference configuration
    prompt_length: int = 128  # Random prompt token length
    generate_length: int = 512  # Number of tokens to generate
    logit_capture_range: Tuple[int, int] = (1, 50)  # Range for random logit capture step
    inference_timeout: int = 300  # Timeout in seconds
    
    # Verification thresholds (from const's implementation)
    cosine_sim_threshold: float = 0.99  # Minimum cosine similarity
    max_abs_diff_threshold: float = 0.1  # Maximum absolute difference
    
    # Commit-reveal timing
    blocks_until_reveal: int = 100  # ~20 minutes (100 blocks * 12s/block)
    block_time: int = 12  # Bittensor block time in seconds
    
    # Scoring parameters
    epsilon: float = 0.01  # Epsilon for dominance comparison
    tempo: int = 360  # Weight update interval in blocks
    
    # Defaults
    default_wallet: str = os.environ.get("WALLET_NAME", "default")
    default_hotkey: str = os.environ.get("HOTKEY_NAME", "default")
    default_network: str = os.environ.get("NETWORK", "finney")


# Global config instance
CONFIG = InferenceVerificationConfig()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ LOGGING                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def log(msg: str, level: str = "info"):
    """Colored logging output."""
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {
        "info": "\033[36m▸\033[0m",      # Cyan
        "success": "\033[32m✓\033[0m",   # Green
        "error": "\033[31m✗\033[0m",     # Red
        "warn": "\033[33m⚠\033[0m",      # Yellow
        "start": "\033[33m→\033[0m",     # Yellow arrow
    }
    print(f"\033[90m{ts}\033[0m {colors.get(level, ' ')} {msg}")


def log_header(title: str):
    """Log a section header."""
    print(f"\n\033[1m{'─' * 60}\033[0m")
    print(f"\033[1m{title}\033[0m")
    print(f"\033[1m{'─' * 60}\033[0m\n")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ VERIFICATION                                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class VerificationResult:
    """Result of logit verification."""
    verified: bool
    cosine_sim: Optional[float] = None
    max_abs_diff: Optional[float] = None
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "cosine_sim": self.cosine_sim,
            "max_abs_diff": self.max_abs_diff,
            "reason": self.reason
        }


def verify_logits(
    miner_logits: List[float], 
    reference_logits: List[float],
    cosine_threshold: float = CONFIG.cosine_sim_threshold,
    max_diff_threshold: float = CONFIG.max_abs_diff_threshold
) -> VerificationResult:
    """
    Compare miner's logits against reference within tolerance.
    
    Uses cosine similarity + max absolute difference as verification metrics.
    This handles numerical instability that would cause divergence with
    direct logit comparison.
    
    Args:
        miner_logits: Logits returned by miner at the capture step
        reference_logits: Logits from reference model at same step
        cosine_threshold: Minimum cosine similarity required (default: 0.99)
        max_diff_threshold: Maximum absolute difference allowed (default: 0.1)
    
    Returns:
        VerificationResult with verified status and metrics
    """
    miner = np.array(miner_logits, dtype=np.float32)
    reference = np.array(reference_logits, dtype=np.float32)
    
    # Check shape match
    if miner.shape != reference.shape:
        return VerificationResult(
            verified=False,
            reason=f"shape_mismatch: miner={miner.shape}, reference={reference.shape}"
        )
    
    # Check for zero norms (would cause division by zero)
    norm_m = np.linalg.norm(miner)
    norm_r = np.linalg.norm(reference)
    
    if norm_m < 1e-9 or norm_r < 1e-9:
        return VerificationResult(
            verified=False,
            reason="zero_norm: logit vectors have near-zero norm"
        )
    
    # Calculate cosine similarity
    cosine_sim = float(np.dot(miner, reference) / (norm_m * norm_r))
    
    # Calculate max absolute difference
    max_abs_diff = float(np.max(np.abs(miner - reference)))
    
    # Verify against thresholds
    verified = (cosine_sim >= cosine_threshold) and (max_abs_diff <= max_diff_threshold)
    
    return VerificationResult(
        verified=verified,
        cosine_sim=cosine_sim,
        max_abs_diff=max_abs_diff,
        reason=None if verified else f"threshold_failed: cosine={cosine_sim:.4f}, max_diff={max_abs_diff:.4f}"
    )


def compute_kl_divergence(
    miner_logits: List[float],
    reference_logits: List[float],
    temperature: float = 1.0
) -> float:
    """
    Compute KL divergence between miner and reference logit distributions.
    
    Alternative verification method mentioned by const:
    "validator gives zero score if KL > epsilon"
    
    Args:
        miner_logits: Logits from miner
        reference_logits: Logits from reference model
        temperature: Temperature for softmax (default: 1.0)
    
    Returns:
        KL divergence value (lower is better, 0 = identical)
    """
    import torch.nn.functional as F
    
    miner_t = torch.tensor(miner_logits, dtype=torch.float32)
    reference_t = torch.tensor(reference_logits, dtype=torch.float32)
    
    # Apply temperature and softmax
    miner_probs = F.softmax(miner_t / temperature, dim=-1)
    reference_probs = F.softmax(reference_t / temperature, dim=-1)
    
    # KL divergence: D_KL(reference || miner)
    kl_div = F.kl_div(
        miner_probs.log(),
        reference_probs,
        reduction='sum'
    )
    
    return float(kl_div)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ REFERENCE MODEL                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class ReferenceModel:
    """
    Reference model for inference verification.
    
    Runs the honest base model (Qwen/Qwen2.5-0.5B-Instruct) to produce
    ground-truth logits for comparison with miner outputs.
    """
    
    def __init__(self, model_name: str = CONFIG.reference_model):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False
    
    async def load(self):
        """Load the reference model and tokenizer."""
        if self._loaded:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        log(f"Loading reference model: {self.model_name}", "info")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        self.model.eval()
        self._loaded = True
        
        log(f"Reference model loaded on {self.device}", "success")
    
    async def inference(
        self, 
        prompt: List[int], 
        gen_len: int, 
        logits_at_step: int
    ) -> Dict[str, Any]:
        """
        Run inference and capture logits at a specific step.
        
        This is the core of the verification mechanism. The validator runs
        the same inference as the miner and compares logits at a random step.
        
        Args:
            prompt: Input token IDs (random tokens for verification)
            gen_len: Number of tokens to generate
            logits_at_step: Step at which to capture logits (1-indexed)
        
        Returns:
            Dict with:
                - tokens: List of generated token IDs
                - captured_logits: Logits at the specified step
                - elapsed_sec: Time taken for inference
        """
        if not self._loaded:
            await self.load()
        
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt], device=device)
        
        generated_tokens = []
        captured_logits = None
        past_key_values = None
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # Initial forward pass (prefill)
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Autoregressive generation
            for step in range(gen_len):
                # Capture logits at the specified step (1-indexed)
                if step + 1 == logits_at_step:
                    captured_logits = next_token_logits[0].cpu().float().tolist()
                
                # Greedy decoding (argmax)
                next_token = next_token_logits.argmax(dim=-1)
                generated_tokens.append(next_token.item())
                
                # Forward pass with KV cache
                outputs = self.model(
                    next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
        
        elapsed_sec = time.perf_counter() - start_time
        
        return {
            "tokens": generated_tokens,
            "captured_logits": captured_logits,
            "elapsed_sec": elapsed_sec
        }
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size for logit dimension verification."""
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ PROMPT GENERATION                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def generate_random_prompt(length: int, vocab_size: int = 32000) -> List[int]:
    """
    Generate a random prompt for inference verification.
    
    Using random tokens ensures miners can't pre-compute responses
    and must actually run the model.
    
    Args:
        length: Number of tokens in the prompt
        vocab_size: Vocabulary size to sample from
    
    Returns:
        List of random token IDs
    """
    # Avoid special tokens (typically 0-10 range)
    return [random.randint(10, vocab_size - 1) for _ in range(length)]


def generate_verification_challenge(
    reference_model: ReferenceModel,
    config: InferenceVerificationConfig = CONFIG
) -> Dict[str, Any]:
    """
    Generate a verification challenge for a miner.
    
    Returns:
        Dict with:
            - prompt: Random token IDs
            - gen_len: Generation length
            - logits_at_step: Step to capture logits
    """
    vocab_size = reference_model.get_vocab_size() or 32000
    
    return {
        "prompt": generate_random_prompt(config.prompt_length, vocab_size),
        "gen_len": config.generate_length,
        "logits_at_step": random.randint(*config.logit_capture_range)
    }


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ CONTAINER EXECUTION                                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class ContainerInferenceResult:
    """Result from running inference in a miner's container."""
    success: bool
    tokens: List[int] = field(default_factory=list)
    captured_logits: Optional[List[float]] = None
    elapsed_sec: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tokens": self.tokens,
            "captured_logits": self.captured_logits,
            "elapsed_sec": self.elapsed_sec,
            "error": self.error
        }


async def run_container_inference(
    hotkey: str,
    docker_image: str,
    prompt: List[int],
    gen_len: int,
    logits_at_step: int,
    timeout: int = CONFIG.inference_timeout
) -> ContainerInferenceResult:
    """
    Run inference on miner's Docker container.
    
    This would use affinetes/Basilica to execute the container.
    For now, this is a placeholder that should be integrated with
    the actual container execution system.
    
    Container interface expected:
        inference(prompt: List[int], gen_len: int, logits_at_step: int)
        → Returns: {tokens: List[int], captured_logits: List[float], elapsed_sec: float}
    
    Args:
        hotkey: Miner's hotkey for identification
        docker_image: Docker image to run
        prompt: Input token IDs
        gen_len: Number of tokens to generate
        logits_at_step: Step at which to capture logits
        timeout: Timeout in seconds
    
    Returns:
        ContainerInferenceResult with inference results or error
    """
    env = None
    try:
        # Try to import affinetes for container execution
        try:
            import affinetes as af
        except ImportError:
            log(f"affinetes not installed - using mock container execution", "warn")
            # Return mock result for testing
            return ContainerInferenceResult(
                success=False,
                error="affinetes not installed - container execution not available"
            )
        
        # Load container environment
        env = af.load_env(image=docker_image)
        
        # Call inference function in container
        result = await env.inference(
            prompt=prompt,
            gen_len=gen_len,
            logits_at_step=logits_at_step,
            _timeout=timeout
        )
        
        return ContainerInferenceResult(
            success=True,
            tokens=result.get("tokens", []),
            captured_logits=result.get("captured_logits"),
            elapsed_sec=float(result.get("elapsed_sec", 0))
        )
        
    except asyncio.TimeoutError:
        log(f"Container timeout for {hotkey[:8]}...", "error")
        return ContainerInferenceResult(
            success=False,
            error=f"timeout after {timeout}s"
        )
    except Exception as e:
        log(f"Container failed for {hotkey[:8]}...: {e}", "error")
        return ContainerInferenceResult(
            success=False,
            error=str(e)
        )
    finally:
        if env:
            try:
                await env.cleanup()
            except Exception:
                pass


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ SCORING & LEADER SELECTION                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class MinerEvaluation:
    """Evaluation result for a single miner."""
    hotkey: str
    block: int
    docker_image: str
    score: float  # 1/throughput (lower is better) or inf if failed
    verified: bool
    throughput: float = 0.0  # tokens/sec
    verification: Optional[VerificationResult] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hotkey": self.hotkey,
            "block": self.block,
            "docker_image": self.docker_image,
            "score": self.score,
            "verified": self.verified,
            "throughput": self.throughput,
            "verification": self.verification.to_dict() if self.verification else None,
            "error": self.error
        }


async def evaluate_miner(
    hotkey: str,
    docker_image: str,
    reference: ReferenceModel,
    config: InferenceVerificationConfig = CONFIG
) -> MinerEvaluation:
    """
    Evaluate a miner: verify correctness + measure throughput.
    
    This is the core evaluation function that:
    1. Generates a random prompt
    2. Runs inference on miner's container
    3. Runs inference on reference model
    4. Compares logits at random step
    5. Calculates throughput if verified
    
    Args:
        hotkey: Miner's hotkey
        docker_image: Miner's Docker image
        reference: Reference model instance
        config: Configuration
    
    Returns:
        MinerEvaluation with score and verification results
    """
    # Generate challenge
    challenge = generate_verification_challenge(reference, config)
    prompt = challenge["prompt"]
    gen_len = challenge["gen_len"]
    logits_at_step = challenge["logits_at_step"]
    
    log(f"  prompt_len={len(prompt)}, gen_len={gen_len}, capture_step={logits_at_step}", "info")
    
    # Run miner inference
    miner_result = await run_container_inference(
        hotkey, docker_image, prompt, gen_len, logits_at_step
    )
    
    if not miner_result.success:
        return MinerEvaluation(
            hotkey=hotkey,
            block=0,  # Will be set by caller
            docker_image=docker_image,
            score=float("inf"),
            verified=False,
            error=miner_result.error
        )
    
    # Run reference model inference
    reference_result = await reference.inference(prompt, gen_len, logits_at_step)
    
    # Check if miner returned logits
    if miner_result.captured_logits is None:
        log("  Miner did not return captured logits", "error")
        return MinerEvaluation(
            hotkey=hotkey,
            block=0,
            docker_image=docker_image,
            score=float("inf"),
            verified=False,
            error="no_logits"
        )
    
    # Verify logits
    verification = verify_logits(
        miner_result.captured_logits,
        reference_result["captured_logits"]
    )
    
    log(f"  cosine={verification.cosine_sim:.4f}, max_diff={verification.max_abs_diff:.4f}", "info")
    
    if not verification.verified:
        log("  FAILED verification", "error")
        return MinerEvaluation(
            hotkey=hotkey,
            block=0,
            docker_image=docker_image,
            score=float("inf"),
            verified=False,
            verification=verification
        )
    
    # Calculate throughput
    elapsed = miner_result.elapsed_sec
    if elapsed <= 0:
        return MinerEvaluation(
            hotkey=hotkey,
            block=0,
            docker_image=docker_image,
            score=float("inf"),
            verified=True,
            error="invalid_elapsed"
        )
    
    throughput = gen_len / elapsed
    score = 1.0 / throughput if throughput > 0 else float("inf")
    
    log(f"  Throughput: {throughput:.1f} tok/sec", "success")
    
    return MinerEvaluation(
        hotkey=hotkey,
        block=0,
        docker_image=docker_image,
        score=score,
        verified=True,
        throughput=throughput,
        verification=verification
    )


def beats(
    evaluations: Dict[str, MinerEvaluation],
    i: str,
    j: str,
    epsilon: float = CONFIG.epsilon
) -> bool:
    """
    Check if miner i beats miner j with epsilon-dominance.
    
    Args:
        evaluations: Dictionary of miner evaluations
        i: Hotkey of miner i
        j: Hotkey of miner j
        epsilon: Epsilon for dominance comparison
    
    Returns:
        True if i beats j
    """
    if i not in evaluations or j not in evaluations:
        return False
    return evaluations[i].score < evaluations[j].score - epsilon


def select_leader(
    evaluations: Dict[str, MinerEvaluation],
    epsilon: float = CONFIG.epsilon
) -> Optional[str]:
    """
    Select leader using epsilon-dominance.
    
    The leader is the miner that:
    1. Is verified (passed logit check)
    2. Is not dominated by any other verified miner
    3. Has the earliest submission block (tie-breaker)
    
    Args:
        evaluations: Dictionary of miner evaluations
        epsilon: Epsilon for dominance comparison
    
    Returns:
        Hotkey of the leader, or None if no valid leader
    """
    if not evaluations:
        return None
    
    # Filter to verified miners only
    verified = [hk for hk in evaluations if evaluations[hk].verified]
    if not verified:
        return None
    
    # Find non-dominated candidates
    candidates = []
    for hk in verified:
        # Check if any other miner beats this one
        dominated = any(beats(evaluations, other, hk, epsilon) for other in verified if other != hk)
        if not dominated:
            candidates.append(hk)
    
    if not candidates:
        return None
    
    # Tie-breaker: earliest submission block
    return min(candidates, key=lambda hk: evaluations[hk].block)


def calculate_weights(
    evaluations: Dict[str, MinerEvaluation],
    hotkeys: List[str]
) -> Dict[str, float]:
    """
    Calculate weights for all miners (winner-take-all).
    
    The leader gets 100% of the weight, everyone else gets 0%.
    
    Args:
        evaluations: Dictionary of miner evaluations
        hotkeys: List of all miner hotkeys in the metagraph
    
    Returns:
        Dictionary mapping hotkey to weight (0.0 or 1.0)
    """
    leader = select_leader(evaluations)
    
    weights = {}
    for hk in hotkeys:
        if hk in evaluations:
            weights[hk] = 1.0 if hk == leader else 0.0
        else:
            weights[hk] = 0.0
    
    return weights


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ COMMIT-REVEAL HELPERS                                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def compute_commitment_hash(data: str, salt: bytes = None) -> str:
    """
    Compute commitment hash for commit-reveal scheme.
    
    Args:
        data: Data to commit (e.g., Docker image name)
        salt: Random salt for hiding commitment
    
    Returns:
        Hex-encoded hash
    """
    if salt is None:
        salt = os.urandom(32)
    
    content = salt + data.encode()
    return hashlib.sha256(content).hexdigest()


def get_reveal_block(current_block: int, blocks_until_reveal: int = CONFIG.blocks_until_reveal) -> int:
    """
    Calculate the block at which commitment will be revealed.
    
    Args:
        current_block: Current block number
        blocks_until_reveal: Number of blocks to wait
    
    Returns:
        Block number for reveal
    """
    return current_block + blocks_until_reveal


def estimate_reveal_time_minutes(blocks_until_reveal: int = CONFIG.blocks_until_reveal) -> int:
    """
    Estimate time until reveal in minutes.
    
    Args:
        blocks_until_reveal: Number of blocks to wait
    
    Returns:
        Estimated minutes until reveal
    """
    return (blocks_until_reveal * CONFIG.block_time) // 60
