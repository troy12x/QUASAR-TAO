#!/usr/bin/env python3
# The MIT License (MIT)
# Copyright 2026 SILX INC
#
# QUASAR Inference Verification Subnet
# Based on const's qllm/quasar.py architecture

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              QUASAR - INFERENCE VERIFICATION SUBNET                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  COMMANDS                                                                    ║
║  ────────                                                                    ║
║  validate - Run the validator loop (continuous)                              ║
║  mine     - Submit a Docker image to the chain (commit-reveal)               ║
║  list     - Show all miner submissions                                       ║
║                                                                              ║
║  MECHANISM                                                                   ║
║  ─────────                                                                   ║
║  Miners submit Docker images exposing an inference server. Validators run   ║
║  containers via Basilica, measuring generation throughput while verifying   ║
║  correctness by comparing logits at a random decode step.                    ║
║                                                                              ║
║  Container interface: inference(prompt, gen_len, logits_at_step)             ║
║  → Returns: {tokens, captured_logits, elapsed_sec}                           ║
║                                                                              ║
║  Verification: cosine similarity + max absolute diff on captured logits      ║
║  Scoring: throughput (tok/sec) if verified, infinity if failed               ║
║  Leader: epsilon-dominance, winner-take-all weights                          ║
║                                                                              ║
║  Reference Model: Qwen/Qwen2.5-0.5B-Instruct (upgradeable to Quasar model)   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import asyncio
import click
from typing import Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quasar.inference_verification import (
    CONFIG,
    InferenceVerificationConfig,
    ReferenceModel,
    MinerEvaluation,
    evaluate_miner,
    select_leader,
    calculate_weights,
    log,
    log_header,
    estimate_reveal_time_minutes,
)

# Bittensor imports
try:
    import bittensor as bt
except ImportError:
    bt = None
    print("Warning: bittensor not installed. Some features will be unavailable.")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ STATE                                                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Global evaluation state (in-memory, could be persisted)
EVALUATED: Dict[str, MinerEvaluation] = {}


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ VALIDATOR LOOP                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

async def poll_and_evaluate(
    subtensor: "bt.AsyncSubtensor",
    block: int,
    reference: ReferenceModel
):
    """
    Poll for new revealed commitments and evaluate them.
    
    This function:
    1. Fetches all revealed commitments from the chain
    2. For each new/updated commitment, evaluates the miner
    3. Stores evaluation results in EVALUATED
    """
    global EVALUATED
    
    try:
        # Get all revealed commitments for our subnet
        commits = await subtensor.get_all_revealed_commitments(CONFIG.netuid, block=block)
    except Exception as e:
        log(f"Failed to fetch commitments: {e}", "error")
        return
    
    if not commits:
        return
    
    for hotkey, commit_data in commits.items():
        if not commit_data:
            continue
        
        # Get latest commitment (last in list)
        commit_block, docker_image = commit_data[-1]
        
        # Skip if already evaluated at this block
        if hotkey in EVALUATED and EVALUATED[hotkey].block == commit_block:
            continue
        
        log(f"New submission from {hotkey[:8]}... (block {commit_block})", "start")
        
        # Evaluate the miner
        result = await evaluate_miner(hotkey, docker_image, reference)
        result.block = commit_block
        
        # Store evaluation
        EVALUATED[hotkey] = result
        
        status = "success" if result.verified else "error"
        log(f"Evaluated {hotkey[:8]}...: verified={result.verified}, score={result.score:.4f}", status)


async def set_weights(
    subtensor: "bt.AsyncSubtensor",
    wallet: "bt.Wallet"
):
    """
    Set weights on the chain based on current evaluations.
    
    Uses winner-take-all: leader gets 100%, others get 0%.
    """
    global EVALUATED
    
    leader = select_leader(EVALUATED)
    if not leader:
        log("No valid leader found", "warn")
        return
    
    throughput = EVALUATED[leader].throughput
    log(f"Leader: {leader[:8]}... ({throughput:.1f} tok/sec)", "success")
    
    # Get metagraph for UID mapping
    try:
        metagraph = await subtensor.metagraph(CONFIG.netuid)
    except Exception as e:
        log(f"Failed to fetch metagraph: {e}", "error")
        return
    
    uids = []
    weights = []
    
    for uid, hotkey in enumerate(metagraph.hotkeys):
        if hotkey in EVALUATED:
            uids.append(uid)
            weights.append(1.0 if hotkey == leader else 0.0)
    
    if not uids:
        log("No UIDs to set weights for", "warn")
        return
    
    try:
        await subtensor.set_weights(
            wallet=wallet,
            netuid=CONFIG.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=True
        )
        log(f"Set weights for {len(uids)} miners (leader gets 100%)", "success")
    except Exception as e:
        log(f"Failed to set weights: {e}", "error")


async def validator_loop(
    wallet_name: str,
    hotkey_name: str,
    network: str
):
    """
    Main validator loop.
    
    Continuously:
    1. Polls for new revealed commitments
    2. Evaluates new miners
    3. Sets weights every TEMPO blocks
    """
    if bt is None:
        log("bittensor not installed - cannot run validator", "error")
        return
    
    # Initialize subtensor and wallet
    subtensor = bt.AsyncSubtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    
    log_header("QUASAR - Inference Verification Validator")
    log(f"Wallet: {wallet_name}/{hotkey_name}", "info")
    log(f"Network: {network}, NETUID: {CONFIG.netuid}", "info")
    log(f"Reference model: {CONFIG.reference_model}", "info")
    
    # Load reference model
    reference = ReferenceModel(CONFIG.reference_model)
    await reference.load()
    
    last_weight_block = 0
    
    while True:
        try:
            current_block = await subtensor.get_current_block()
            
            # Poll and evaluate new submissions
            await poll_and_evaluate(subtensor, current_block, reference)
            
            # Set weights every TEMPO blocks
            if current_block - last_weight_block >= CONFIG.tempo:
                log_header(f"Weight Update (block {current_block})")
                await set_weights(subtensor, wallet)
                last_weight_block = current_block
            
            # Sleep for one block
            await asyncio.sleep(CONFIG.block_time)
            
        except Exception as e:
            log(f"Error in validator loop: {e}", "error")
            await asyncio.sleep(CONFIG.block_time)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ MINER SUBMISSION                                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

async def submit_docker_image(
    wallet_name: str,
    hotkey_name: str,
    network: str,
    docker_image: str
):
    """
    Submit a Docker image to the chain using commit-reveal.
    
    The image will be hidden for BLOCKS_UNTIL_REVEAL blocks, then
    automatically revealed for validators to evaluate.
    """
    if bt is None:
        log("bittensor not installed - cannot submit", "error")
        return
    
    subtensor = bt.AsyncSubtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    
    log_header("Submit Miner Docker Image")
    log(f"Wallet: {wallet_name}/{hotkey_name}", "info")
    log(f"Network: {network}, NETUID: {CONFIG.netuid}", "info")
    log(f"Image: {docker_image}", "info")
    
    # Check registration
    try:
        is_registered = await subtensor.is_hotkey_registered(
            CONFIG.netuid,
            wallet.hotkey.ss58_address
        )
    except Exception as e:
        log(f"Failed to check registration: {e}", "error")
        return
    
    if not is_registered:
        log(f"Hotkey {hotkey_name} not registered on subnet {CONFIG.netuid}", "error")
        log(f"Register first: btcli subnet register --netuid {CONFIG.netuid}", "info")
        return
    
    log("Submitting commit-reveal commitment...", "start")
    
    try:
        await subtensor.set_reveal_commitment(
            wallet=wallet,
            netuid=CONFIG.netuid,
            data=docker_image,
            blocks_until_reveal=CONFIG.blocks_until_reveal,
            block_time=CONFIG.block_time
        )
        
        reveal_minutes = estimate_reveal_time_minutes()
        log(f"Commitment submitted! Will reveal in ~{reveal_minutes} minutes", "success")
        log(f"Image: {docker_image}", "info")
        
    except Exception as e:
        log(f"Failed to submit commitment: {e}", "error")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ LIST MINERS                                                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

async def list_miner_submissions(network: str):
    """
    List all miner Docker image submissions on the chain.
    """
    if bt is None:
        log("bittensor not installed - cannot list", "error")
        return
    
    subtensor = bt.AsyncSubtensor(network=network)
    
    log_header(f"Miner Submissions - Subnet {CONFIG.netuid}")
    log(f"Network: {network}", "info")
    
    try:
        current_block = await subtensor.get_current_block()
        commits = await subtensor.get_all_revealed_commitments(
            CONFIG.netuid,
            block=current_block
        )
    except Exception as e:
        log(f"Failed to fetch commitments: {e}", "error")
        return
    
    if not commits:
        log("No revealed commitments found", "warn")
        return
    
    try:
        metagraph = await subtensor.metagraph(CONFIG.netuid)
        hotkey_to_uid = {hk: uid for uid, hk in enumerate(metagraph.hotkeys)}
    except Exception as e:
        log(f"Failed to fetch metagraph: {e}", "error")
        hotkey_to_uid = {}
    
    # Table header
    print()
    print(f"{'UID':>5} {'Hotkey':<12} {'Block':>8} {'Image':<50}")
    print(f"{'─' * 5} {'─' * 12} {'─' * 8} {'─' * 50}")
    
    # Sort by block (most recent first)
    sorted_commits = sorted(
        commits.items(),
        key=lambda x: x[1][-1][0] if x[1] else 0,
        reverse=True
    )
    
    for hotkey, commit_data in sorted_commits:
        if not commit_data:
            continue
        
        commit_block, docker_image = commit_data[-1]
        uid = hotkey_to_uid.get(hotkey, "?")
        short_hk = f"{hotkey[:8]}..."
        
        # Truncate long image names
        if len(docker_image) > 50:
            docker_image = docker_image[:47] + "..."
        
        print(f"{uid:>5} {short_hk:<12} {commit_block:>8} {docker_image:<50}")
    
    print()
    log(f"Total: {len(commits)} miners with submissions", "info")
    log(f"Current block: {current_block}", "info")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ CLI COMMANDS                                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@click.group()
def cli():
    """Quasar - Inference Verification Subnet"""
    pass


@cli.command()
@click.option("--wallet", default=CONFIG.default_wallet, help="Wallet name")
@click.option("--hotkey", default=CONFIG.default_hotkey, help="Hotkey name")
@click.option("--network", default=CONFIG.default_network, help="Network (finney/test/local)")
def validate(wallet: str, hotkey: str, network: str):
    """Run the validator loop (continuous)."""
    asyncio.run(validator_loop(wallet, hotkey, network))


@cli.command()
@click.option("--wallet", default=CONFIG.default_wallet, help="Wallet name")
@click.option("--hotkey", default=CONFIG.default_hotkey, help="Hotkey name")
@click.option("--network", default=CONFIG.default_network, help="Network (finney/test/local)")
@click.option("--image", prompt="Docker Hub image", help="Docker image (e.g. user/repo:tag)")
def mine(wallet: str, hotkey: str, network: str, image: str):
    """Submit a Docker image to the chain (commit-reveal)."""
    asyncio.run(submit_docker_image(wallet, hotkey, network, image))


@cli.command("list")
@click.option("--network", default=CONFIG.default_network, help="Network (finney/test/local)")
def list_miners(network: str):
    """Show all miner Docker submissions."""
    asyncio.run(list_miner_submissions(network))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ ADDITIONAL COMMANDS FOR TESTING                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@cli.command()
def test_reference():
    """Test the reference model locally."""
    import random
    
    async def _test():
        log_header("Testing Reference Model")
        log(f"Model: {CONFIG.reference_model}", "info")
        
        reference = ReferenceModel(CONFIG.reference_model)
        await reference.load()
        
        # Generate test challenge
        vocab_size = reference.get_vocab_size() or 32000
        prompt = [random.randint(10, vocab_size - 1) for _ in range(CONFIG.prompt_length)]
        gen_len = 50  # Short for testing
        logits_at_step = 10
        
        log(f"Running inference: prompt_len={len(prompt)}, gen_len={gen_len}, capture_step={logits_at_step}", "start")
        
        result = await reference.inference(prompt, gen_len, logits_at_step)
        
        log(f"Generated {len(result['tokens'])} tokens in {result['elapsed_sec']:.2f}s", "success")
        log(f"Throughput: {gen_len / result['elapsed_sec']:.1f} tok/sec", "info")
        
        if result['captured_logits']:
            log(f"Captured logits at step {logits_at_step}: shape={len(result['captured_logits'])}", "info")
        else:
            log("No logits captured!", "error")
    
    asyncio.run(_test())


@cli.command()
@click.argument("logits_file_1")
@click.argument("logits_file_2")
def verify_logits(logits_file_1: str, logits_file_2: str):
    """Verify two logit files against each other (for testing)."""
    import json
    from quasar.inference_verification import verify_logits as _verify
    
    with open(logits_file_1) as f:
        logits_1 = json.load(f)
    
    with open(logits_file_2) as f:
        logits_2 = json.load(f)
    
    result = _verify(logits_1, logits_2)
    
    log_header("Logit Verification Result")
    log(f"Verified: {result.verified}", "success" if result.verified else "error")
    log(f"Cosine Similarity: {result.cosine_sim:.6f}", "info")
    log(f"Max Absolute Diff: {result.max_abs_diff:.6f}", "info")
    
    if result.reason:
        log(f"Reason: {result.reason}", "warn")


if __name__ == "__main__":
    cli()
