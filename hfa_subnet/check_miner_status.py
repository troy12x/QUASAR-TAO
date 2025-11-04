#!/usr/bin/env python3
"""
Quick diagnostic to see where miner is stuck
"""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bittensor as bt
import time

print("=" * 80)
print("MINER DIAGNOSTIC - Testing each step")
print("=" * 80)

# Test 1: Wallet
print("\n1. Testing wallet loading...")
try:
    import argparse
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    
    config = bt.config(parser=parser, args=[
        '--wallet.name', 'hfa_silx',
        '--wallet.hotkey', 'hfa_silx_hot',
        '--subtensor.network', 'test',
        '--netuid', '439'
    ])
    
    wallet = bt.wallet(config=config)
    print(f"✅ Wallet loaded: {wallet}")
except Exception as e:
    print(f"❌ Wallet failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Subtensor connection
print("\n2. Testing subtensor connection...")
try:
    print(f"   Connecting to {config.subtensor.network}...")
    start = time.time()
    subtensor = bt.subtensor(config=config)
    elapsed = time.time() - start
    print(f"✅ Subtensor connected in {elapsed:.2f}s")
    print(f"   Chain: {subtensor.chain_endpoint}")
except Exception as e:
    print(f"❌ Subtensor failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Metagraph download
print("\n3. Testing metagraph download...")
try:
    netuid = 439
    print(f"   Downloading metagraph for netuid {netuid}...")
    print(f"   This can take 30-120 seconds on first run...")
    start = time.time()
    
    # Show progress while downloading
    import threading
    downloading = True
    
    def show_progress():
        dots = 0
        while downloading:
            print(f"\r   Still downloading... {int(time.time() - start)}s elapsed" + "." * (dots % 4) + "   ", end='', flush=True)
            dots += 1
            time.sleep(1)
    
    progress_thread = threading.Thread(target=show_progress)
    progress_thread.start()
    
    metagraph = subtensor.metagraph(netuid)
    downloading = False
    progress_thread.join()
    
    elapsed = time.time() - start
    print(f"\n✅ Metagraph downloaded in {elapsed:.2f}s")
    print(f"   Neurons in subnet: {len(metagraph.hotkeys)}")
except Exception as e:
    downloading = False
    print(f"\n❌ Metagraph failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Registration check
print("\n4. Testing registration check...")
try:
    is_registered = subtensor.is_hotkey_registered(
        netuid=netuid,
        hotkey_ss58=wallet.hotkey.ss58_address,
    )
    if is_registered:
        print(f"✅ Hotkey is registered on netuid {netuid}")
    else:
        print(f"❌ Hotkey NOT registered on netuid {netuid}")
        print(f"   Register with: btcli subnet register --netuid {netuid}")
except Exception as e:
    print(f"❌ Registration check failed: {e}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print("\nIf all steps passed, your miner should work.")
print("The slowest step is usually metagraph download (step 3).")
