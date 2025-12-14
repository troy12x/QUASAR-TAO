import bittensor as bt
import os

print("🏗️ Creating wallet 'miner' with hotkey 'default'...")

wallet = bt.Wallet(name='miner', hotkey='default')

# Just use the standard create methods with use_password=False
# This generates a NEW mnemonic/keypair automatically.
print("Creating coldkey...")
wallet.create_new_coldkey(use_password=False, overwrite=True)

print("Creating hotkey...")
wallet.create_new_hotkey(use_password=False, overwrite=True)

print("✅ Wallet created successfully!")
print(f"Path: {wallet.path}")
print(f"Coldkey: {wallet.coldkey.ss58_address}")
print(f"Hotkey: {wallet.hotkey.ss58_address}")
