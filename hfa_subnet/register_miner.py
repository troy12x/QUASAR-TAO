import bittensor as bt
import sys

# Hardcoded for the user's specific case based on previous context
wallet_name = "miner"
hotkey_name = "default"
netuid = 1

print(f"Attempting to register wallet '{wallet_name}:{hotkey_name}' on netuid {netuid}...")

try:
    subtensor = bt.Subtensor() # Defaults to Finney/Mainnet usually, or whatever is configured
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    
    print(f"Wallet: {wallet}")
    print(f"Subtensor: {subtensor}")
    
    # Check balance
    balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
    print(f"Current Balance: {balance}")
    
    # Register
    print("Registering... (this will burn/recycle TAO if on mainnet)")
    success = subtensor.register(
        wallet=wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        prompt=True # It might prompt in the terminal
    )
    
    if success:
        print("✅ Registration Successful!")
    else:
        print("❌ Registration Failed.")
        
except Exception as e:
    print(f"Error during registration: {e}")
    print("\nIf you do not have TAO to register, please run the miner in MOCK mode:")
    print("python neurons/miner.py --wallet.name miner --wallet.hotkey default --mock")
