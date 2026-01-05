import bittensor as bt
from fastapi import Request, HTTPException, Depends
from typing import Optional

# Subnet configuration
SUBNET_NETUID = 24  # Quasar subnet

# Cache for metagraph to avoid frequent API calls
_metagraph_cache = None
_metagraph_cache_time = 0
METAGRAPH_CACHE_TTL = 300  # 5 minutes

def get_metagraph():
    """Get cached metagraph or fetch fresh one."""
    global _metagraph_cache, _metagraph_cache_time
    import time
    
    current_time = time.time()
    if _metagraph_cache is None or (current_time - _metagraph_cache_time) > METAGRAPH_CACHE_TTL:
        try:
            _metagraph_cache = bt.metagraph(SUBNET_NETUID)
            _metagraph_cache_time = current_time
        except Exception as e:
            # If we can't fetch metagraph, use cached version if available
            if _metagraph_cache is None:
                raise HTTPException(status_code=503, detail=f"Failed to fetch metagraph: {str(e)}")
    
    return _metagraph_cache

def verify_signature(request: Request):
    """
    Middleware/Dependency to verify Bittensor signatures and subnet registration.
    
    Expects headers:
    - 'Hotkey': The SS58 address of the hotkey.
    - 'Signature': The signature of the hotkey (signing the hotkey itself for simplicity, or a timestamp).
    
    Also verifies that the hotkey is registered on subnet 24.
    """
    hotkey = request.headers.get("Hotkey")
    signature = request.headers.get("Signature")

    if not hotkey or not signature:
        raise HTTPException(status_code=401, detail="Missing Hotkey or Signature headers")

    try:
        # 1. Verify signature
        keypair = bt.Keypair(ss58_address=hotkey)
        if not keypair.verify(hotkey, signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        # 2. Verify hotkey is registered on subnet 24
        metagraph = get_metagraph()
        
        # Check if hotkey exists in metagraph
        if hotkey not in metagraph.hotkeys:
            raise HTTPException(
                status_code=403, 
                detail=f"Hotkey {hotkey} is not registered on subnet {SUBNET_NETUID}"
            )
        
        # Get UID for logging
        uid = metagraph.hotkeys.index(hotkey)
        
        # Check if validator or miner (stake > 0 means validator)
        stake = metagraph.S[uid].item()
        is_validator = stake > 0
        
        return hotkey
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

def verify_validator_signature(request: Request):
    """
    Additional check to ensure the hotkey is a validator (has stake).
    Used for endpoints that only validators should access.
    """
    hotkey = verify_signature(request)  # First verify signature and registration
    
    try:
        metagraph = get_metagraph()
        uid = metagraph.hotkeys.index(hotkey)
        stake = metagraph.S[uid].item()
        
        if stake <= 0:
            raise HTTPException(
                status_code=403,
                detail=f"Hotkey {hotkey} is not a validator on subnet {SUBNET_NETUID}"
            )
        
        return hotkey
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Validator verification error: {str(e)}")
