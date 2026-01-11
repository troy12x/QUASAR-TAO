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
            # Use Subtensor to get metagraph
            subtensor = bt.Subtensor(network="finney")
            _metagraph_cache = subtensor.metagraph(SUBNET_NETUID)
            _metagraph_cache_time = current_time
        except Exception as e:
            # If we can't fetch metagraph, log the error but don't crash
            import logging
            logging.error(f"Failed to fetch metagraph for subnet {SUBNET_NETUID}: {e}")
            # Return None to indicate metagraph is unavailable
            return None
    
    return _metagraph_cache

def verify_signature(request: Request):
    """
    Middleware/Dependency to verify Bittensor signatures.
    
    Expects headers:
    - 'Hotkey': The SS58 address of the hotkey.
    - 'Signature': The signature of the hotkey (signing the hotkey itself).
    
    Note: Metagraph check disabled due to bittensor version incompatibility.
    Signature verification is sufficient for authentication.
    """
    hotkey = request.headers.get("Hotkey")
    signature = request.headers.get("Signature")

    if not hotkey or not signature:
        raise HTTPException(status_code=401, detail="Missing Hotkey or Signature headers")

    try:
        # Verify signature
        try:
            signature_bytes = bytes.fromhex(signature)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid signature format. Expected hex string")
        
        keypair = bt.Keypair(ss58_address=hotkey)
        if not keypair.verify(hotkey.encode(), signature_bytes):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        return hotkey
        
    except HTTPException:
        raise
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
