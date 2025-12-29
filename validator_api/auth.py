import bittensor as bt
from fastapi import Request, HTTPException, Depends
from typing import Optional

def verify_signature(request: Request):
    """
    Middleware/Dependency to verify Bittensor signatures.
    Expects headers:
    - 'Hotkey': The SS58 address of the hotkey.
    - 'Signature': The signature of the hotkey (signing the hotkey itself for simplicity, or a timestamp).
    """
    hotkey = request.headers.get("Hotkey")
    signature = request.headers.get("Signature")

    if not hotkey or not signature:
        raise HTTPException(status_code=401, detail="Missing Hotkey or Signature headers")

    try:
        # For simplicity, we assume the signature is of the hotkey itself.
        # In a real-world scenario, you might sign a nonce or a timestamp to prevent replay attacks.
        keypair = bt.Keypair(ss58_address=hotkey)
        if keypair.verify(hotkey, signature):
            return hotkey
        else:
            raise HTTPException(status_code=401, detail="Invalid signature")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

# Alternative: Verify against a metagraph if needed to ensure the hotkey is registered.
# However, this might be slow and better handled by periodically syncing the metagraph in the app.
