"""Fetch a task from the deployed QUASAR-SUBNET API using validator hotkey auth.

Endpoint: https://quasar-subnet.onrender.com/get_task
Auth: headers Hotkey + Signature (signature over the hotkey bytes)
"""

import requests
import bittensor as bt

API_URL = "https://quasar-subnet.onrender.com"
GET_TASK_ENDPOINT = f"{API_URL}/get_task"

# Validator Hotkey Mnemonic (provided)
VALIDATOR_MNEMONIC = "bottom metal radar abuse cool bamboo agent reveal fever bachelor way ranch"


def fetch_task():
    print(f"Fetching task from {GET_TASK_ENDPOINT}...")

    # Build keypair and headers
    try:
        keypair = bt.Keypair.create_from_mnemonic(VALIDATOR_MNEMONIC)
        hotkey = keypair.ss58_address
        signature_bytes = keypair.sign(hotkey.encode())
        signature_hex = signature_bytes.hex()
        headers = {
            "Hotkey": hotkey,
            "Signature": signature_hex,
        }
        print(f"Using hotkey: {hotkey}")
    except Exception as e:
        print(f"❌ Failed to build keypair/signature: {e}")
        return None

    try:
        resp = requests.get(GET_TASK_ENDPOINT, headers=headers, timeout=15)
        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            try:
                print(f"Response: {resp.json()}")
            except Exception:
                print(f"Raw response: {resp.text[:500]}")
            return None

        data = resp.json()
        print("\n" + "=" * 60)
        print("TASK DETAILS")
        print("=" * 60)
        print(f"Task ID: {data.get('id')}")
        print(f"Dataset: {data.get('dataset_name')}")
        print(f"Task Type: {data.get('task_type')}")
        print(f"Context Length: {data.get('context_length')}")
        print(f"Difficulty: {data.get('difficulty_level')}")
        print(f"Evaluation Metrics: {data.get('evaluation_metrics')}")
        print(f"Created At: {data.get('created_at')}")

        prompt = data.get("prompt", "")
        print("\nPROMPT (first 800 chars)")
        print("-" * 40)
        print(prompt[:800] + ("..." if len(prompt) > 800 else ""))

        return data
    except Exception as e:
        print(f"❌ Error fetching task: {e}")
        return None


if __name__ == "__main__":
    fetch_task()
