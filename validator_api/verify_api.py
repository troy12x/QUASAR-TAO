import requests
import bittensor as bt
import time
import json
import hashlib

import sys

API_URL = "http://localhost:8000"
if len(sys.argv) > 1:
    API_URL = sys.argv[1]

def test_api():
    # 1. Setup mock wallet
    wallet = bt.wallet(name="mock_validator")
    hotkey_ss58 = wallet.hotkey.ss58_address
    
    def get_headers():
        signature = f"0x{wallet.hotkey.sign(hotkey_ss58).hex()}"
        return {
            "Hotkey": hotkey_ss58,
            "Signature": signature
        }

    print("--- Testing Health ---")
    try:
        r = requests.get(f"{API_URL}/health")
        print(f"Health: {r.status_code}, {r.json()}")
    except Exception as e:
        print(f"Health check failed (is server running?): {e}")
        return

    print("\n--- Testing GET /get_task ---")
    r = requests.get(f"{API_URL}/get_task", headers=get_headers())
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        task = r.json()
        print(f"Task: {task}")
        task_id = task['id']
    else:
        print(f"Error: {r.text}")
        return

    print("\n--- Testing POST /report_result ---")
    # Simulate a miner response
    response_text = "The latest advancements in quantum computing include topological qubits..."
    
    report_data = {
        "task_id": task_id,
        "miner_hotkey": "5MinerHotkey...", # Mock miner
        "miner_uid": 123,
        "response_text": response_text
    }
    
    r = requests.post(f"{API_URL}/report_result", headers=get_headers(), json=report_data)
    print(f"Status: {r.status_code}")
    print(f"Result: {r.json()}")

    print("\n--- Testing Duplicate Detection ---")
    # Try reporting the same response_text for a different miner
    report_data["miner_hotkey"] = "5AnotherMiner..."
    r = requests.post(f"{API_URL}/report_result", headers=get_headers(), json=report_data)
    print(f"Status: {r.status_code}")
    print(f"Result: {r.json()} (Expected: duplicate_response)")

    print("\n--- Testing GET /get_scores ---")
    r = requests.get(f"{API_URL}/get_scores", headers=get_headers())
    print(f"Status: {r.status_code}")
    print(f"Scores: {r.json()}")

if __name__ == "__main__":
    test_api()
