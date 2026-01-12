#!/usr/bin/env python3
"""Compare MinerScore hotkeys with metagraph hotkeys."""

import sys
import os

# Set DATABASE_URL from command line
if len(sys.argv) > 1:
    DATABASE_URL = sys.argv[1]
else:
    DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("Usage: python compare_hotkeys.py <DATABASE_URL>")
    sys.exit(1)

os.environ["DATABASE_URL"] = DATABASE_URL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validator_api.database import SessionLocal
from validator_api import models

def compare_hotkeys():
    """Compare database hotkeys with what's expected."""
    print("Checking MinerScore hotkeys...")
    print("-" * 60)

    db = SessionLocal()

    try:
        # Get all hotkeys from MinerScore
        miner_scores = db.query(models.MinerScore).all()

        print(f"\nTotal MinerScore entries: {len(miner_scores)}")
        print("\nHotkeys in database:")
        for i, ms in enumerate(miner_scores):
            print(f"  {i+1}. {ms.hotkey} (score={ms.score}, league={ms.league})")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()

if __name__ == "__main__":
    sys.exit(compare_hotkeys())
