#!/usr/bin/env python3
"""Diagnostic script to check Result and MinerScore tables directly."""

import sys
import os

# Set DATABASE_URL from command line
if len(sys.argv) > 1:
    DATABASE_URL = sys.argv[1]
else:
    DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("Usage: python check_tables.py <DATABASE_URL>")
    sys.exit(1)

os.environ["DATABASE_URL"] = DATABASE_URL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validator_api.database import engine, SessionLocal
from validator_api import models

def check_tables():
    """Check Result and MinerScore tables."""
    print("Checking Result and MinerScore tables...")
    print("-" * 60)

    db = SessionLocal()

    try:
        # Check Result table
        print("\n[RESULT TABLE]")
        result_count = db.query(models.Result).count()
        print(f"Total Result entries: {result_count}")

        if result_count > 0:
            results = db.query(models.Result).limit(5).all()
            for i, r in enumerate(results):
                print(f"  {i+1}. id={r.id} hotkey={r.miner_hotkey[:12]}... task_id={r.task_id[:20]}... score={r.score}")

        # Check MinerScore table
        print("\n[MINERSCORE TABLE]")
        miner_score_count = db.query(models.MinerScore).count()
        print(f"Total MinerScore entries: {miner_score_count}")

        if miner_score_count > 0:
            scores = db.query(models.MinerScore).limit(5).all()
            for i, s in enumerate(scores):
                print(f"  {i+1}. hotkey={s.hotkey[:12]}... model={s.model_name} league={s.league} score={s.score} tasks={s.tasks_completed}")
        else:
            print("  ℹ️  MinerScore table is empty - this is why /get_scores returns 500")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()

if __name__ == "__main__":
    sys.exit(check_tables())
