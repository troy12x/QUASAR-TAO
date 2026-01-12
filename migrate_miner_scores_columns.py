#!/usr/bin/env python3
"""Migration script to add missing columns to miner_scores table."""

import sys
import os

# Set DATABASE_URL from command line
if len(sys.argv) > 1:
    DATABASE_URL = sys.argv[1]
else:
    DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("Usage: python migrate_miner_scores_columns.py <DATABASE_URL>")
    sys.exit(1)

os.environ["DATABASE_URL"] = DATABASE_URL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validator_api.database import engine
from sqlalchemy import text

def migrate_columns():
    """Add missing columns to miner_scores table."""
    print("Migrating miner_scores table columns...")
    print("-" * 60)

    with engine.connect() as conn:
        try:
            # Check current columns
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'miner_scores'
                ORDER BY ordinal_position
            """))
            columns = result.fetchall()
            print(f"Current columns: {[c[0] for c in columns]}")

            # Add missing columns if they don't exist
            column_names = [c[0] for c in columns]

            if 'model_name' not in column_names:
                print("Adding 'model_name' column...")
                conn.execute(text("ALTER TABLE miner_scores ADD COLUMN model_name VARCHAR"))
                conn.commit()
                print("✅ Added 'model_name'")
            else:
                print("ℹ️  'model_name' already exists")

            if 'league' not in column_names:
                print("Adding 'league' column...")
                conn.execute(text("ALTER TABLE miner_scores ADD COLUMN league VARCHAR"))
                conn.commit()
                print("✅ Added 'league'")
            else:
                print("ℹ️  'league' already exists")

            if 'tasks_completed' not in column_names:
                print("Adding 'tasks_completed' column...")
                conn.execute(text("ALTER TABLE miner_scores ADD COLUMN tasks_completed INTEGER DEFAULT 0"))
                conn.commit()
                print("✅ Added 'tasks_completed'")
            else:
                print("ℹ️  'tasks_completed' already exists")

            if 'last_updated' not in column_names:
                print("Adding 'last_updated' column...")
                conn.execute(text("ALTER TABLE miner_scores ADD COLUMN last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
                conn.commit()
                print("✅ Added 'last_updated'")
            else:
                print("ℹ️  'last_updated' already exists")

            # Verify final schema
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'miner_scores'
                ORDER BY ordinal_position
            """))
            columns = result.fetchall()
            print(f"\nFinal columns: {[c[0] for c in columns]}")

            return 0

        except Exception as e:
            print(f"❌ Migration failed: {e}")
            import traceback
            traceback.print_exc()
            conn.rollback()
            return 1

if __name__ == "__main__":
    sys.exit(migrate_columns())
