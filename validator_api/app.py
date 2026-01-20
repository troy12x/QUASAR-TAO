from fastapi import FastAPI, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel
import uuid
import sys
import os
import random
import time
import hashlib
import requests
import json
import subprocess
import tempfile
import shutil
from collections import defaultdict

# Add parent directory to path to import quasar
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from . import models
from . import auth
from .database import engine, get_db

from fastapi.middleware.cors import CORSMiddleware

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Add new columns if they don't exist (migration)
from sqlalchemy import text
with engine.connect() as conn:
    # Check database type
    db_type = conn.execute(text("SELECT current_database()")).scalar()
    if "postgresql" in str(engine.url):
        # PostgreSQL: use information_schema
        result = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'speed_submissions'
        """))
        columns = [row[0] for row in result]
        if "vram_mb" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN vram_mb REAL"))
            conn.commit()
        if "benchmarks" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN benchmarks TEXT"))
            conn.commit()
        if "validated" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN validated BOOLEAN DEFAULT FALSE"))
            conn.commit()
    else:
        # SQLite: use PRAGMA
        result = conn.execute(text("PRAGMA table_info(speed_submissions)"))
        columns = [row[1] for row in result]
        if "vram_mb" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN vram_mb REAL"))
            conn.commit()
        if "benchmarks" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN benchmarks TEXT"))
            conn.commit()
        if "validated" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN validated BOOLEAN DEFAULT 0"))
            conn.commit()

app = FastAPI(title="Quasar Validator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import validation constants
REQUIRED_IMPORTS = [
    "import torch",
    "import torch.nn.functional as F",
    "import triton",
    "import triton.language as tl",
    "from fla.ops.utils.index import prepare_chunk_indices",
    "from fla.ops.quasar.forward_substitution import forward_substitution_kernel",
    "from fla.utils import IS_AMD",
    "from fla.utils import autocast_custom_bwd",
    "from fla.utils import autocast_custom_fwd",
    "from fla.utils import autotune_cache_kwargs",
    "from fla.utils import check_shared_mem",
    "from fla.utils import input_guard",
]

FORBIDDEN_IMPORTS = [
    "from fla.ops.gla",
    "from fla.ops.kda",
    "import fla.ops.gla",
    "import fla.ops.kda",
]

def validate_imports(repo_path: str) -> tuple[bool, List[str]]:
    """Validate that files have required imports and no forbidden imports."""
    quasar_dir = os.path.join(repo_path, "fla/ops/quasar")
    target_files = ["chunk.py", "chunk_intra_token_parallel.py", "forward_substitution.py", "fused_recurrent.py", "gate.py"]

    errors = []

    for filename in target_files:
        file_path = os.path.join(quasar_dir, filename)
        if not os.path.exists(file_path):
            errors.append(f"Missing file: {filename}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for forbidden imports
        for forbidden in FORBIDDEN_IMPORTS:
            if forbidden in content:
                errors.append(f"{filename}: Forbidden import found: {forbidden}")

        # Check for required imports (only for main files like chunk.py)
        if filename == "chunk.py":
            for required in REQUIRED_IMPORTS:
                if required not in content:
                    errors.append(f"{filename}: Missing required import: {required}")

    return len(errors) == 0, errors

# Rate limiting for DDOS protection
# Simple in-memory rate limiter: {hotkey: [timestamp1, timestamp2, ...]}
rate_limit_store = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 10  # max requests per window

def check_rate_limit(hotkey: str):
    """Check if hotkey has exceeded rate limit."""
    now = time.time()
    # Remove old timestamps outside the window
    rate_limit_store[hotkey] = [t for t in rate_limit_store[hotkey] if now - t < RATE_LIMIT_WINDOW]

    if len(rate_limit_store[hotkey]) >= RATE_LIMIT_MAX_REQUESTS:
        print(f"⚠️ [RATE_LIMIT] Hotkey {hotkey[:12]}... exceeded rate limit")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        )

    # Add current timestamp
    rate_limit_store[hotkey].append(now)

# League configuration
LEAGUES = ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "1M"]
LEAGUE_MULTIPLIERS = {
    "100k": 0.5,
    "200k": 0.75,
    "300k": 1.0,
    "400k": 1.25,
    "500k": 1.5,
    "600k": 1.75,
    "700k": 2.0,
    "800k": 2.25,
    "900k": 2.5,
    "1M": 3.0
}

def get_league(context_length: int) -> str:
    """Determine league based on context length."""
    for i, league in enumerate(LEAGUES):
        max_tokens = (i + 1) * 100_000
        if context_length <= max_tokens:
            return league
    return "1M"  # Fallback to highest league

class WeightEntry(BaseModel):
    uid: int
    hotkey: str
    weight: float

class GetWeightsResponse(BaseModel):
    epoch: int
    weights: List[WeightEntry]

@app.post("/submit_kernel")
def submit_kernel(
    req: models.SpeedSubmissionRequest,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """
    Submit kernel optimization results from miners.
    Stores fork URL, commit hash, performance metrics, and signature.
    Validates imports before accepting submission.
    """
    import traceback
    try:
        print(f"📥 [SUBMIT_KERNEL] Miner: {req.miner_hotkey[:8]} | Fork: {req.fork_url}")
        print(f"📥 [SUBMIT_KERNEL] Commit: {req.commit_hash[:12]}... | Performance: {req.tokens_per_sec:.2f} tokens/sec")
        if req.vram_mb is not None:
            print(f"📥 [SUBMIT_KERNEL] VRAM_MB: {req.vram_mb:.2f}")
        if req.benchmarks is not None:
            try:
                print(f"📥 [SUBMIT_KERNEL] Benchmarks: {len(req.benchmarks)} seq lengths")
            except Exception:
                print(f"📥 [SUBMIT_KERNEL] Benchmarks: (unprintable)")

        # Verify the hotkey matches the authenticated miner
        if req.miner_hotkey != hotkey:
            raise HTTPException(status_code=403, detail="Hotkey mismatch")

        # Check if miner is registered
        miner_reg = db.query(models.MinerRegistration).filter(
            models.MinerRegistration.hotkey == hotkey
        ).first()

        if not miner_reg:
            raise HTTPException(status_code=404, detail="Miner not registered")

        # Clone and validate imports before accepting submission
        print(f"🔍 [SUBMIT_KERNEL] Cloning repo for import validation...")
        temp_dir = tempfile.mkdtemp(prefix="quasar_submit_")
        try:
            repo_path = os.path.join(temp_dir, "repo")
            subprocess.run(
                ["git", "clone", "--depth", "1", req.fork_url, repo_path],
                check=True,
                capture_output=True,
                timeout=60
            )

            # Checkout specific commit if provided
            if req.commit_hash:
                subprocess.run(
                    ["git", "checkout", req.commit_hash],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    timeout=30
                )

            # Validate imports
            print(f"🔍 [SUBMIT_KERNEL] Validating imports...")
            imports_valid, import_errors = validate_imports(repo_path)
            if not imports_valid:
                print(f"❌ [SUBMIT_KERNEL] Import validation failed:")
                for error in import_errors:
                    print(f"  - {error}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Import validation failed",
                        "errors": import_errors
                    }
                )
            print(f"✅ [SUBMIT_KERNEL] Import validation passed")
        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # Create new speed submission
        new_submission = models.SpeedSubmission(
            miner_hotkey=req.miner_hotkey,
            miner_uid=miner_reg.uid,
            fork_url=req.fork_url,
            commit_hash=req.commit_hash,
            target_sequence_length=req.target_sequence_length,
            tokens_per_sec=req.tokens_per_sec,
            vram_mb=req.vram_mb,
            benchmarks=json.dumps(req.benchmarks) if req.benchmarks else None,
            signature=req.signature
        )

        db.add(new_submission)
        db.commit()
        db.refresh(new_submission)

        print(f"✅ [SUBMIT_OPT] Submission saved with ID: {new_submission.id}")
        return models.SpeedSubmissionResponse(
            submission_id=new_submission.id,
            miner_hotkey=new_submission.miner_hotkey,
            fork_url=new_submission.fork_url,
            commit_hash=new_submission.commit_hash,
            target_sequence_length=new_submission.target_sequence_length,
            tokens_per_sec=new_submission.tokens_per_sec,
            vram_mb=new_submission.vram_mb,
            benchmarks=json.loads(new_submission.benchmarks) if new_submission.benchmarks else None,
            created_at=new_submission.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [SUBMIT_KERNEL] Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    return models.SpeedSubmissionResponse(
        submission_id=new_submission.id,
        miner_hotkey=new_submission.miner_hotkey,
        fork_url=new_submission.fork_url,
        commit_hash=new_submission.commit_hash,
        target_sequence_length=new_submission.target_sequence_length,
        tokens_per_sec=new_submission.tokens_per_sec,
        created_at=new_submission.created_at
    )

@app.get("/get_submission_stats")
def get_submission_stats(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get submission statistics for the system.
    Returns recent submissions with performance metrics.
    Only returns unvalidated submissions for validator to process.
    """
    import traceback
    try:
        # Get recent unvalidated submissions
        recent_submissions = (
            db.query(models.SpeedSubmission)
            .filter(models.SpeedSubmission.validated == False)
            .order_by(models.SpeedSubmission.created_at.desc())
            .limit(limit)
            .all()
        )

        # Calculate stats
        total_submissions = db.query(models.SpeedSubmission).count()

        if recent_submissions:
            avg_tokens_per_sec = sum(s.tokens_per_sec for s in recent_submissions) / len(recent_submissions)
            max_tokens_per_sec = max(s.tokens_per_sec for s in recent_submissions)
            min_tokens_per_sec = min(s.tokens_per_sec for s in recent_submissions)
        else:
            avg_tokens_per_sec = max_tokens_per_sec = min_tokens_per_sec = 0.0

        # Get top performers
        top_submissions = (
            db.query(models.SpeedSubmission)
            .order_by(models.SpeedSubmission.tokens_per_sec.desc())
            .limit(10)
            .all()
        )

        def parse_benchmarks(benchmarks_str):
            """Parse benchmarks JSON string with error handling."""
            if not benchmarks_str:
                return None
            try:
                return json.loads(benchmarks_str)
            except Exception as e:
                print(f"Error parsing benchmarks: {e}")
                return None

        return {
            "total_submissions": total_submissions,
            "recent_submissions": [
                {
                    "id": s.id,
                    "miner_hotkey": s.miner_hotkey,
                    "fork_url": s.fork_url,
                    "commit_hash": s.commit_hash,
                    "target_sequence_length": s.target_sequence_length,
                    "tokens_per_sec": s.tokens_per_sec,
                    "vram_mb": s.vram_mb,
                    "benchmarks": parse_benchmarks(s.benchmarks),
                    "validated": s.validated,
                    "created_at": s.created_at.isoformat()
                }
                for s in recent_submissions
            ],
            "stats": {
                "avg_tokens_per_sec": round(avg_tokens_per_sec, 2),
                "max_tokens_per_sec": round(max_tokens_per_sec, 2),
                "min_tokens_per_sec": round(min_tokens_per_sec, 2),
                "total_submissions": total_submissions
            },
            "top_performers": [
                {
                    "id": s.id,
                    "miner_hotkey": s.miner_hotkey,
                    "tokens_per_sec": s.tokens_per_sec,
                    "target_sequence_length": s.target_sequence_length,
                    "created_at": s.created_at.isoformat()
                }
                for s in top_submissions
            ]
        }
    except Exception as e:
        print(f"Error in get_submission_stats: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/mark_validated")
def mark_validated(
    req: dict,
    db: Session = Depends(get_db)
):
    """
    Mark a submission as validated.
    Used by validators to avoid re-evaluating the same submission.
    """
    submission_id = req.get("submission_id")
    if not submission_id:
        raise HTTPException(status_code=400, detail="submission_id required")

    submission = db.query(models.SpeedSubmission).filter(models.SpeedSubmission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    submission.validated = True
    db.commit()

    return {"status": "ok", "submission_id": submission_id}

@app.post("/register_miner")
def register_miner(
    req: models.RegisterMinerRequest,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """
    Register a miner with a specific model and league.
    Miners can register multiple times for different (model, league) combinations.
    """
    if req.league not in LEAGUES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid league. Must be one of: {', '.join(LEAGUES)}"
        )

    # Check if already registered for this (hotkey, model, league) combo
    existing = db.query(models.MinerScore).filter(
        models.MinerScore.hotkey == hotkey,
        models.MinerScore.model_name == req.model_name,
        models.MinerScore.league == req.league
    ).first()

    if existing:
        # Check if MinerRegistration exists, create if not
        registration = db.query(models.MinerRegistration).filter(
            models.MinerRegistration.hotkey == hotkey
        ).first()

        if not registration:
            new_registration = models.MinerRegistration(
                hotkey=hotkey,
                uid=0
            )
            db.add(new_registration)
            db.commit()
            print(f"✅ [REGISTER] Created missing MinerRegistration for {hotkey[:8]}")

        print(f"ℹ️ [REGISTER] Miner {hotkey[:8]} already registered for {req.model_name} in {req.league}")
        return {
            "status": "already_registered",
            "hotkey": hotkey,
            "model_name": req.model_name,
            "league": req.league,
            "current_score": existing.score
        }

    # Create new registration
    new_score = models.MinerScore(
        hotkey=hotkey,
        model_name=req.model_name,
        league=req.league,
        score=0.0,
        tasks_completed=0
    )
    db.add(new_score)

    # Create MinerRegistration entry
    new_registration = models.MinerRegistration(
        hotkey=hotkey,
        uid=0  # Will be updated when miner is found on metagraph
    )
    db.add(new_registration)

    db.commit()

    print(f"✅ [REGISTER] Miner {hotkey[:8]} registered for {req.model_name} in {req.league}")
    return {
        "status": "registered",
        "hotkey": hotkey,
        "model_name": req.model_name,
        "league": req.league
    }

@app.get("/get_weights")
def get_weights(
    db: Session = Depends(get_db)
):
    """
    Get weights for validators to submit to Bittensor.
    Winner-takes-all: best miner gets 100% of weight.
    """
    # Get latest scores from MinerScore table
    miner_scores = db.query(models.MinerScore).all()
    
    if not miner_scores:
        return GetWeightsResponse(epoch=int(time.time()), weights=[])
    
    # Find winner (highest score)
    winner = max(miner_scores, key=lambda x: x.score)
    
    # Map hotkey to UID (simplified - in production you'd need a registry)
    # For now, return hotkey-based weights
    weights = []
    
    # Winner gets 1.0 (100%)
    weights.append(WeightEntry(
        uid=0,  # Placeholder - validators should map hotkey to UID
        hotkey=winner.hotkey,
        weight=1.0
    ))
    
    print(f"[WEIGHTS] Winner: {winner.hotkey[:12]}... with score {winner.score:.4f}")
    
    return GetWeightsResponse(epoch=int(time.time()), weights=weights)
