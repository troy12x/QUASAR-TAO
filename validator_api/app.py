from fastapi import FastAPI, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
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
from collections import defaultdict

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Load .env from project root (parent of validator_api)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    env_path = os.path.join(parent_dir, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
except ImportError:
    # python-dotenv not installed, skip (env vars can still be set manually)
    pass

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

# IP Banning Configuration
MAX_FAILURES_BEFORE_BAN = 5
BAN_DURATION_HOURS = 24

def check_ip_ban(ip_address: str, db: Session) -> tuple[bool, Optional[str]]:
    """
    Check if IP is banned.
    Returns: (is_banned, reason)
    """
    if not ip_address:
        return False, None
    
    ip_ban = db.query(models.IPBan).filter(
        models.IPBan.ip_address == ip_address
    ).first()
    
    if not ip_ban:
        return False, None
    
    # Check if ban has expired
    if ip_ban.is_banned and ip_ban.banned_until:
        if datetime.utcnow() < ip_ban.banned_until:
            remaining = (ip_ban.banned_until - datetime.utcnow()).total_seconds() / 3600
            return True, f"IP banned for {remaining:.1f} more hours"
        else:
            # Ban expired, reset
            ip_ban.is_banned = False
            ip_ban.banned_until = None
            ip_ban.failure_count = 0
            db.commit()
            return False, None
    
    return False, None

def record_failure(ip_address: str, db: Session):
    """Record a failed submission for IP tracking."""
    if not ip_address:
        return
    
    ip_ban = db.query(models.IPBan).filter(
        models.IPBan.ip_address == ip_address
    ).first()
    
    if not ip_ban:
        ip_ban = models.IPBan(
            ip_address=ip_address,
            failure_count=1,
            last_failure_time=datetime.utcnow()
        )
        db.add(ip_ban)
    else:
        ip_ban.failure_count += 1
        ip_ban.last_failure_time = datetime.utcnow()
        
        # Ban if exceeds threshold
        if ip_ban.failure_count >= MAX_FAILURES_BEFORE_BAN:
            from datetime import timedelta
            ip_ban.is_banned = True
            ip_ban.banned_until = datetime.utcnow() + timedelta(hours=BAN_DURATION_HOURS)
            print(f"🚫 [IP_BAN] Banned IP {ip_address} for {BAN_DURATION_HOURS} hours "
                  f"(failures: {ip_ban.failure_count})")
    
    db.commit()

def record_success(ip_address: str, db: Session):
    """Reset failure count on successful submission."""
    if not ip_address:
        return
    
    ip_ban = db.query(models.IPBan).filter(
        models.IPBan.ip_address == ip_address
    ).first()
    
    if ip_ban and ip_ban.failure_count > 0:
        ip_ban.failure_count = 0
        db.commit()

# Helper function for solution hash calculation
def calculate_solution_hash(tokens_per_sec: float, target_sequence_length: int, 
                          benchmarks: Optional[Dict] = None) -> str:
    """
    Calculate hash of solution to detect identical results.
    Used for first-submission-wins logic.
    """
    # Normalize to 2 decimal places to account for minor variations
    normalized_tps = round(tokens_per_sec, 2)
    
    # Create hashable representation
    solution_data = {
        "tokens_per_sec": normalized_tps,
        "target_sequence_length": target_sequence_length,
        "benchmarks": benchmarks or {}
    }
    
    # Sort benchmarks for consistent hashing
    if benchmarks:
        solution_data["benchmarks"] = dict(sorted(benchmarks.items()))
    
    # Create hash
    solution_str = json.dumps(solution_data, sort_keys=True)
    return hashlib.sha256(solution_str.encode()).hexdigest()[:16]

# Add new columns if they don't exist (migration)
from sqlalchemy import text
with engine.connect() as conn:
    # Check database type
    is_postgresql = "postgresql" in str(engine.url)
    
    if is_postgresql:
        # PostgreSQL: use information_schema
        # Check speed_submissions columns
        result = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'speed_submissions'
        """))
        columns = [row[0] for row in result]
        
        # Add missing columns
        if "vram_mb" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN vram_mb REAL"))
            conn.commit()
        if "benchmarks" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN benchmarks TEXT"))
            conn.commit()
        if "validated" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN validated BOOLEAN DEFAULT FALSE"))
            conn.commit()
        if "round_id" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN round_id INTEGER"))
            conn.commit()
        if "ip_address" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN ip_address VARCHAR"))
            conn.commit()
        if "is_baseline" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN is_baseline BOOLEAN DEFAULT FALSE"))
            conn.commit()
        if "solution_hash" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN solution_hash VARCHAR"))
            conn.commit()
        
        # Check if competition_rounds table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'competition_rounds'
            )
        """))
        if not result.scalar():
            # Table doesn't exist, will be created by create_all
            pass
        
        # Check if ip_bans table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'ip_bans'
            )
        """))
        if not result.scalar():
            # Table doesn't exist, will be created by create_all
            pass
    else:
        # SQLite: use PRAGMA
        result = conn.execute(text("PRAGMA table_info(speed_submissions)"))
        columns = [row[1] for row in result]
        
        # Add missing columns
        if "vram_mb" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN vram_mb REAL"))
            conn.commit()
        if "benchmarks" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN benchmarks TEXT"))
            conn.commit()
        if "validated" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN validated BOOLEAN DEFAULT 0"))
            conn.commit()
        if "round_id" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN round_id INTEGER"))
            conn.commit()
        if "ip_address" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN ip_address TEXT"))
            conn.commit()
        if "is_baseline" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN is_baseline BOOLEAN DEFAULT 0"))
            conn.commit()
        if "solution_hash" not in columns:
            conn.execute(text("ALTER TABLE speed_submissions ADD COLUMN solution_hash TEXT"))
            conn.commit()
    
    # Commit any pending changes
            conn.commit()

app = FastAPI(title="Quasar Validator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def get_league_for_seq_len(seq_len: int) -> str:
    """Get league based on sequence length."""
    if seq_len >= 1_000_000:
        return "1M"
    elif seq_len >= 900_000:
        return "900k"
    elif seq_len >= 800_000:
        return "800k"
    elif seq_len >= 700_000:
        return "700k"
    elif seq_len >= 600_000:
        return "600k"
    elif seq_len >= 500_000:
        return "500k"
    elif seq_len >= 400_000:
        return "400k"
    elif seq_len >= 300_000:
        return "300k"
    elif seq_len >= 200_000:
        return "200k"
    else:
        return "100k"

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
    request: Request,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """
    Submit kernel optimization results from miners.
    Stores fork URL, commit hash, performance metrics, and signature.
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

        # Extract IP address from request (for IP banning)
        client_ip = None
        if hasattr(request, 'client') and request.client:
            client_ip = request.client.host
        # Also check X-Forwarded-For header (for proxies/load balancers)
        if not client_ip:
            client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.headers.get("X-Real-IP", "").strip()
        
        # Check IP ban BEFORE processing submission
        is_banned, ban_reason = check_ip_ban(client_ip, db)
        if is_banned:
            raise HTTPException(
                status_code=403,
                detail=f"IP address banned: {ban_reason}"
            )

        # Calculate solution hash for duplicate detection
        solution_hash = calculate_solution_hash(
            req.tokens_per_sec,
            req.target_sequence_length,
            req.benchmarks
        )
        
        # Get current round and assign to submission
        current_round = (
            db.query(models.CompetitionRound)
            .filter(models.CompetitionRound.status == "active")
            .order_by(models.CompetitionRound.round_number.desc())
            .first()
        )
        
        if not current_round:
            # Create first round if none exists
            from datetime import timedelta
            current_round = models.CompetitionRound(
                round_number=1,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(hours=48),
                status="active"
            )
            db.add(current_round)
            db.commit()
            db.refresh(current_round)
            print(f"✅ [ROUND] Created first round #{current_round.round_number}")

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
            signature=req.signature,
            round_id=current_round.id,
            solution_hash=solution_hash,
            ip_address=client_ip
        )

        db.add(new_submission)
        db.commit()
        db.refresh(new_submission)

        # Record successful submission (reset failure count)
        if client_ip:
            record_success(client_ip, db)

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
    
    # Record successful validation (reset failure count for IP)
    if submission.ip_address:
        record_success(submission.ip_address, db)

    return {"status": "ok", "submission_id": submission_id}

@app.post("/record_failure")
def record_failure_endpoint(
    req: dict,
    db: Session = Depends(get_db)
):
    """Record a failed submission for IP tracking."""
    ip_address = req.get("ip_address")
    if ip_address:
        record_failure(ip_address, db)
        return {"status": "ok", "ip_address": ip_address, "message": "Failure recorded"}
    return {"status": "ok", "message": "No IP address provided"}

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

# Updated reward distribution for top 4
REWARD_DISTRIBUTION = [0.60, 0.25, 0.10, 0.05]  # 60%, 25%, 10%, 5%
TOP_N_MINERS = 4

@app.get("/get_weights")
def get_weights(
    round_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Get weights for top 4 performers in a specific round.
    
    Reward distribution:
    - 1st place: 60%
    - 2nd place: 25%
    - 3rd place: 10%
    - 4th place: 5%
    
    Args:
        round_id: Optional round ID. If not specified, uses the most recent completed round.
    """
    # If round_id not specified, get current completed round
    if round_id is None:
        round_obj = (
            db.query(models.CompetitionRound)
            .filter(models.CompetitionRound.status == "completed")
            .order_by(models.CompetitionRound.round_number.desc())
            .first()
        )
    else:
        round_obj = db.query(models.CompetitionRound).filter(
            models.CompetitionRound.id == round_id
        ).first()
    
    if not round_obj:
        print("[WEIGHTS] No completed round found")
        return GetWeightsResponse(epoch=int(time.time()), weights=[])
    
    # Get all validated submissions for this round
    submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.round_id == round_obj.id)
        .filter(models.SpeedSubmission.validated == True)
        .all()
    )
    
    if not submissions:
        print(f"[WEIGHTS] No validated submissions in round {round_obj.round_number}")
        return GetWeightsResponse(epoch=int(time.time()), weights=[])
    
    # Calculate rankings with first-submission-wins logic
    # Note: baseline_submission_id on a round is the baseline for the NEXT round
    # When calculating weights for a round, we need the baseline that was active DURING that round
    # For the first round (lowest round_number): no baseline (None)
    # For subsequent rounds: use previous round's baseline_submission_id
    
    # Find the first round (lowest round_number)
    first_round = db.query(models.CompetitionRound).order_by(
        models.CompetitionRound.round_number.asc()
    ).first()
    
    if first_round and round_obj.round_number == first_round.round_number:
        # This is the first round - no baseline
        baseline_id = None
    else:
        # Get previous round's baseline
        prev_round = db.query(models.CompetitionRound).filter(
            models.CompetitionRound.round_number == round_obj.round_number - 1
        ).first()
        baseline_id = prev_round.baseline_submission_id if prev_round else None
    
    rankings = calculate_rankings(submissions, baseline_id, db)
    
    if not rankings:
        print(f"[WEIGHTS] No valid rankings for round {round_obj.round_number}")
        return GetWeightsResponse(epoch=int(time.time()), weights=[])
    
    # Distribute weights to top 4
    weights = []
    print(f"[WEIGHTS] Distributing rewards for round {round_obj.round_number}:")
    
    for i, ranking in enumerate(rankings[:TOP_N_MINERS]):
        weight = REWARD_DISTRIBUTION[i] if i < len(REWARD_DISTRIBUTION) else 0.0
        
        # Get UID from MinerRegistration if available
        miner_reg = db.query(models.MinerRegistration).filter(
            models.MinerRegistration.hotkey == ranking["miner_hotkey"]
        ).first()
        uid = miner_reg.uid if miner_reg else 0
        
        weights.append(WeightEntry(
            uid=uid,
            hotkey=ranking["miner_hotkey"],
            weight=weight
        ))
        
        print(f"  #{ranking['rank']}: {ranking['miner_hotkey'][:12]}... - "
              f"weight={weight:.2%} "
              f"(weighted_score={ranking['weighted_score']:.0f})")
    
    return GetWeightsResponse(epoch=int(time.time()), weights=weights)

# ==================== ROUND MANAGEMENT ENDPOINTS ====================

@app.get("/get_current_round", response_model=models.RoundResponse)
def get_current_round(db: Session = Depends(get_db)):
    """Get the current active round."""
    try:
        current_round = (
            db.query(models.CompetitionRound)
            .filter(models.CompetitionRound.status == "active")
            .order_by(models.CompetitionRound.round_number.desc())
            .first()
        )
        
        if not current_round:
            # Create new round if none exists
            from datetime import timedelta
            # Get the highest round number
            last_round = (
                db.query(models.CompetitionRound)
                .order_by(models.CompetitionRound.round_number.desc())
                .first()
            )
            next_round_number = (last_round.round_number + 1) if last_round else 1
            
            current_round = models.CompetitionRound(
                round_number=next_round_number,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(hours=48),
                status="active"
            )
            db.add(current_round)
            db.commit()
            db.refresh(current_round)
            print(f"✅ [ROUND] Created new round #{current_round.round_number}")
        
        # Calculate time remaining
        now = datetime.utcnow()
        time_remaining = max(0, int((current_round.end_time - now).total_seconds()))
        
        # Count submissions in this round
        submission_count = (
            db.query(models.SpeedSubmission)
            .filter(models.SpeedSubmission.round_id == current_round.id)
            .count()
        )
        
        # Create response with proper datetime handling
        response_data = {
            "id": current_round.id,
            "round_number": current_round.round_number,
            "start_time": current_round.start_time.isoformat() if isinstance(current_round.start_time, datetime) else str(current_round.start_time),
            "end_time": current_round.end_time.isoformat() if isinstance(current_round.end_time, datetime) else str(current_round.end_time),
            "status": current_round.status,
            "time_remaining_seconds": time_remaining,
            "baseline_submission_id": current_round.baseline_submission_id,
            "winner_hotkey": current_round.winner_hotkey,
            "total_submissions": submission_count
        }
        
        print(f"✅ [GET_CURRENT_ROUND] Returning round {current_round.round_number}")
        return models.RoundResponse(**response_data)
        
    except Exception as e:
        import traceback
        error_msg = f"❌ [GET_CURRENT_ROUND] Error: {e}"
        print(error_msg)
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting current round: {str(e)}"
        )

@app.post("/create_round", response_model=models.RoundResponse)
def create_round(
    req: models.CreateRoundRequest,
    db: Session = Depends(get_db)
):
    """Create a new competition round."""
    # Get last round number
    last_round = (
        db.query(models.CompetitionRound)
        .order_by(models.CompetitionRound.round_number.desc())
        .first()
    )
    
    next_round_number = (last_round.round_number + 1) if last_round else 1
    
    # Mark previous round as completed
    if last_round and last_round.status == "active":
        last_round.status = "completed"
        db.commit()
    
    # Create new round
    from datetime import timedelta
    new_round = models.CompetitionRound(
        round_number=next_round_number,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(hours=req.duration_hours),
        status="active",
        baseline_submission_id=req.baseline_submission_id
    )
    
    db.add(new_round)
    db.commit()
    db.refresh(new_round)
    
    print(f"✅ [ROUND] Created round #{next_round_number} (baseline: {req.baseline_submission_id})")
    
    return models.RoundResponse(
        id=new_round.id,
        round_number=new_round.round_number,
        start_time=new_round.start_time,
        end_time=new_round.end_time,
        status=new_round.status,
        time_remaining_seconds=int((new_round.end_time - datetime.utcnow()).total_seconds()),
        baseline_submission_id=new_round.baseline_submission_id
    )

def calculate_rankings(
    submissions: List[models.SpeedSubmission],
    baseline_submission_id: Optional[int],
    db: Session
) -> List[Dict]:
    """
    Calculate rankings with first-submission-wins logic.
    
    Ranking criteria (in order):
    1. Weighted score (tokens_per_sec * league_multiplier) - DESC
    2. Created timestamp (first submission wins) - ASC
    3. Submission ID (tiebreaker) - ASC
    """
    # Get baseline if exists
    baseline = None
    if baseline_submission_id:
        baseline = db.query(models.SpeedSubmission).filter(
            models.SpeedSubmission.id == baseline_submission_id
        ).first()
    
    # Calculate weighted scores
    ranked_submissions = []
    for sub in submissions:
        # Skip if below baseline (for round 2+)
        if baseline:
            baseline_league = get_league_for_seq_len(baseline.target_sequence_length)
            baseline_multiplier = LEAGUE_MULTIPLIERS.get(baseline_league, 1.0)
            baseline_weighted = baseline.tokens_per_sec * baseline_multiplier
            
            sub_league = get_league_for_seq_len(sub.target_sequence_length)
            sub_multiplier = LEAGUE_MULTIPLIERS.get(sub_league, 1.0)
            sub_weighted = sub.tokens_per_sec * sub_multiplier
            
            if sub_weighted <= baseline_weighted:
                continue  # Skip submissions that don't beat baseline
        
        # Calculate weighted score
        league = get_league_for_seq_len(sub.target_sequence_length)
        multiplier = LEAGUE_MULTIPLIERS.get(league, 1.0)
        weighted_score = sub.tokens_per_sec * multiplier
        
        ranked_submissions.append({
            "submission_id": sub.id,
            "miner_hotkey": sub.miner_hotkey,
            "tokens_per_sec": sub.tokens_per_sec,
            "target_sequence_length": sub.target_sequence_length,
            "league": league,
            "multiplier": multiplier,
            "weighted_score": weighted_score,
            "created_at": sub.created_at,
            "solution_hash": sub.solution_hash
        })
    
    # Sort by: weighted_score DESC, created_at ASC, submission_id ASC
    ranked_submissions.sort(
        key=lambda x: (
            -x["weighted_score"],  # Negative for descending
            x["created_at"],  # Ascending (first wins)
            x["submission_id"]  # Tiebreaker
        )
    )
    
    # Add rank numbers
    for i, sub in enumerate(ranked_submissions, start=1):
        sub["rank"] = i
    
    return ranked_submissions

@app.get("/get_submission_rate")
def get_submission_rate(
    window_minutes: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get current submission rate (submissions per minute).
    Used by validators to adjust polling frequency dynamically.
    
    Args:
        window_minutes: Time window in minutes to calculate rate (default: 10)
    
    Returns:
        Dictionary with submissions_per_minute, recent_submissions, window_minutes
    """
    cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
    
    recent_submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.created_at >= cutoff_time)
        .count()
    )
    
    submissions_per_minute = recent_submissions / window_minutes if window_minutes > 0 else 0
    
    return {
        "submissions_per_minute": round(submissions_per_minute, 2),
        "recent_submissions": recent_submissions,
        "window_minutes": window_minutes
    }

@app.post("/finalize_round/{round_id}")
def finalize_round(round_id: int, db: Session = Depends(get_db)):
    """
    Finalize a round: evaluate all submissions and determine winners.
    Called at round deadline.
    """
    round_obj = db.query(models.CompetitionRound).filter(
        models.CompetitionRound.id == round_id
    ).first()
    
    if not round_obj:
        raise HTTPException(status_code=404, detail="Round not found")
    
    if round_obj.status != "active":
        raise HTTPException(status_code=400, detail="Round already finalized")
    
    # Mark as evaluating
    round_obj.status = "evaluating"
    db.commit()
    
    # Get all validated submissions for this round
    submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.round_id == round_id)
        .filter(models.SpeedSubmission.validated == True)
        .all()
    )
    
    if not submissions:
        print(f"[ROUND] No validated submissions in round {round_id}")
        round_obj.status = "completed"
        db.commit()
        return {"status": "completed", "winners": []}
    
    # Calculate rankings (with first-submission-wins logic)
    rankings = calculate_rankings(submissions, round_obj.baseline_submission_id, db)
    
    # Update round with winner
    if rankings:
        winner = rankings[0]
        round_obj.winner_hotkey = winner["miner_hotkey"]
        round_obj.baseline_submission_id = winner["submission_id"]
        db.commit()
        
        # Mark winning submission as baseline for next round
        winner_submission = db.query(models.SpeedSubmission).filter(
            models.SpeedSubmission.id == winner["submission_id"]
        ).first()
        if winner_submission:
            winner_submission.is_baseline = True
            db.commit()
    
    round_obj.status = "completed"
    db.commit()
    
    print(f"✅ [ROUND] Round {round_id} finalized. Winner: {round_obj.winner_hotkey}")
    
    return {
        "status": "completed",
        "round_id": round_id,
        "winner": round_obj.winner_hotkey,
        "rankings": rankings[:4]  # Top 4
    }
