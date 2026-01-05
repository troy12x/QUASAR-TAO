from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
from typing import List, Optional, Dict
import uuid
import sys
import os
import random
import time

# Add parent directory to path to import quasar
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import quasar
from quasar.benchmarks.benchmark_loader import BenchmarkLoader

import hashlib
from . import models
from . import auth
from . import scoring
from .database import engine, get_db

from fastapi.middleware.cors import CORSMiddleware

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Quasar Validator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize BenchmarkLoader
print("🔄 Initializing BenchmarkLoader in API...")
loader = BenchmarkLoader(config={
    'mrcr': {
        'enabled': True,
        'n_needles_range': [2, 4, 8]
    }
})
print("✅ BenchmarkLoader ready.")

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

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/get_task", response_model=models.MinerTaskResponse)
def get_task(max_context_length: Optional[int] = None, db: Session = Depends(get_db), hotkey: str = Depends(auth.verify_signature)):
    """
    Generates a new production task using BenchmarkLoader.
    Returns task WITHOUT expected_output for miners.
    """
    print(f"📥 [GET_TASK] Request from {hotkey[:8]}... (max_ctx: {max_context_length})")
    difficulty = random.choice(["easy", "medium", "hard", "extreme"])
    print(f"🎲 [GET_TASK] Selected difficulty: {difficulty}")
    tasks = loader.load_benchmark_tasks(1, benchmark_types=['longbench'], difficulty=difficulty, max_context_length=max_context_length)
    
    if not tasks:
        print("❌ [GET_TASK] Failed to generate task!")
        raise HTTPException(status_code=500, detail="Failed to generate task")
    
    task = tasks[0]
    print(f"📤 [GET_TASK] Generated task: {task.task_id} (len: {task.context_length})")
    
    # Store in DB
    db_task = models.Task(
        id=task.task_id,
        dataset_name=task.dataset_name,
        task_type=task.task_type,
        context=task.context,
        prompt=task.prompt,
        expected_output=task.expected_output,
        context_length=task.context_length,
        difficulty_level=task.difficulty_level,
        evaluation_metrics=",".join(task.evaluation_metrics)
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    
    # Return WITHOUT expected_output for miners
    return models.MinerTaskResponse(
        id=db_task.id,
        dataset_name=db_task.dataset_name,
        task_type=db_task.task_type,
        context=db_task.context,
        prompt=db_task.prompt,
        context_length=db_task.context_length,
        difficulty_level=db_task.difficulty_level,
        evaluation_metrics=db_task.evaluation_metrics.split(",") if db_task.evaluation_metrics else [],
        created_at=db_task.created_at
    )

@app.get("/get_task_details/{task_id}", response_model=models.MinerTaskResponse)
def get_task_details(task_id: str, db: Session = Depends(get_db), hotkey: str = Depends(auth.verify_signature)):
    """
    Returns the details of a specific task WITHOUT expected_output.
    Used by miners to fetch heavy context data.
    """
    print(f"📥 [GET_TASK_DETAILS] Request for {task_id} from {hotkey[:8]}...")
    db_task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not db_task:
        print(f"❌ [GET_TASK_DETAILS] Task {task_id} not found!")
        raise HTTPException(status_code=404, detail="Task not found")
        
    # Return WITHOUT expected_output for miners
    return models.MinerTaskResponse(
        id=db_task.id,
        dataset_name=db_task.dataset_name,
        task_type=db_task.task_type,
        context=db_task.context,
        prompt=db_task.prompt,
        context_length=db_task.context_length,
        difficulty_level=db_task.difficulty_level,
        evaluation_metrics=db_task.evaluation_metrics.split(",") if db_task.evaluation_metrics else [],
        created_at=db_task.created_at
    )

@app.post("/report_result")
def report_result(
    result_in: models.ResultCreate,
    db: Session = Depends(get_db),
    validator_hotkey: str = Depends(auth.verify_validator_signature)  # Only validators can report
):
    """
    Validators report miner responses to the API.
    The API performs authoritative scoring and league-based competition.
    """
    print(f"📥 [REPORT_RESULT] Task: {result_in.task_id} | Miner: {result_in.miner_hotkey[:8]} (UID: {result_in.miner_uid})")

    # 1. Fetch Task details
    db_task = db.query(models.Task).filter(models.Task.id == result_in.task_id).first()
    if not db_task:
        print(f"❌ [REPORT_RESULT] Task {result_in.task_id} not found in DB")
        raise HTTPException(status_code=404, detail="Task not found")

    # 2. Calculate response hash if not provided
    resp_text = result_in.response_text or ""
    resp_hash = result_in.response_hash or hashlib.sha256(resp_text.encode()).hexdigest()

    # 3. Duplicate detection
    existing_result = db.query(models.Result).filter(
        models.Result.task_id == result_in.task_id,
        models.Result.response_hash == resp_hash
    ).first()

    if existing_result:
        if existing_result.miner_hotkey != result_in.miner_hotkey:
            print(f"⚠️ [REPORT_RESULT] Duplicate response from different miner ({result_in.miner_hotkey[:8]})")
            return {"status": "rejected", "reason": "duplicate_response"}
        print(f"ℹ️ [REPORT_RESULT] Already reported by {result_in.miner_hotkey[:8]}")
        return {"status": "already_reported", "score": existing_result.score}

    # 4. Calculate base accuracy score (0.0 - 1.0)
    base_score, method = scoring.calculate_score(
        response_text=resp_text,
        expected_output=db_task.expected_output,
        dataset_name=db_task.dataset_name,
        context_length=db_task.context_length,
        all_classes=result_in.all_classes
    )

    # 5. Determine league from context length
    league = get_league(db_task.context_length)
    league_multiplier = LEAGUE_MULTIPLIERS.get(league, 1.0)

    # 6. Find miner's registration (model_name, league)
    miner_registrations = db.query(models.MinerScore).filter(
        models.MinerScore.hotkey == result_in.miner_hotkey,
        models.MinerScore.league == league
    ).all()

    if not miner_registrations:
        print(f"⚠️ [REPORT_RESULT] Miner {result_in.miner_hotkey[:8]} not registered for league {league}")
        # Still save the result but can't calculate league-based reward
        db_result = models.Result(
            task_id=result_in.task_id,
            miner_hotkey=result_in.miner_hotkey,
            miner_uid=result_in.miner_uid,
            response_hash=resp_hash,
            response_text=resp_text,
            score=base_score
        )
        db.add(db_result)
        db.commit()
        return {"status": "reported_no_league", "score": base_score}

    # 7. For each model registration, calculate league-based reward
    for reg in miner_registrations:
        model_name = reg.model_name

        # Get top score for this (league, model) combo
        top_entry = db.query(models.MinerScore).filter(
            models.MinerScore.league == league,
            models.MinerScore.model_name == model_name,
            models.MinerScore.tasks_completed >= 10
        ).order_by(models.MinerScore.score.desc()).first()

        if top_entry and top_entry.score > 0:
            # Normalize against top performer
            normalized_score = base_score / top_entry.score
        else:
            # No top performer yet, use base score
            normalized_score = base_score

        # Apply league multiplier
        final_score = normalized_score * league_multiplier
        final_score = max(0.0, min(final_score, 3.0))  # Clamp to max league multiplier

        # Update miner's EMA score for this (model, league)
        alpha = 0.1
        reg.score = alpha * final_score + (1 - alpha) * reg.score
        reg.tasks_completed += 1

        print(f"🎯 [REPORT_RESULT] {model_name}/{league}: base={base_score:.4f}, norm={normalized_score:.4f}, final={final_score:.4f}")

    # 8. Save result
    db_result = models.Result(
        task_id=result_in.task_id,
        miner_hotkey=result_in.miner_hotkey,
        miner_uid=result_in.miner_uid,
        response_hash=resp_hash,
        response_text=resp_text,
        score=base_score  # Store base score, not league-adjusted
    )
    db.add(db_result)
    db.commit()

    # Return the final score from the first registration
    first_reg = miner_registrations[0]
    return {
        "status": "reported",
        "base_score": base_score,
        "league": league,
        "league_multiplier": league_multiplier,
        "final_score": first_reg.score
    }

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
    db.commit()

    print(f"✅ [REGISTER] Miner {hotkey[:8]} registered for {req.model_name} in {req.league}")
    return {
        "status": "registered",
        "hotkey": hotkey,
        "model_name": req.model_name,
        "league": req.league
    }

@app.get("/get_league_info/{league}/{model_name}", response_model=models.LeagueInfoResponse)
def get_league_info(
    league: str,
    model_name: str,
    db: Session = Depends(get_db)
):
    """
    Get top score and stats for a specific (league, model) combination.
    Used by miners to see competition before registering.
    """
    if league not in LEAGUES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid league. Must be one of: {', '.join(LEAGUES)}"
        )

    # Get top score for this (league, model) combo
    top_entry = db.query(models.MinerScore).filter(
        models.MinerScore.league == league,
        models.MinerScore.model_name == model_name,
        models.MinerScore.tasks_completed >= 10  # Only consider miners with 10+ tasks
    ).order_by(models.MinerScore.score.desc()).first()

    # Count active miners (with 10+ tasks)
    active_count = db.query(models.MinerScore).filter(
        models.MinerScore.league == league,
        models.MinerScore.model_name == model_name,
        models.MinerScore.tasks_completed >= 10
    ).count()

    return models.LeagueInfoResponse(
        league=league,
        model_name=model_name,
        top_score=top_entry.score if top_entry else 0.0,
        top_hotkey=top_entry.hotkey if top_entry else None,
        active_miners=active_count
    )

@app.get("/get_scores", response_model=List[models.MinerScoreResponse])
def get_scores(
    league: Optional[str] = None,
    model_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Returns the authoritative scores for all miners.
    Can filter by league and/or model_name.
    Used by validators to set weights on-chain.
    """
    query = db.query(models.MinerScore)

    if league:
        if league not in LEAGUES:
            raise HTTPException(status_code=400, detail=f"Invalid league: {league}")
        query = query.filter(models.MinerScore.league == league)

    if model_name:
        query = query.filter(models.MinerScore.model_name == model_name)

    return query.all()

@app.get("/stats/miner/{hotkey}")
def get_miner_stats(hotkey: str, db: Session = Depends(get_db)):
    """
    Returns detailed statistics for a specific miner for the dashboard.
    """
    # 1. Basic Counts
    total = db.query(models.Result).filter(models.Result.miner_hotkey == hotkey).count()
    accepted = db.query(models.Result).filter(models.Result.miner_hotkey == hotkey, models.Result.score > 0).count()
    rejected = db.query(models.Result).filter(models.Result.miner_hotkey == hotkey, models.Result.score == 0).count()
    
    # 2. Aggregates
    avg_score = db.query(func.avg(models.Result.score)).filter(models.Result.miner_hotkey == hotkey).scalar() or 0.0
    
    # 3. EMA Score (from MinerScore table)
    miner_score_entry = db.query(models.MinerScore).filter(models.MinerScore.hotkey == hotkey).first()
    ema_score = miner_score_entry.score if miner_score_entry else 0.0

    return {
        "hotkey": hotkey,
        "total_submissions": total,
        "accepted": accepted,
        "rejected": rejected,
        "approval_rate": (accepted / total * 100) if total > 0 else 0.0,
        "average_score": float(avg_score),
        "ema_score": ema_score,
        "last_updated": datetime.utcnow()
    }

@app.get("/stats/global")
def get_global_stats(db: Session = Depends(get_db)):
    """
    Returns global network statistics for the dashboard.
    """
    # 1. Counts
    total_miners = db.query(models.MinerScore).count()
    total_submissions = db.query(models.Result).count()
    accepted = db.query(models.Result).filter(models.Result.score > 0).count()
    rejected = db.query(models.Result).filter(models.Result.score == 0).count()
    
    # 2. Avg Score
    avg_score = db.query(func.avg(models.Result.score)).scalar() or 0.0
    
    return {
        "total_miners": total_miners,
        "total_submissions": total_submissions,
        "accepted": accepted,
        "rejected": rejected,
        "approval_rate": (accepted / total_submissions * 100) if total_submissions > 0 else 0.0,
        "average_score": float(avg_score),
        "last_updated": datetime.utcnow()
    }
