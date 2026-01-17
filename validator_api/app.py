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
from collections import defaultdict

# Add parent directory to path to import quasar
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from . import models
from . import auth
from . import scoring
from . import longcode_loader
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

# Configuration
CHALLENGE_URL = os.getenv("CHALLENGE_URL", "http://localhost:8080")

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

# Initialize datasets
print("🔄 Initializing datasets in API...")
longcode_dataset = longcode_loader.LongcodeDataset()
print(f"✅ LongcodeDataset ready with {len(longcode_dataset)} samples.")

print(f"✅ Challenge container URL: {CHALLENGE_URL}")
print(f"   (Code execution will be handled by challenge container)")

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

    # 4. For longcode tasks, score is already calculated by /submit_longcode
    # For direct reporting, use the provided score
    base_score = result_in.score if result_in.score is not None else 0.0
    method = "longcode_code_execution"

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

    print(f"🎯 [REPORT_RESULT] Answer extraction: predicted={predicted_answer}, truth={expected_answer}, correct={is_correct}")

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

@app.get("/get_pending_submissions")
def get_pending_submissions(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Returns pending submissions for local Docker evaluation.
    Used by validators to fetch submissions and evaluate them locally.
    """
    pending = (
        db.query(models.Result)
        .join(models.Task, models.Task.id == models.Result.task_id)
        .filter(models.Task.dataset_name == "longcode")
        .filter(models.Result.score.is_(None))
        .order_by(models.Result.created_at.desc())
        .limit(limit)
        .all()
    )
    
    results = []
    for r in pending:
        results.append({
            "id": str(r.id),
            "task_id": r.task_id,
            "miner_hotkey": r.miner_hotkey,
            "miner_uid": r.miner_uid,
            "response_text": r.response_text,
            "created_at": r.created_at.isoformat()
        })
    
    return results

class UpdateScoreRequest(BaseModel):
    result_id: int
    score: float

@app.post("/update_score")
def update_score(
    request: UpdateScoreRequest,
    db: Session = Depends(get_db)
):
    """
    Update the score for a submission.
    Used by validators after local Docker evaluation.
    """
    result = db.query(models.Result).filter(models.Result.id == int(request.result_id)).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result.score = request.score
    
    # Update MinerScore with EMA
    miner_score = db.query(models.MinerScore).filter(
        models.MinerScore.hotkey == result.miner_hotkey
    ).first()
    
    if miner_score:
        alpha = 0.1
        new_score = alpha * request.score + (1 - alpha) * miner_score.score
        miner_score.score = new_score
        miner_score.tasks_completed += 1
        miner_score.last_updated = datetime.utcnow()
    else:
        miner_score = models.MinerScore(
            hotkey=result.miner_hotkey,
            model_name="unknown",
            league="100k",
            score=request.score,
            tasks_completed=1,
            last_updated=datetime.utcnow()
        )
        db.add(miner_score)
    
    db.commit()
    
    return {"status": "updated", "score": request.score}

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

@app.post("/receive_answers")
def receive_answers(
    submission: models.MinerSubmission,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """
    Miners submit their answers directly to this endpoint.
    The API stores the answer for validators to score later.
    Checks if miner is registered on the network.
    """
    print(f"📥 [RECEIVE_ANSWERS] Task: {submission.task_id} | Miner: {hotkey[:8]}...")

    # 1. Verify task exists
    db_task = db.query(models.Task).filter(models.Task.id == submission.task_id).first()
    if not db_task:
        print(f"❌ [RECEIVE_ANSWERS] Task {submission.task_id} not found")
        raise HTTPException(status_code=404, detail="Task not found")

    # 2. Check if miner already submitted for this task
    existing = db.query(models.Result).filter(
        models.Result.task_id == submission.task_id,
        models.Result.miner_hotkey == hotkey
    ).first()

    if existing:
        print(f"⚠️ [RECEIVE_ANSWERS] Miner {hotkey[:8]} already submitted for task {submission.task_id}")
        return {"status": "already_submitted", "result_id": existing.id}

    # 3. Calculate response hash
    resp_text = submission.answer or ""
    resp_hash = hashlib.sha256(resp_text.encode()).hexdigest()

    # 4. Store the submission (score will be calculated by validator later)
    db_result = models.Result(
        task_id=submission.task_id,
        miner_hotkey=hotkey,
        miner_uid=submission.miner_uid,
        response_hash=resp_hash,
        response_text=resp_text,
        score=None  # Will be scored by validator
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)

    print(f"✅ [RECEIVE_ANSWERS] Stored submission from {hotkey[:8]} for task {submission.task_id}")
    return {
        "status": "received",
        "result_id": db_result.id,
        "task_id": submission.task_id,
        "miner_hotkey": hotkey
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
        "avg_score": avg_score
    }

@app.get("/get_task/{task_id}")
def get_task(task_id: str, db: Session = Depends(get_db)):
    """Get task details including test cases for evaluation."""
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    import json
    test_cases = json.loads(task.expected_output) if task.expected_output else []
    
    return {
        "id": task.id,
        "dataset_name": task.dataset_name,
        "prompt": task.prompt,
        "test_cases": test_cases
    }

# ============================================================================
# Longcode Benchmark Endpoints (Code Submission Model)
# ============================================================================

@app.get("/get_longcode_task", response_model=models.MinerTaskResponse)
def get_longcode_task(
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """
    Returns a random sample from longcode dataset.
    Returns task WITHOUT expected_output for miners.
    Includes template_code that miners need to complete.

    Requires signature authentication and miner registration.
    Rate limited to 1 active task per miner and 10 requests per minute.
    """
    # Check rate limit (DDOS protection)
    check_rate_limit(hotkey)

    print(f"📥 [GET_LONGCODE_TASK] Request from {hotkey[:12]}...")

    # Check if miner is registered
    registration = db.query(models.MinerRegistration).filter(
        models.MinerRegistration.hotkey == hotkey
    ).first()

    if not registration:
        print(f"❌ [GET_LONGCODE_TASK] Unregistered miner: {hotkey[:12]}...")
        raise HTTPException(status_code=403, detail="Miner not registered. Please register first.")

    # Update last_seen
    registration.last_seen = int(time.time())
    db.commit()

    # Check if miner has active task (not completed and not expired)
    active_assignment = db.query(models.TaskAssignment).filter(
        models.TaskAssignment.miner_hotkey == hotkey,
        models.TaskAssignment.completed == False,
        models.TaskAssignment.expired == False
    ).first()

    if active_assignment:
        # Check if task is expired (10 minutes = 600 seconds)
        if time.time() - active_assignment.assigned_at < 600:
            print(f"⚠️ [GET_LONGCODE_TASK] Miner {hotkey[:12]}... already has active task")
            raise HTTPException(
                status_code=429,
                detail="Already have active task. Complete it before requesting another."
            )
        else:
            # Mark as expired, allow new task
            print(f"⏰ [GET_LONGCODE_TASK] Task expired for {hotkey[:12]}...")
            active_assignment.expired = True
            db.commit()

    print(f"📥 [GET_LONGCODE_TASK] Request received")

    # Sample from longcode dataset
    samples = longcode_dataset.sample(n=1)

    if not samples:
        print("❌ [GET_LONGCODE_TASK] Failed to sample from dataset!")
        raise HTTPException(status_code=500, detail="Failed to sample from dataset")

    sample = samples[0]
    task_id = f"longcode_{sample.sample_id}_{uuid.uuid4().hex}"

    print(f"📤 [GET_LONGCODE_TASK] Generated task: {task_id}")

    # Store in DB with template_code
    # Serialize test cases to JSON string for storage
    import json
    db_task = models.Task(
        id=task_id,
        dataset_name="longcode",
        task_type="code_injection",
        context=sample.get_prompt_text(),
        prompt=sample.get_prompt_text(),
        expected_output=json.dumps([{"input_code": tc.input_code, "expected_output": tc.expected_output} for tc in sample.test_cases]),
        context_length=len(sample.get_prompt_text()),
        difficulty_level=sample.context_length,
        evaluation_metrics="code_execution",
    )
    try:
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error: failed to persist task")

    # Create task assignment
    assignment = models.TaskAssignment(
        task_id=task_id,
        miner_hotkey=hotkey,
        assigned_at=int(time.time()),
        completed=False
    )
    db.add(assignment)
    db.commit()
    
    # Return with template_code for miners
    response = models.MinerTaskResponse(
        id=db_task.id,
        dataset_name=db_task.dataset_name,
        task_type=db_task.task_type,
        context=db_task.context,
        prompt=db_task.prompt,
        context_length=db_task.context_length,
        difficulty_level=db_task.difficulty_level,
        evaluation_metrics=["code_execution"],
        created_at=db_task.created_at,
        template_code=sample.get_template_code(),
        timeout=sample.timeout
    )
    
    return response

@app.post("/submit_longcode_pending")
def submit_longcode_pending(
    submission: dict,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    task_id = submission.get("task_id")
    code = submission.get("code")

    if not task_id or not code:
        raise HTTPException(status_code=400, detail="Missing task_id or code")

    db_task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found")

    if db_task.dataset_name != "longcode":
        raise HTTPException(status_code=400, detail="Not a longcode task")

    existing_result = db.query(models.Result).filter(
        models.Result.task_id == task_id,
        models.Result.miner_hotkey == hotkey
    ).first()
    if existing_result:
        return {
            "status": "already_submitted",
            "result_id": existing_result.id,
            "score": existing_result.score
        }

    code_hash = hashlib.sha256(code.encode()).hexdigest()
    db_result = models.Result(
        task_id=task_id,
        miner_hotkey=hotkey,
        miner_uid=submission.get("miner_uid", 0),
        response_hash=code_hash,
        response_text=code,
        score=None
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)

    # Mark task assignment as completed
    assignment = db.query(models.TaskAssignment).filter(
        models.TaskAssignment.task_id == task_id,
        models.TaskAssignment.miner_hotkey == hotkey
    ).first()

    if assignment:
        assignment.completed = True
        assignment.completed_at = int(time.time())
        db.commit()
        print(f"✅ [SUBMIT] Task {task_id} marked as completed for {hotkey[:12]}...")
    else:
        print(f"⚠️ [SUBMIT] No assignment found for task {task_id} and miner {hotkey[:12]}...")

    return {
        "status": "stored",
        "result_id": db_result.id,
        "task_id": task_id,
        "miner_hotkey": hotkey
    }

@app.post("/submit_longcode")
def submit_longcode(
    submission: dict,
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """
    Miners submit their completed code for a longcode task.
    The API executes the code against test cases and scores it.
    """
    task_id = submission.get("task_id")
    code = submission.get("code")
    function_name = submission.get("function_name", "solve")
    
    print(f"📥 [SUBMIT_LONGCODE] Task: {task_id} | Miner: {hotkey[:8]}")
    print(f"   Function: {function_name}")
    print(f"   Code length: {len(code) if code else 0} chars")
    
    # 1. Fetch Task details
    db_task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not db_task:
        print(f"❌ [SUBMIT_LONGCODE] Task {task_id} not found in DB")
        raise HTTPException(status_code=404, detail="Task not found")
    
    if db_task.dataset_name != "longcode":
        raise HTTPException(status_code=400, detail="Not a longcode task")
    
    # 2. Check for duplicate submission
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    existing_result = db.query(models.Result).filter(
        models.Result.task_id == task_id,
        models.Result.miner_hotkey == hotkey
    ).first()
    
    if existing_result:
        print(f"⚠️ [SUBMIT_LONGCODE] Miner {hotkey[:8]} already submitted for task {task_id}")
        return {
            "status": "already_submitted",
            "result_id": existing_result.id,
            "score": existing_result.score
        }
    
    # 3. Get test cases from task
    import json
    test_cases_json = db_task.expected_output
    test_cases = json.loads(test_cases_json) if test_cases_json else []
    
    if not test_cases or not isinstance(test_cases, list):
        print(f"❌ [SUBMIT_LONGCODE] No test cases found for task {task_id}")
        raise HTTPException(status_code=500, detail="No test cases for this task")
    
    # 4. Execute code against test cases using challenge container
    print(f"🔬 [SUBMIT_LONGCODE] Executing code against {len(test_cases)} test cases...")
    print(f"   Using challenge container: {CHALLENGE_URL}")

    passed = 0
    failed = 0
    timeouts = 0
    test_results = []

    for i, test_case in enumerate(test_cases):
        test_input = test_case.get("input_code", "")
        expected_output = test_case.get("expected_output")

        print(f"   Test {i+1}/{len(test_cases)}: input={test_input[:50]}...", flush=True)

        try:
            # Call challenge container to execute code
            response = requests.post(
                f"{CHALLENGE_URL}/execute",
                json={
                    "code": code,
                    "function_name": function_name,
                    "test_input": test_input
                },
                timeout=EXECUTION_TIMEOUT + 5
            )
            response.raise_for_status()
            result = response.json()

            if result.get("timeout"):
                print(f"      ⏱️  Timeout", flush=True)
                timeouts += 1
                test_results.append({
                    "test_case": i,
                    "status": "timeout",
                    "expected": expected_output,
                    "actual": None,
                    "error": result.get("error")
                })
            elif not result.get("success"):
                print(f"      ❌ Error: {result.get('error')[:50]}", flush=True)
                failed += 1
                test_results.append({
                    "test_case": i,
                    "status": "error",
                    "expected": expected_output,
                    "actual": None,
                    "error": result.get("error")
                })
            else:
                actual_output = result.get("output")
                # Compare with expected output
                if str(actual_output) == str(expected_output):
                    print(f"      ✅ Pass", flush=True)
                    passed += 1
                    test_results.append({
                        "test_case": i,
                        "status": "passed",
                        "expected": expected_output,
                        "actual": actual_output
                    })
                else:
                    print(f"      ❌ Fail: expected {expected_output}, got {actual_output}", flush=True)
                    failed += 1
                    test_results.append({
                        "test_case": i,
                        "status": "failed",
                        "expected": expected_output,
                        "actual": actual_output
                    })

        except requests.exceptions.Timeout:
            print(f"      ⏱️  Request timeout", flush=True)
            timeouts += 1
            test_results.append({
                "test_case": i,
                "status": "timeout",
                "expected": expected_output,
                "actual": None,
                "error": "Request timeout"
            })
        except Exception as e:
            print(f"      ❌ Exception: {str(e)[:50]}", flush=True)
            failed += 1
            test_results.append({
                "test_case": i,
                "status": "error",
                "expected": expected_output,
                "actual": None,
                "error": str(e)
            })

    total_tests = len(test_cases)
    score = passed / total_tests if total_tests > 0 else 0.0

    evaluation_result = {
        "total_tests": total_tests,
        "passed": passed,
        "failed": failed,
        "timeouts": timeouts,
        "score": score,
        "results": test_results
    }
    
    print(f"📊 [SUBMIT_LONGCODE] Evaluation complete:")
    print(f"   Total: {evaluation_result['total_tests']}")
    print(f"   Passed: {evaluation_result['passed']}")
    print(f"   Failed: {evaluation_result['failed']}")
    print(f"   Timeouts: {evaluation_result['timeouts']}")
    print(f"   Score: {evaluation_result['score']:.4f}")
    
    # 5. Store result
    db_result = models.Result(
        task_id=task_id,
        miner_hotkey=hotkey,
        miner_uid=submission.get("miner_uid", 0),
        response_hash=code_hash,
        response_text=code,
        score=evaluation_result["score"]
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)

    # 6. Update MinerScore (EMA)
    miner_score = db.query(models.MinerScore).filter(
        models.MinerScore.hotkey == hotkey
    ).first()

    if miner_score:
        # Update existing score with EMA
        alpha = 0.1  # EMA smoothing factor
        new_score = alpha * evaluation_result["score"] + (1 - alpha) * miner_score.score
        miner_score.score = new_score
        miner_score.tasks_completed += 1
        miner_score.last_updated = datetime.utcnow()
    else:
        # Create new MinerScore entry
        # Default league and model_name for longcode submissions
        miner_score = models.MinerScore(
            hotkey=hotkey,
            model_name="unknown",  # Could be passed in submission
            league="100k",  # Default league
            score=evaluation_result["score"],
            tasks_completed=1,
            last_updated=datetime.utcnow()
        )
        db.add(miner_score)

    db.commit()

    return {
        "status": "evaluated",
        "result_id": db_result.id,
        "score": evaluation_result["score"],
        "total_tests": evaluation_result["total_tests"],
        "passed": evaluation_result["passed"],
        "failed": evaluation_result["failed"],
        "timeouts": evaluation_result["timeouts"],
        "test_results": evaluation_result["results"]
    }

@app.delete("/delete_submission/{result_id}")
def delete_submission(
    result_id: int,
    db: Session = Depends(get_db)
):
    """Delete a submission by result_id (for cleanup/testing)."""
    result = db.query(models.Result).filter(models.Result.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    db.delete(result)
    db.commit()
    
    print(f"  Deleted submission {result_id} from miner {result.miner_hotkey[:8]}")
    return {"status": "deleted", "result_id": result_id}

class WeightEntry(BaseModel):
    uid: int
    hotkey: str
    weight: float

class GetWeightsResponse(BaseModel):
    epoch: int
    weights: List[WeightEntry]

@app.get("/get_task_stats")
def get_task_stats(db: Session = Depends(get_db)):
    """
    Get task statistics for the system.
    Returns total tasks, completed tasks, pending tasks, and active assignments.
    """
    total_tasks = db.query(models.Task).count()
    completed_tasks = db.query(models.Result).count()
    pending_tasks = db.query(models.Task).filter(
        ~models.Task.id.in_(
            db.query(models.Result.task_id)
        )
    ).count()
    active_assignments = db.query(models.TaskAssignment).filter(
        models.TaskAssignment.completed == False,
        models.TaskAssignment.expired == False
    ).count()

    return {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "pending_tasks": pending_tasks,
        "active_assignments": active_assignments
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
