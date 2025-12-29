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

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/get_task", response_model=models.TaskResponse)
def get_task(max_context_length: Optional[int] = None, db: Session = Depends(get_db), hotkey: str = Depends(auth.verify_signature)):
    """
    Generates a new production task using BenchmarkLoader.
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
    
    # Return with list formatted metrics
    return models.TaskResponse(
        id=db_task.id,
        dataset_name=db_task.dataset_name,
        task_type=db_task.task_type,
        context=db_task.context,
        prompt=db_task.prompt,
        expected_output=db_task.expected_output,
        context_length=db_task.context_length,
        difficulty_level=db_task.difficulty_level,
        evaluation_metrics=db_task.evaluation_metrics.split(",") if db_task.evaluation_metrics else [],
        created_at=db_task.created_at
    )

@app.get("/get_task_details/{task_id}", response_model=models.TaskResponse)
def get_task_details(task_id: str, db: Session = Depends(get_db), hotkey: str = Depends(auth.verify_signature)):
    """
    Returns the full details of a specific task.
    Used by miners to fetch heavy context data.
    """
    print(f"📥 [GET_TASK_DETAILS] Request for {task_id} from {hotkey[:8]}...")
    db_task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not db_task:
        print(f"❌ [GET_TASK_DETAILS] Task {task_id} not found!")
        raise HTTPException(status_code=404, detail="Task not found")
        
    return models.TaskResponse(
        id=db_task.id,
        dataset_name=db_task.dataset_name,
        task_type=db_task.task_type,
        context=db_task.context,
        prompt=db_task.prompt,
        expected_output=db_task.expected_output,
        context_length=db_task.context_length,
        difficulty_level=db_task.difficulty_level,
        evaluation_metrics=db_task.evaluation_metrics.split(",") if db_task.evaluation_metrics else [],
        created_at=db_task.created_at
    )

@app.post("/report_result")
def report_result(
    result_in: models.ResultCreate, 
    db: Session = Depends(get_db), 
    validator_hotkey: str = Depends(auth.verify_signature)
):
    """
    Validators report miner responses to the API.
    The API performs authoritative scoring and duplicate detection.
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
        # If same miner, we'll just update or ignore (idempotent)
        print(f"ℹ️ [REPORT_RESULT] Already reported by {result_in.miner_hotkey[:8]}")
        return {"status": "already_reported", "score": existing_result.score}

    # 4. Authoritative Scoring
    final_score = scoring.calculate_score(
        response_text=resp_text,
        expected_output=db_task.expected_output,
        dataset_name=db_task.dataset_name,
        context_length=db_task.context_length,
        all_classes=result_in.all_classes
    )

    # 5. Save Result
    db_result = models.Result(
        task_id=result_in.task_id,
        miner_hotkey=result_in.miner_hotkey,
        miner_uid=result_in.miner_uid,
        response_hash=resp_hash,
        response_text=resp_text,
        score=final_score
    )
    db.add(db_result)
    
    # 6. Update Aggregate Miner Score (EMA)
    db_score = db.query(models.MinerScore).filter(models.MinerScore.hotkey == result_in.miner_hotkey).first()
    if not db_score:
        db_score = models.MinerScore(hotkey=result_in.miner_hotkey, score=final_score)
        db.add(db_score)
    else:
        alpha = 0.1
        db_score.score = alpha * final_score + (1 - alpha) * db_score.score
    
    db.commit()
    print(f"🎯 [REPORT_RESULT] Scored: {final_score:.4f} | Saved to DB.")
    return {"status": "reported", "score": final_score}

@app.get("/get_scores", response_model=List[models.MinerScoreResponse])
def get_scores(db: Session = Depends(get_db)):
    """
    Returns the authoritative scores for all miners.
    Used by validators to set weights on-chain.
    """
    return db.query(models.MinerScore).all()

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
