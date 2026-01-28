# QUASAR-SUBNET: Updated Implementation Plan

## Overview
This plan implements a **round-based competition system** with 2-day rounds, first-submission-wins logic, IP banning, dynamic validation, and baseline comparison for progressive improvement.

## Key Requirements Summary

1. **Reward Distribution:** 60%, 25%, 10%, 5% for top 4
2. **Round-Based:** 2-day rounds with deadline evaluation
3. **First Submission Wins:** Timestamp-based tiebreaker
4. **IP Banning:** Ban IPs after repeated failures
5. **Dynamic Validation:** Frequency based on submission rate
6. **Baseline System:** Round 2+ uses previous winner as baseline

---

## PHASE 1: Database Schema Updates ⭐ CRITICAL

### Objective
Add tables and columns to support rounds, IP tracking, baselines, and first-submission-wins.

### Files to Modify
1. `validator_api/models.py` - Add new models
2. `validator_api/database.py` - Migration scripts

### Implementation

#### 1.1 Add Round Model
**File:** `validator_api/models.py`

```python
class CompetitionRound(Base):
    """Represents a competition round."""
    __tablename__ = "competition_rounds"
    
    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, unique=True, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    status = Column(String, default="active")  # "active", "evaluating", "completed"
    baseline_submission_id = Column(Integer, ForeignKey("speed_submissions.id"), nullable=True)
    winner_hotkey = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    baseline_submission = relationship("SpeedSubmission", foreign_keys=[baseline_submission_id])
    submissions = relationship("SpeedSubmission", back_populates="round")
```

#### 1.2 Add IP Tracking Model
**File:** `validator_api/models.py`

```python
class IPBan(Base):
    """Track IP addresses and ban status."""
    __tablename__ = "ip_bans"
    
    ip_address = Column(String, primary_key=True, index=True)
    failure_count = Column(Integer, default=0)
    last_failure_time = Column(DateTime, nullable=True)
    banned_until = Column(DateTime, nullable=True)
    is_banned = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

#### 1.3 Update SpeedSubmission Model
**File:** `validator_api/models.py` - Add to existing `SpeedSubmission` class

```python
# Add these columns to SpeedSubmission
round_id = Column(Integer, ForeignKey("competition_rounds.id"), nullable=True)
ip_address = Column(String, nullable=True)  # Track IP for banning
is_baseline = Column(Boolean, default=False)  # Mark baseline submissions
solution_hash = Column(String, nullable=True)  # Hash of solution for duplicate detection

# Relationships
round = relationship("CompetitionRound", back_populates="submissions")
```

#### 1.4 Add Solution Hash Calculation
**File:** `validator_api/app.py` - Add helper function

```python
import hashlib
import json

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
```

---

## PHASE 2: Round Management System ⭐ CRITICAL

### Objective
Implement round creation, deadline handling, and round transitions.

### Files to Modify
1. `validator_api/app.py` - Add round endpoints
2. `validator_api/models.py` - Add round response models

### Implementation

#### 2.1 Add Round Response Models
**File:** `validator_api/models.py`

```python
class RoundResponse(BaseModel):
    """Round information response."""
    id: int
    round_number: int
    start_time: datetime
    end_time: datetime
    status: str
    time_remaining_seconds: int
    baseline_submission_id: Optional[int] = None
    winner_hotkey: Optional[str] = None
    total_submissions: int = 0
    
    class Config:
        from_attributes = True

class CreateRoundRequest(BaseModel):
    """Request to create a new round."""
    duration_hours: int = 48  # Default 2 days
    baseline_submission_id: Optional[int] = None  # For round 2+
```

#### 2.2 Add Round Endpoints
**File:** `validator_api/app.py`

```python
@app.get("/get_current_round", response_model=models.RoundResponse)
def get_current_round(db: Session = Depends(get_db)):
    """Get the current active round."""
    current_round = (
        db.query(models.CompetitionRound)
        .filter(models.CompetitionRound.status == "active")
        .order_by(models.CompetitionRound.round_number.desc())
        .first()
    )
    
    if not current_round:
        # Create first round if none exists
        from datetime import datetime, timedelta
        current_round = models.CompetitionRound(
            round_number=1,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=48),
            status="active"
        )
        db.add(current_round)
        db.commit()
        db.refresh(current_round)
    
    # Calculate time remaining
    now = datetime.utcnow()
    time_remaining = max(0, int((current_round.end_time - now).total_seconds()))
    
    # Count submissions in this round
    submission_count = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.round_id == current_round.id)
        .count()
    )
    
    return models.RoundResponse(
        id=current_round.id,
        round_number=current_round.round_number,
        start_time=current_round.start_time,
        end_time=current_round.end_time,
        status=current_round.status,
        time_remaining_seconds=time_remaining,
        baseline_submission_id=current_round.baseline_submission_id,
        winner_hotkey=current_round.winner_hotkey,
        total_submissions=submission_count
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
    from datetime import datetime, timedelta
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
    
    # Get all submissions for this round
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
```

---

## PHASE 3: First-Submission-Wins Logic ⭐ CRITICAL

### Objective
Implement tiebreaker logic where identical solutions are ranked by submission timestamp (first wins).

### Files to Modify
1. `validator_api/app.py` - Update ranking calculation

### Implementation

#### 3.1 Update Ranking Function
**File:** `validator_api/app.py` - Add new function

```python
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
```

#### 3.2 Update Submission Endpoint
**File:** `validator_api/app.py` - Update `submit_kernel()` function

Add solution hash calculation when creating submission:

```python
# In submit_kernel() function, after creating new_submission:
# Calculate solution hash for duplicate detection
new_submission.solution_hash = calculate_solution_hash(
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

if current_round:
    new_submission.round_id = current_round.id
else:
    # Create first round if none exists
    from datetime import datetime, timedelta
    current_round = models.CompetitionRound(
        round_number=1,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(hours=48),
        status="active"
    )
    db.add(current_round)
    db.commit()
    db.refresh(current_round)
    new_submission.round_id = current_round.id

# Extract IP address from request (if available)
from fastapi import Request as FastAPIRequest
# Note: You'll need to pass Request to the endpoint
# @app.post("/submit_kernel")
# def submit_kernel(
#     req: models.SpeedSubmissionRequest,
#     request: FastAPIRequest,  # ADD THIS
#     db: Session = Depends(get_db),
#     hotkey: str = Depends(auth.verify_signature)
# ):
client_ip = request.client.host if hasattr(request, 'client') else None
new_submission.ip_address = client_ip
```

---

## PHASE 4: IP Banning System ⭐ CRITICAL

### Objective
Track failed submissions by IP and ban after repeated failures.

### Files to Modify
1. `validator_api/app.py` - Add IP tracking and banning
2. `validator_api/models.py` - Already added IPBan model

### Implementation

#### 4.1 Add IP Banning Logic
**File:** `validator_api/app.py`

```python
# Configuration
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
```

#### 4.2 Update Submission Endpoint with IP Banning
**File:** `validator_api/app.py` - Update `submit_kernel()`

```python
@app.post("/submit_kernel")
def submit_kernel(
    req: models.SpeedSubmissionRequest,
    request: FastAPIRequest,  # ADD THIS
    db: Session = Depends(get_db),
    hotkey: str = Depends(auth.verify_signature)
):
    """Submit kernel optimization results from miners."""
    # Extract IP address
    client_ip = request.client.host if hasattr(request, 'client') else None
    
    # Check IP ban
    is_banned, ban_reason = check_ip_ban(client_ip, db)
    if is_banned:
        raise HTTPException(
            status_code=403,
            detail=f"IP address banned: {ban_reason}"
        )
    
    # ... rest of existing submission logic ...
    
    # After validation fails (if it does), record failure
    # This should be called from validator when validation fails
    # For now, we'll add it to the validation endpoint
```

#### 4.3 Update Validator to Record Failures
**File:** `neurons/validator.py` - Update `validate_submission()`

```python
# In validate_submission(), when validation fails:
if not imports_valid or actual_performance <= 0:
    # Record failure for IP banning
    if submission.get("ip_address"):
        try:
            requests.post(
                f"{VALIDATOR_API_URL}/record_failure",
                json={"ip_address": submission.get("ip_address")},
                timeout=10
            )
        except:
            pass  # Don't fail validation if IP tracking fails
    
    return {
        "submission_id": submission.get("id"),
        "miner_hotkey": submission.get("miner_hotkey"),
        "score": 0.0,
        "is_valid": False,
        "errors": import_errors if not imports_valid else ["Performance test failed"]
    }
```

Add endpoint to API:
```python
@app.post("/record_failure")
def record_failure_endpoint(req: dict, db: Session = Depends(get_db)):
    """Record a failed submission for IP tracking."""
    ip_address = req.get("ip_address")
    if ip_address:
        record_failure(ip_address, db)
    return {"status": "ok"}
```

---

## PHASE 5: Dynamic Validation Frequency 🔥 HIGH

### Objective
Adjust validator polling interval based on submission rate.

### Files to Modify
1. `neurons/validator.py` - Add dynamic polling logic
2. `validator_api/app.py` - Add submission rate endpoint

### Implementation

#### 5.1 Add Submission Rate Endpoint
**File:** `validator_api/app.py`

```python
@app.get("/get_submission_rate")
def get_submission_rate(
    window_minutes: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get current submission rate (submissions per minute).
    Used by validators to adjust polling frequency.
    """
    from datetime import timedelta
    
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
```

#### 5.2 Update Validator with Dynamic Polling
**File:** `neurons/validator.py` - Update `forward()` method

```python
async def forward(self):
    """Main validation loop with dynamic polling."""
    print("[VALIDATOR] ➡️ Starting validation cycle...", flush=True)
    
    try:
        # Check submission rate
        try:
            response = requests.get(
                f"{VALIDATOR_API_URL}/get_submission_rate",
                params={"window_minutes": 10},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                submissions_per_min = data.get("submissions_per_minute", 0)
                
                # Adjust polling interval based on rate
                if submissions_per_min > 5:
                    # High activity: poll every 1 minute
                    polling_interval = 60
                elif submissions_per_min > 1:
                    # Medium activity: poll every 2 minutes
                    polling_interval = 120
                else:
                    # Low activity: poll every 5 minutes
                    polling_interval = 300
                
                print(f"[VALIDATOR] 📊 Submission rate: {submissions_per_min:.2f}/min, "
                      f"polling every {polling_interval}s", flush=True)
            else:
                polling_interval = 300  # Default
        except Exception as e:
            print(f"[VALIDATOR] ⚠️ Failed to get submission rate: {e}", flush=True)
            polling_interval = 300  # Default
        
        # Evaluate performance submissions
        evaluated_scores = self.evaluate_performance_submissions()
        
        if not evaluated_scores:
            print("[VALIDATOR] ⚠️ No pending submissions", flush=True)
        else:
            print(f"[VALIDATOR] ✅ Evaluated {len(evaluated_scores)} submissions", flush=True)
        
        # Fetch and submit weights (only at round end)
        # Check if current round is ending soon
        try:
            response = requests.get(f"{VALIDATOR_API_URL}/get_current_round", timeout=10)
            if response.status_code == 200:
                round_data = response.json()
                time_remaining = round_data.get("time_remaining_seconds", 3600)
                
                # If round ends in < 5 minutes, finalize and submit weights
                if 0 < time_remaining < 300:
                    print(f"[VALIDATOR] ⏰ Round ending in {time_remaining}s, finalizing...", flush=True)
                    # Trigger round finalization
                    requests.post(
                        f"{VALIDATOR_API_URL}/finalize_round/{round_data['id']}",
                        timeout=60
                    )
                    # Then fetch and submit weights
                    await self.fetch_and_submit_weights_to_chain()
        except Exception as e:
            print(f"[VALIDATOR] ⚠️ Round check failed: {e}", flush=True)
        
        # Wait before next cycle
        print(f"[VALIDATOR] ⏱️ Waiting {polling_interval}s...", flush=True)
        time.sleep(polling_interval)
        
    except Exception as e:
        print(f"[VALIDATOR] ❌ Error: {e}", flush=True)
        traceback.print_exc()
        time.sleep(300)  # Wait 5 minutes on error
```

---

## PHASE 6: Updated Reward Distribution ⭐ CRITICAL

### Objective
Implement 60%, 25%, 10%, 5% distribution for top 4.

### Files to Modify
1. `validator_api/app.py` - Update `get_weights()` endpoint

### Implementation

#### 6.1 Update Weight Distribution
**File:** `validator_api/app.py` - Replace `get_weights()` function

```python
# Updated reward distribution
REWARD_DISTRIBUTION = [0.60, 0.25, 0.10, 0.05]  # 60%, 25%, 10%, 5%
TOP_N_MINERS = 4

@app.get("/get_weights", response_model=GetWeightsResponse)
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
        return GetWeightsResponse(epoch=int(time.time()), weights=[])
    
    # Get all validated submissions for this round
    submissions = (
        db.query(models.SpeedSubmission)
        .filter(models.SpeedSubmission.round_id == round_obj.id)
        .filter(models.SpeedSubmission.validated == True)
        .all()
    )
    
    if not submissions:
        return GetWeightsResponse(epoch=int(time.time()), weights=[])
    
    # Calculate rankings with first-submission-wins
    rankings = calculate_rankings(submissions, round_obj.baseline_submission_id, db)
    
    if not rankings:
        return GetWeightsResponse(epoch=int(time.time()), weights=[])
    
    # Distribute weights to top 4
    weights = []
    print(f"[WEIGHTS] Distributing rewards for round {round_obj.round_number}:")
    
    for i, ranking in enumerate(rankings[:TOP_N_MINERS]):
        weight = REWARD_DISTRIBUTION[i] if i < len(REWARD_DISTRIBUTION) else 0.0
        
        # Get UID from MinerRegistration
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
```

---

## PHASE 7: Baseline Comparison System ⭐ CRITICAL

### Objective
Round 2+ submissions must beat the previous round's winner (baseline).

### Implementation

The baseline comparison is already implemented in `calculate_rankings()` function (Phase 3). We just need to ensure it's used correctly:

1. **Round 1:** No baseline, all submissions eligible
2. **Round 2+:** Baseline is previous round's winner
3. **Validation:** Submissions below baseline are filtered out in ranking

The logic is in `calculate_rankings()`:
```python
if baseline:
    # Calculate baseline weighted score
    baseline_weighted = baseline.tokens_per_sec * baseline_multiplier
    
    # Calculate submission weighted score
    sub_weighted = sub.tokens_per_sec * sub_multiplier
    
    if sub_weighted <= baseline_weighted:
        continue  # Skip - doesn't beat baseline
```

---

## Testing Checklist

### Test 1: Round Creation
- [ ] Create first round
- [ ] Verify 48-hour duration
- [ ] Check round status

### Test 2: First-Submission-Wins
- [ ] Submit two identical solutions
- [ ] Verify first submission ranks higher
- [ ] Check timestamp sorting

### Test 3: IP Banning
- [ ] Submit 5 invalid submissions from same IP
- [ ] Verify IP is banned
- [ ] Check ban expiration after 24 hours

### Test 4: Dynamic Validation
- [ ] Submit many submissions quickly
- [ ] Verify validator polls more frequently
- [ ] Check polling slows when idle

### Test 5: Reward Distribution
- [ ] Submit 4+ miners with different scores
- [ ] Finalize round
- [ ] Verify weights: 60%, 25%, 10%, 5%

### Test 6: Baseline System
- [ ] Complete Round 1
- [ ] Start Round 2 with baseline
- [ ] Submit solution below baseline
- [ ] Verify it's filtered out
- [ ] Submit solution above baseline
- [ ] Verify it's ranked

---

## Summary

**Total Implementation Time: ~20-24 hours**

- Phase 1 (Database): 3 hours
- Phase 2 (Rounds): 4 hours
- Phase 3 (First-Wins): 2 hours
- Phase 4 (IP Banning): 3 hours
- Phase 5 (Dynamic Validation): 2 hours
- Phase 6 (Rewards): 2 hours
- Phase 7 (Baseline): 1 hour (mostly done)
- Testing: 4-6 hours

**Ready to start implementation!**
