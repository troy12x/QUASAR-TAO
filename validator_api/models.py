from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict

# SQLAlchemy Models
class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True)
    dataset_name = Column(String)
    task_type = Column(String)
    context = Column(String)
    prompt = Column(String)
    expected_output = Column(String)
    context_length = Column(Integer)
    difficulty_level = Column(String)
    evaluation_metrics = Column(String)  # Stored as comma-separated string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    results = relationship("Result", back_populates="task")

class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, ForeignKey("tasks.id"))
    miner_hotkey = Column(String, index=True)
    miner_uid = Column(Integer)
    response_hash = Column(String, index=True)
    response_text = Column(String)
    score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("Task", back_populates="results")

class MinerScore(Base):
    __tablename__ = "miner_scores"

    hotkey = Column(String, primary_key=True, index=True)
    model_name = Column(String, index=True)  # e.g., "Qwen-2.5-0.5B", "Kimi-48B"
    league = Column(String, index=True)  # e.g., "100k", "200k", ..., "1M"
    score = Column(Float, default=0.0)
    tasks_completed = Column(Integer, default=0)  # Track tasks per league
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MinerRegistration(Base):
    __tablename__ = "miner_registrations"

    hotkey = Column(String, primary_key=True, index=True)
    uid = Column(Integer, nullable=False)
    registered_at = Column(Integer, nullable=False, default=lambda: int(datetime.utcnow().timestamp()))
    last_seen = Column(Integer, nullable=False, default=lambda: int(datetime.utcnow().timestamp()))

class SpeedSubmission(Base):
    __tablename__ = "speed_submissions"

    id = Column(Integer, primary_key=True, index=True)
    miner_hotkey = Column(String, index=True)
    miner_uid = Column(Integer)
    fork_url = Column(String)
    commit_hash = Column(String)
    target_sequence_length = Column(Integer)
    tokens_per_sec = Column(Float)
    vram_mb = Column(Float, nullable=True)
    benchmarks = Column(String, nullable=True)  # JSON string of benchmarks
    signature = Column(String)
    validated = Column(Boolean, default=False)  # Track if submission has been validated
    created_at = Column(DateTime, default=datetime.utcnow)

class TaskAssignment(Base):
    __tablename__ = "task_assignments"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    miner_hotkey = Column(String, nullable=False, index=True)
    assigned_at = Column(Integer, nullable=False, default=lambda: int(datetime.utcnow().timestamp()))
    completed = Column(Boolean, default=False)
    completed_at = Column(Integer, nullable=True)
    expired = Column(Boolean, default=False)

    task = relationship("Task", backref="assignments")

# Pydantic Schemas
class TaskBase(BaseModel):
    dataset_name: str
    task_type: str
    context: str
    prompt: str
    expected_output: str
    context_length: int
    difficulty_level: str
    evaluation_metrics: List[str]

class TaskCreate(TaskBase):
    id: str

class TaskResponse(TaskBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

# Miner-specific task response WITHOUT expected output
class MinerTaskResponse(BaseModel):
    id: str
    dataset_name: str
    task_type: str
    context: str
    prompt: str
    context_length: int
    difficulty_level: str
    evaluation_metrics: List[str]
    created_at: datetime
    template_code: Optional[str] = None  # Template code for miners to complete
    timeout: Optional[int] = None  # Execution timeout

    class Config:
        from_attributes = True

class ResultBase(BaseModel):
    task_id: str
    miner_hotkey: str
    miner_uid: int
    response_text: str
    response_hash: Optional[str] = None # Calculated by API if not provided
    all_classes: Optional[List[str]] = None

class ResultCreate(ResultBase):
    pass

class MinerScoreResponse(BaseModel):
    hotkey: str
    model_name: str
    league: str
    score: float
    tasks_completed: int
    last_updated: datetime

    class Config:
        from_attributes = True

class RegisterMinerRequest(BaseModel):
    hotkey: str
    model_name: str
    league: str  # "100k", "200k", ..., "1M"

class LeagueInfoResponse(BaseModel):
    league: str
    model_name: str
    top_score: float
    top_hotkey: Optional[str]
    active_miners: int

class MinerSubmission(BaseModel):
    task_id: str
    answer: str
    miner_uid: int

class SpeedSubmissionRequest(BaseModel):
    miner_hotkey: str
    fork_url: str
    commit_hash: str
    target_sequence_length: int
    tokens_per_sec: float
    vram_mb: Optional[float] = None
    benchmarks: Optional[Dict[int, Dict[str, float]]] = None
    signature: str

class SpeedSubmissionResponse(BaseModel):
    submission_id: int
    miner_hotkey: str
    fork_url: str
    commit_hash: str
    target_sequence_length: int
    tokens_per_sec: float
    created_at: datetime

    class Config:
        from_attributes = True
