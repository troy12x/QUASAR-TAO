"""Challenge Server for QUASAR-SUBNET

Evaluates miner submissions by querying their model endpoints
and scoring answers against the docmath dataset.
"""

import os
import sys
import json
import time
import uuid
import requests
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path to import validator_api modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from validator_api.docmath_loader import DocmathDataset
from validator_api.answer_extractor import extract_and_compare

# Configuration
DATASET_PATH = os.getenv("DATASET_PATH", "/data/docmath.jsonl")
CHALLENGE_HOST = os.getenv("CHALLENGE_HOST", "0.0.0.0")
CHALLENGE_PORT = int(os.getenv("CHALLENGE_PORT", "8080"))
SAMPLES_PER_EVALUATION = int(os.getenv("SAMPLES_PER_EVALUATION", "50"))
TIMEOUT_SECS = int(os.getenv("TIMEOUT_SECS", "300"))

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=CHALLENGE_PORT)
args = parser.parse_args()
CHALLENGE_PORT = args.port

# Initialize dataset
print(f"🔄 Loading docmath dataset from {DATASET_PATH}...")
dataset = DocmathDataset(DATASET_PATH)
print(f"✅ Dataset loaded with {len(dataset)} samples")

# Create FastAPI app
app = FastAPI(title="QUASAR-SUBNET Challenge Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class EvaluationRequest(BaseModel):
    request_id: str
    submission_id: str
    participant_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    epoch: int
    deadline: int


class EvaluationResponse(BaseModel):
    request_id: str
    success: bool
    error: str | None = None
    score: float
    results: Dict[str, Any]
    execution_time_ms: int
    cost: float | None = None


class HealthResponse(BaseModel):
    status: str
    dataset_size: int
    version: str


class ConfigResponse(BaseModel):
    challenge_id: str
    version: str
    evaluation_config: Dict[str, Any]


class MinerSubmission(BaseModel):
    task_id: str
    answer: str
    miner_uid: int = None


def query_miner_model(model_endpoint: str, prompt: str, api_key: str = None) -> str:
    """Query the miner's model endpoint"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "prompt": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.0
    }
    
    try:
        print(f"  POST {model_endpoint}")
        print(f"  Payload: {payload}")
        response = requests.post(
            model_endpoint,
            json=payload,
            headers=headers,
            timeout=30
        )
        print(f"  Response status: {response.status_code}")
        print(f"  Response body: {response.text[:500]}")
        response.raise_for_status()
        data = response.json()
        output = data.get("output", "")
        print(f"  Extracted output: {output[:100]}")
        return output
    except Exception as e:
        print(f"  ❌ Error querying miner model: {type(e).__name__}: {e}")
        return ""


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        dataset_size=len(dataset),
        version="1.0.0"
    )


@app.get("/config", response_model=ConfigResponse)
def get_config():
    """Get challenge configuration"""
    return ConfigResponse(
        challenge_id="quasar-subnet",
        version="1.0.0",
        evaluation_config={
            "samples_per_evaluation": SAMPLES_PER_EVALUATION,
            "timeout_secs": TIMEOUT_SECS,
            "max_retries": 3
        }
    )


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest):
    """Evaluate a miner submission"""
    print(f"\n📥 [EVALUATE] Request: {request.request_id}")
    print(f"  Submission ID: {request.submission_id}")
    print(f"  Participant ID: {request.participant_id}")
    
    start_time = time.time()
    
    try:
        # Get miner model endpoint
        model_endpoint = request.data.get("model_endpoint")
        if not model_endpoint:
            model_endpoint = request.data.get("miner_endpoint")
        model_api_key = request.data.get("model_api_key")
        
        if not model_endpoint:
            raise HTTPException(status_code=400, detail="Missing model_endpoint in request data")

        model_endpoint = str(model_endpoint).strip()
        if model_endpoint.endswith("/"):
            model_endpoint = model_endpoint[:-1]
        if not model_endpoint.endswith("/generate"):
            model_endpoint = model_endpoint + "/generate"
        
        print(f"  Model endpoint: {model_endpoint}")
        
        # Sample questions from dataset
        samples = dataset.sample(n=SAMPLES_PER_EVALUATION)
        print(f"  Sampled {len(samples)} questions")
        
        # Evaluate each sample
        results = []
        correct_count = 0
        failed_count = 0
        timeout_count = 0
        
        for i, sample in enumerate(samples):
            prompt_text = sample.get_prompt_text()
            expected_answer = sample.get_expected_answer()
            
            print(f"  [{i+1}/{len(samples)}] Querying model...")
            
            # Query miner model
            model_output = query_miner_model(model_endpoint, prompt_text, model_api_key)
            
            if not model_output:
                failed_count += 1
                results.append({
                    "sample_id": sample.sample_id,
                    "correct": False,
                    "predicted": None,
                    "ground_truth": expected_answer,
                    "error": "Failed to query model"
                })
                continue
            
            # Extract and compare answer
            print(f"    📝 Model output: {model_output[:200]}...")
            print(f"    🎯 Ground truth: {expected_answer}")
            is_correct, predicted = extract_and_compare(model_output, expected_answer)
            print(f"    🔍 Extracted answer: {predicted}")
            
            if is_correct:
                correct_count += 1
                print(f"    ✅ Correct: {predicted}")
            else:
                print(f"    ❌ Incorrect: {predicted} (expected: {expected_answer})")
            
            results.append({
                "sample_id": sample.sample_id,
                "correct": is_correct,
                "predicted": predicted,
                "ground_truth": expected_answer
            })
        
        # Calculate score
        total_evaluated = len(results)
        score = correct_count / total_evaluated if total_evaluated > 0 else 0.0
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        print(f"\n📊 [EVALUATE] Results:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Evaluated: {total_evaluated}")
        print(f"  Correct: {correct_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Timeout: {timeout_count}")
        print(f"  Score: {score:.4f}")
        print(f"  Execution time: {execution_time_ms}ms")
        
        return EvaluationResponse(
            request_id=request.request_id,
            success=True,
            error=None,
            score=score,
            results={
                "total_samples": len(samples),
                "correct": correct_count,
                "failed": failed_count,
                "timeout": timeout_count,
                "details": results
            },
            execution_time_ms=execution_time_ms,
            cost=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        print(f"❌ [EVALUATE] Error: {e}")
        
        return EvaluationResponse(
            request_id=request.request_id,
            success=False,
            error=str(e),
            score=0.0,
            results={},
            execution_time_ms=execution_time_ms,
            cost=None
        )


@app.post("/receive_answers")
def receive_answers(submission: MinerSubmission):
    """
    Receive miner answer and score it against docmath dataset.
    This is the correct way to use the challenge container.
    """
    print(f"\n📥 [RECEIVE_ANSWERS] Task: {submission.task_id}")
    print(f"  Answer: {submission.answer[:200]}...")

    # Find the task in dataset (use task_id as index)
    task_index = int(submission.task_id) if submission.task_id.isdigit() else 0
    if task_index >= len(dataset.samples):
        task_index = 0

    sample = dataset.samples[task_index]
    expected_answer = sample.get_expected_answer()

    print(f"  🎯 Ground truth: {expected_answer}")

    # Extract and compare answer
    is_correct, predicted = extract_and_compare(submission.answer, expected_answer)
    print(f"  🔍 Extracted answer: {predicted}")
    print(f"  ✅ Correct: {is_correct}")

    return {
        "task_id": submission.task_id,
        "correct": is_correct,
        "predicted": predicted,
        "ground_truth": expected_answer,
        "score": 1.0 if is_correct else 0.0
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("QUASAR-SUBNET CHALLENGE SERVER")
    print("=" * 60)
    print(f"Host: {CHALLENGE_HOST}")
    print(f"Port: {CHALLENGE_PORT}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Samples per evaluation: {SAMPLES_PER_EVALUATION}")
    print(f"Timeout: {TIMEOUT_SECS}s")
    print("\nEndpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /config           - Challenge configuration")
    print("  POST /evaluate         - Evaluate miner submission (legacy)")
    print("  POST /receive_answers  - Receive miner answer and score it")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=CHALLENGE_HOST,
        port=CHALLENGE_PORT,
        log_level="info"
    )
