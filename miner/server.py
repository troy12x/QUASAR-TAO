"""Miner HTTP API for QUASAR-SUBNET

Exposes a model endpoint for validators to query.
Miners run this server to serve their model predictions.
"""

import os
import sys
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "100"))

# Create FastAPI app
app = FastAPI(title="QUASAR-SUBNET Miner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    prompt: List[Message]
    max_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.0


class GenerateResponse(BaseModel):
    output: str
    tokens_used: int


# Load model
print(f"🔄 Loading model from {MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    print(f"✅ Model loaded successfully")
    print(f"   Device: {model.device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Using mock model for testing")
    model = None
    tokenizer = None


def generate_response(messages: List[Message], max_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> str:
    """Generate response from model"""
    if model is None or tokenizer is None:
        # Mock response for testing
        return "Therefore, the answer is MOCK."
    
    # Format messages for Qwen
    text = ""
    for msg in messages:
        if msg.role == "user":
            text += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            text += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    
    text += "<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return response


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device) if model else "cpu"
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate response from model"""
    try:
        output = generate_response(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Estimate tokens used (rough approximation)
        tokens_used = len(output.split())
        
        return GenerateResponse(
            output=output,
            tokens_used=tokens_used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("QUASAR-SUBNET MINER API")
    print("=" * 60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Model: {MODEL_PATH}")
    print(f"Max tokens: {MAX_NEW_TOKENS}")
    print("\nEndpoints:")
    print("  GET  /health   - Health check")
    print("  POST /generate - Generate response from model")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
