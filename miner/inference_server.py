#!/usr/bin/env python3
# The MIT License (MIT)
# Copyright 2026 SILX INC
#
# Miner Inference Server for QUASAR Inference Verification Subnet
#
# This server exposes the required inference interface for validators:
#   inference(prompt, gen_len, logits_at_step) → {tokens, captured_logits, elapsed_sec}

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              QUASAR MINER - INFERENCE SERVER                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  This server must be exposed by miner Docker containers.                     ║
║                                                                              ║
║  REQUIRED INTERFACE:                                                         ║
║  POST /inference                                                             ║
║  Request:                                                                    ║
║    {                                                                         ║
║      "prompt": [token_ids...],      // List of input token IDs               ║
║      "gen_len": int,                 // Number of tokens to generate         ║
║      "logits_at_step": int           // Step to capture logits (1-indexed)   ║
║    }                                                                         ║
║                                                                              ║
║  Response:                                                                   ║
║    {                                                                         ║
║      "tokens": [generated_ids...],   // Generated token IDs                  ║
║      "captured_logits": [floats...], // Logits at logits_at_step             ║
║      "elapsed_sec": float            // Time taken for inference             ║
║    }                                                                         ║
║                                                                              ║
║  The captured_logits are compared against the reference model (Qwen2.5-0.5B) ║
║  using cosine similarity + max absolute difference for verification.         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ CONFIGURATION                                                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Model configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Server configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ MODEL LOADING                                                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Global model and tokenizer
model = None
tokenizer = None


def load_model():
    """Load the model and tokenizer with memory optimizations."""
    global model, tokenizer
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    
    # Load tokenizer first (lightweight)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Memory-efficient loading options for CPU
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,  # Reduces peak memory during loading
    }
    
    if DEVICE == "cpu":
        # For CPU, use float32 and no device_map
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = None
    else:
        # For CUDA, use float16 and auto device_map
        load_kwargs["torch_dtype"] = DTYPE
        load_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **load_kwargs
    )
    
    if DEVICE == "cpu" or not hasattr(model, 'hf_device_map'):
        model = model.to(DEVICE)
    
    model.eval()
    
    # Enable memory-efficient attention if available
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    
    print(f"Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for model loading."""
    load_model()
    yield


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ API MODELS                                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    prompt: List[int]  # Input token IDs
    gen_len: int  # Number of tokens to generate
    logits_at_step: int  # Step to capture logits (1-indexed)


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""
    tokens: List[int]  # Generated token IDs
    captured_logits: Optional[List[float]]  # Logits at the specified step
    elapsed_sec: float  # Time taken for inference


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    model_name: str
    device: str


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ FASTAPI APP                                                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

app = FastAPI(
    title="QUASAR Miner Inference Server",
    description="Inference server for QUASAR subnet miners",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model_name=MODEL_NAME,
        device=DEVICE
    )


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    Run inference and capture logits at a specific step.
    
    This is the core endpoint that validators call to evaluate miners.
    The logits at logits_at_step are compared against the reference model.
    """
    global model, tokenizer
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        device = next(model.parameters()).device
        input_ids = torch.tensor([request.prompt], device=device)
        
        generated_tokens = []
        captured_logits = None
        past_key_values = None
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # Initial forward pass (prefill)
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Autoregressive generation
            for step in range(request.gen_len):
                # Capture logits at the specified step (1-indexed)
                if step + 1 == request.logits_at_step:
                    captured_logits = next_token_logits[0].cpu().float().tolist()
                
                # Greedy decoding (argmax)
                next_token = next_token_logits.argmax(dim=-1)
                generated_tokens.append(next_token.item())
                
                # Forward pass with KV cache
                outputs = model(
                    next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
        
        elapsed_sec = time.perf_counter() - start_time
        
        return InferenceResponse(
            tokens=generated_tokens,
            captured_logits=captured_logits,
            elapsed_sec=elapsed_sec
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def model_info():
    """Get model information."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": MODEL_NAME,
        "vocab_size": len(tokenizer),
        "device": DEVICE,
        "dtype": str(DTYPE),
        "model_config": {
            "hidden_size": getattr(model.config, "hidden_size", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_attention_heads": getattr(model.config, "num_attention_heads", None),
        }
    }


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ MAIN                                                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("Starting QUASAR Miner Inference Server")
    print(f"Model: {MODEL_NAME}")
    print(f"Host: {HOST}:{PORT}")
    
    uvicorn.run(app, host=HOST, port=PORT)
