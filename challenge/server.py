"""Challenge Server for QUASAR-SUBNET

Simplified server for health checks and configuration.
The main evaluation logic is now in validator_api.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
CHALLENGE_HOST = os.getenv("CHALLENGE_HOST", "0.0.0.0")
CHALLENGE_PORT = int(os.getenv("CHALLENGE_PORT", "8080"))

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=CHALLENGE_PORT)
args = parser.parse_args()
CHALLENGE_PORT = args.port

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
class HealthResponse(BaseModel):
    status: str
    version: str


class ConfigResponse(BaseModel):
    challenge_id: str
    version: str
    evaluation_config: dict


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        version="2.0.0"
    )


@app.get("/config")
def get_config():
    """Get challenge configuration"""
    return ConfigResponse(
        challenge_id="quasar-longcode",
        version="2.0.0",
        evaluation_config={
            "benchmark": "longcode",
            "description": "Code submission benchmark with sandboxed execution",
            "evaluation_method": "code_execution",
            "test_cases": "multiple",
            "security": "sandboxed"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("QUASAR-SUBNET CHALLENGE SERVER (Longcode)")
    print("=" * 60)
    print(f"Host: {CHALLENGE_HOST}")
    print(f"Port: {CHALLENGE_PORT}")
    print(f"Version: 2.0.0")
    print("\nEndpoints:")
    print("  GET  /health  - Health check")
    print("  GET  /config  - Challenge configuration")
    print("\nNote: Main evaluation is handled by validator_api")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=CHALLENGE_HOST,
        port=CHALLENGE_PORT,
        log_level="info"
    )
