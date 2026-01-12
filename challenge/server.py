"""Challenge Server for QUASAR-SUBNET

Executes miner-submitted code in a Docker container for secure evaluation.
"""

import os
import subprocess
import json
import tempfile
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
CHALLENGE_HOST = os.getenv("CHALLENGE_HOST", "0.0.0.0")
CHALLENGE_PORT = int(os.getenv("CHALLENGE_PORT", "8080"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30"))

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


class ExecuteRequest(BaseModel):
    code: str
    function_name: str
    test_input: Any


class ExecuteResponse(BaseModel):
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timeout: bool = False


def execute_in_docker(code: str, function_name: str, test_input: Any) -> ExecuteResponse:
    """Execute code in a Docker container."""
    import time
    start_time = time.time()

    # Create temporary file for code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        code_file = f.name

    # Create temporary file for test input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"function_name": function_name, "test_input": test_input}, f)
        input_file = f.name

    try:
        # Run Docker container to execute the code
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{code_file}:/app/code.py",
                "-v", f"{input_file}:/app/input.json",
                "-w", "/app",
                "python:3.11-slim",
                "python", "-c",
                f"""
import json
import sys
from datetime import datetime

# Load input
with open('/app/input.json') as f:
    data = json.load(f)
    function_name = data['function_name']
    test_input = data['test_input']

# Execute code
try:
    exec(open('/app/code.py').read(), globals())
    if function_name not in globals():
        print(json.dumps({{"success": False, "error": f"Function {{function_name}} not found"}}))
        sys.exit(1)
    
    result = globals()[function_name](test_input)
    print(json.dumps({{"success": True, "output": result}}))
except Exception as e:
    import traceback
    print(json.dumps({{"success": False, "error": str(e), "traceback": traceback.format_exc()}}))
    sys.exit(1)
"""
            ],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT
        )

        execution_time = (time.time() - start_time) * 1000

        if result.returncode != 0:
            # Try to parse error from stdout
            try:
                error_data = json.loads(result.stdout.strip())
                return ExecuteResponse(
                    success=False,
                    output=None,
                    error=error_data.get("error", result.stderr or "Unknown error"),
                    execution_time_ms=execution_time
                )
            except:
                return ExecuteResponse(
                    success=False,
                    output=None,
                    error=result.stderr or "Execution failed",
                    execution_time_ms=execution_time
                )

        # Parse successful output
        try:
            output_data = json.loads(result.stdout.strip())
            return ExecuteResponse(
                success=output_data.get("success", False),
                output=output_data.get("output"),
                error=output_data.get("error"),
                execution_time_ms=execution_time
            )
        except:
            return ExecuteResponse(
                success=False,
                output=None,
                error=f"Failed to parse output: {result.stdout}",
                execution_time_ms=execution_time
            )

    except subprocess.TimeoutExpired:
        return ExecuteResponse(
            success=False,
            output=None,
            error=f"Execution timeout after {EXECUTION_TIMEOUT}s",
            execution_time_ms=EXECUTION_TIMEOUT * 1000,
            timeout=True
        )
    except Exception as e:
        return ExecuteResponse(
            success=False,
            output=None,
            error=f"Docker execution error: {str(e)}",
            execution_time_ms=(time.time() - start_time) * 1000
        )
    finally:
        # Cleanup temp files
        try:
            os.unlink(code_file)
            os.unlink(input_file)
        except:
            pass


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
            "description": "Code submission benchmark with Docker-based execution",
            "evaluation_method": "docker_execution",
            "test_cases": "multiple",
            "security": "docker_sandboxed"
        }
    )


@app.post("/execute", response_model=ExecuteResponse)
def execute_code(request: ExecuteRequest):
    """Execute code in a Docker container"""
    print(f"[CHALLENGE] Executing function: {request.function_name}", flush=True)
    print(f"[CHALLENGE] Test input: {request.test_input}", flush=True)
    print(f"[CHALLENGE] Code length: {len(request.code)} chars", flush=True)

    result = execute_in_docker(request.code, request.function_name, request.test_input)

    print(f"[CHALLENGE] Result: success={result.success}, output={result.output}", flush=True)
    return result


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("QUASAR-SUBNET CHALLENGE SERVER (Longcode + Docker)")
    print("=" * 60)
    print(f"Host: {CHALLENGE_HOST}")
    print(f"Port: {CHALLENGE_PORT}")
    print(f"Version: 2.0.0")
    print(f"Execution timeout: {EXECUTION_TIMEOUT}s")
    print("\nEndpoints:")
    print("  GET  /health   - Health check")
    print("  GET  /config   - Challenge configuration")
    print("  POST /execute  - Execute code in Docker container")
    print("=" * 60)

    uvicorn.run(
        app,
        host=CHALLENGE_HOST,
        port=CHALLENGE_PORT,
        log_level="info"
    )
