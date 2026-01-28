"""Challenge Server for QUASAR-SUBNET

Coordinates two-container execution for secure code evaluation:
- Code Runner Container: Runs miner code as HTTP server
- Test Container: Executes tests against the code

Architecture follows term-challenge SDK 2.0 pattern.
"""

import os
import subprocess
import json
import tempfile
import uuid
import asyncio
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# Configuration
CHALLENGE_HOST = os.getenv("CHALLENGE_HOST", "0.0.0.0")
CHALLENGE_PORT = int(os.getenv("CHALLENGE_PORT", "8080"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30"))
CODE_RUNNER_PORT = 8765

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


class DockerManager:
    """Manages Docker containers for code execution."""
    
    def __init__(self):
        self.runner_container_id = None
        self.runner_container_ip = None
    
    def start_runner_container(self, code: str) -> str:
        """Start a code runner container with the miner's code."""
        print("[DOCKER] Starting code runner container...", flush=True)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            code_file = f.name
        
        container_name = f"quasar-runner-{uuid.uuid4().hex[:8]}"
        
        try:
            # Create container
            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "-p", f"{CODE_RUNNER_PORT}:{CODE_RUNNER_PORT}",
                "-v", f"{code_file}:/app/code.py:ro",
                "-w", "/app",
                "python:3.11-slim",
                "tail", "-f", "/dev/null"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to create container: {result.stderr}")
            
            self.runner_container_id = result.stdout.strip()
            
            # Get container IP
            inspect_cmd = ["docker", "inspect", self.runner_container_id]
            inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True)
            inspect_data = json.loads(inspect_result.stdout)
            self.runner_container_ip = inspect_data[0]["NetworkSettings"]["IPAddress"]
            
            # Start code runner HTTP server
            exec_cmd = [
                "docker", "exec", "-d", self.runner_container_id,
                "python", "-c",
                "import sys; sys.path.insert(0, '/app'); exec(open('/app/code.py').read(), globals()); "
                "from code_runner import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8765)"
            ]
            
            subprocess.run(exec_cmd, capture_output=True)
            
            # Wait for health check
            health_url = f"http://localhost:{CODE_RUNNER_PORT}/health"
            for i in range(50):
                try:
                    response = requests.get(health_url, timeout=1)
                    if response.status_code == 200:
                        print(f"[DOCKER] Runner container ready: {self.runner_container_id[:12]}", flush=True)
                        return self.runner_container_id
                except:
                    time.sleep(0.1)
            
            raise Exception("Runner container failed to start")
            
        finally:
            os.unlink(code_file)
    
    def execute_in_runner(self, function_name: str, test_input: Any) -> ExecuteResponse:
        """Execute code in the runner container via HTTP."""
        if not self.runner_container_id:
            raise Exception("No runner container started")
        
        start_time = time.time()
        
        try:
            # Send execute request
            execute_url = f"http://localhost:{CODE_RUNNER_PORT}/execute"
            response = requests.post(
                execute_url,
                json={
                    "code": "",
                    "function_name": function_name,
                    "test_input": test_input
                },
                timeout=EXECUTION_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            execution_time_ms = (time.time() - start_time) * 1000
            
            return ExecuteResponse(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                execution_time_ms=execution_time_ms,
                timeout=result.get("timeout", False)
            )
            
        except requests.exceptions.Timeout:
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
                error=f"HTTP request error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def cleanup(self):
        """Stop and remove the runner container."""
        if self.runner_container_id:
            print(f"[DOCKER] Cleaning up container: {self.runner_container_id[:12]}", flush=True)
            subprocess.run(["docker", "stop", self.runner_container_id], capture_output=True)
            subprocess.run(["docker", "rm", self.runner_container_id], capture_output=True)
            self.runner_container_id = None
            self.runner_container_ip = None


# Global Docker manager
docker_manager = DockerManager()


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
            "description": "Code submission benchmark with two-container Docker execution",
            "evaluation_method": "docker_two_container",
            "test_cases": "multiple",
            "security": "docker_sandboxed"
        }
    )


@app.post("/execute", response_model=ExecuteResponse)
def execute_code(request: ExecuteRequest):
    """Execute code using two-container architecture."""
    print(f"[CHALLENGE] Executing function: {request.function_name}", flush=True)
    print(f"[CHALLENGE] Test input: {request.test_input}", flush=True)
    print(f"[CHALLENGE] Code length: {len(request.code)} chars", flush=True)
    
    try:
        # Start runner container with code
        docker_manager.start_runner_container(request.code)
        
        # Execute via HTTP
        result = docker_manager.execute_in_runner(request.function_name, request.test_input)
        
        print(f"[CHALLENGE] Result: success={result.success}, output={result.output}", flush=True)
        return result
        
    except Exception as e:
        print(f"[CHALLENGE] Error: {e}", flush=True)
        return ExecuteResponse(
            success=False,
            output=None,
            error=str(e),
            execution_time_ms=0.0
        )
    finally:
        # Cleanup
        docker_manager.cleanup()


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("QUASAR-SUBNET CHALLENGE SERVER (Two-Container Docker)")
    print("=" * 60)
    print(f"Host: {CHALLENGE_HOST}")
    print(f"Port: {CHALLENGE_PORT}")
    print(f"Version: 2.0.0")
    print(f"Execution timeout: {EXECUTION_TIMEOUT}s")
    print(f"Code runner port: {CODE_RUNNER_PORT}")
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
