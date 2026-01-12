#!/usr/bin/env python3
"""
Code Runner HTTP Server - Runs miner code in a container.

This server runs as an HTTP server inside a Docker container.
It receives code, executes it with test inputs, and returns results.

Protocol (SDK 2.0 style):
- GET /health - Health check
- POST /execute - Execute code with test input
"""

import os
import sys
import json
import traceback
import ast
import time
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# Configuration
RUNNER_PORT = int(os.getenv("RUNNER_PORT", "8765"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30"))

app = FastAPI(title="QUASAR Code Runner")


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


class HealthResponse(BaseModel):
    status: str
    version: str


def validate_code(code: str) -> bool:
    """Validate code for dangerous operations."""
    try:
        tree = ast.parse(code)
        
        # Check for dangerous imports
        dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'pickle', 'marshal',
            'ctypes', 'multiprocessing', 'socket', 'http', 'urllib',
            'ftplib', 'telnetlib', 'smtplib', 'imaplib', 'poplib',
            'nntplib', 'ssl', 'hashlib', 'secrets', 'random'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in dangerous_modules:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in dangerous_modules:
                    return False
            elif isinstance(node, ast.Call):
                # Check for dangerous builtins
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'compile', 'open', '__import__']:
                        return False
        
        return True
    except:
        return False


def execute_code(code: str, function_name: str, test_input: Any) -> ExecuteResponse:
    """Execute code and return result."""
    start_time = time.time()
    
    try:
        # Validate code
        if not validate_code(code):
            return ExecuteResponse(
                success=False,
                output=None,
                error="Code contains dangerous operations",
                execution_time_ms=0.0
            )
        
        # Create namespace
        namespace: Dict[str, Any] = {}
        
        # Execute code to define function
        exec(code, namespace)
        
        # Check if function exists
        if function_name not in namespace:
            return ExecuteResponse(
                success=False,
                output=None,
                error=f"Function '{function_name}' not found",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Execute function
        func = namespace[function_name]
        result = func(test_input)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return ExecuteResponse(
            success=True,
            output=result,
            execution_time_ms=execution_time_ms
        )
        
    except SyntaxError as e:
        return ExecuteResponse(
            success=False,
            output=None,
            error=f"Syntax error: {e}",
            execution_time_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        return ExecuteResponse(
            success=False,
            output=None,
            error=f"Execution error: {e}",
            execution_time_ms=(time.time() - start_time) * 1000
        )


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0"
    )


@app.post("/execute", response_model=ExecuteResponse)
def execute(request: ExecuteRequest):
    """Execute code with test input."""
    print(f"[RUNNER] Executing: {request.function_name}", flush=True)
    print(f"[RUNNER] Input: {request.test_input}", flush=True)
    print(f"[RUNNER] Code length: {len(request.code)}", flush=True)
    
    result = execute_code(request.code, request.function_name, request.test_input)
    
    print(f"[RUNNER] Result: success={result.success}, output={result.output}", flush=True)
    return result


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("QUASAR CODE RUNNER HTTP SERVER")
    print("=" * 60)
    print(f"Port: {RUNNER_PORT}")
    print(f"Execution timeout: {EXECUTION_TIMEOUT}s")
    print(f"Version: 1.0.0")
    print("\nEndpoints:")
    print("  GET  /health   - Health check")
    print("  POST /execute  - Execute code")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=RUNNER_PORT,
        log_level="info"
    )
