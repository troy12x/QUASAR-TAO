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
import inspect
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# Configuration
RUNNER_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv("RUNNER_PORT", "8765"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30"))

app = FastAPI(title="QUASAR Code Runner")


def _extract_code(raw: str) -> str:
    if raw is None:
        return ""
    text = raw.strip()
    if "```" not in text:
        return text
    parts = text.split("```")
    if len(parts) < 3:
        return text
    candidate = parts[1]
    candidate = candidate.lstrip("\r\n")
    first_line, sep, rest = candidate.partition("\n")
    if first_line.strip().lower() in {"python", "py"}:
        return rest
    return candidate


def _list_defined_functions(namespace: Dict[str, Any]) -> Dict[str, Any]:
    funcs: Dict[str, Any] = {}
    for k, v in namespace.items():
        if k.startswith("__"):
            continue
        if callable(v) and inspect.isfunction(v):
            funcs[k] = v
    return funcs


def _list_defined_classes(namespace: Dict[str, Any]) -> Dict[str, Any]:
    classes: Dict[str, Any] = {}
    for k, v in namespace.items():
        if k.startswith("__"):
            continue
        if inspect.isclass(v):
            classes[k] = v
    return classes


def _pick_function(function_name: str, namespace: Dict[str, Any]) -> Optional[str]:
    funcs = _list_defined_functions(namespace)
    if function_name and function_name in funcs:
        return function_name
    for preferred in ("solve", "main", "solution"):
        if preferred in funcs:
            return preferred
    if len(funcs) == 1:
        return next(iter(funcs.keys()))
    if len(funcs) > 1:
        return sorted(funcs.keys())[0]
    return None


def _pick_class_method(function_name: str, namespace: Dict[str, Any]) -> Optional[tuple[str, str]]:
    classes = _list_defined_classes(namespace)
    preferred_methods = [m for m in (function_name, "solve", "main", "solution") if m]

    if "Solution" in classes:
        cls = classes["Solution"]
        for m in preferred_methods:
            if hasattr(cls, m) and callable(getattr(cls, m)):
                return ("Solution", m)

    for cls_name, cls in classes.items():
        for m in preferred_methods:
            if hasattr(cls, m) and callable(getattr(cls, m)):
                return (cls_name, m)
    return None


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
        code = _extract_code(code)
        # If code is empty, read from miner_code.py
        if not code or code.strip() == "":
            try:
                with open("/app/miner_code.py", "r") as f:
                    code = _extract_code(f.read())
                print(f"[RUNNER] Read code from /app/miner_code.py ({len(code)} chars)", flush=True)
                print(f"[RUNNER] Code preview: {code[:200]}...", flush=True)
            except FileNotFoundError:
                return ExecuteResponse(
                    success=False,
                    output=None,
                    error="No code provided and miner_code.py not found",
                    execution_time_ms=0.0
                )
        else:
            print(f"[RUNNER] Received code in request ({len(code)} chars)", flush=True)
            print(f"[RUNNER] Code preview: {code[:200]}...", flush=True)
        
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

        funcs = _list_defined_functions(namespace)
        classes = _list_defined_classes(namespace)
        chosen_name = _pick_function(function_name, namespace)
        chosen_method = None if chosen_name else _pick_class_method(function_name, namespace)

        print(f"[RUNNER] Namespace keys: {list(namespace.keys())}", flush=True)
        print(f"[RUNNER] Functions: {list(funcs.keys())}", flush=True)
        print(f"[RUNNER] Classes: {list(classes.keys())}", flush=True)
        print(f"[RUNNER] Requested function: {function_name}", flush=True)
        print(f"[RUNNER] Chosen function: {chosen_name}", flush=True)
        print(f"[RUNNER] Chosen class method: {chosen_method}", flush=True)

        if not chosen_name and not chosen_method:
            return ExecuteResponse(
                success=False,
                output=None,
                error=(
                    f"No callable entrypoint found. Requested '{function_name}'. "
                    f"Functions: {list(funcs.keys())}. Classes: {list(classes.keys())}"
                ),
                execution_time_ms=(time.time() - start_time) * 1000
            )

        # Execute function
        if chosen_name:
            func = namespace[chosen_name]
            try:
                result = func(test_input)
            except TypeError:
                result = func()
        else:
            cls_name, method_name = chosen_method
            cls = namespace[cls_name]
            obj = cls()
            method = getattr(obj, method_name)
            try:
                result = method(test_input)
            except TypeError:
                result = method()
        
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
