"""Sandbox Code Executor for QUASAR-SUBNET

Executes miner-submitted code in a sandboxed environment
and runs test cases to verify correctness.
"""

import ast
import sys
import io
import contextlib
import traceback
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ExecutionResult:
    """Result of executing code"""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timeout: bool = False


class SandboxExecutor:
    """Safely execute Python code in a constrained environment"""
    
    def __init__(self, timeout_sec: int = 30):
        """
        Initialize sandbox executor
        
        Args:
            timeout_sec: Maximum execution time in seconds
        """
        self.timeout_sec = timeout_sec
        self.allowed_modules = {
            # Standard library modules (safe ones)
            'typing', 'collections', 'itertools', 'functools',
            're', 'json', 'math', 'random', 'datetime', 'time',
            'string', 'copy', 'dataclasses', 'enum',
            # For parsing
            'ast', 'inspect',
        }
    
    def execute_function(
        self,
        function_code: str,
        function_name: str,
        test_input: Any
    ) -> ExecutionResult:
        """
        Execute a function with test input
        
        Args:
            function_code: Complete code containing the function
            function_name: Name of the function to execute
            test_input: Input to pass to the function
        
        Returns:
            ExecutionResult with output or error
        """
        start_time = datetime.now()
        
        try:
            # Parse and validate the code
            tree = ast.parse(function_code)
            
            # Check for dangerous operations
            self._validate_code(tree)
            
            # Create execution namespace
            namespace: Dict[str, Any] = {}
            
            # Add allowed modules
            for module_name in self.allowed_modules:
                try:
                    namespace[module_name] = __import__(module_name)
                except ImportError:
                    pass
            
            # Execute the function definition
            with self._capture_output():
                exec(function_code, namespace)
            
            # Check if function exists
            if function_name not in namespace:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Function '{function_name}' not found in submitted code"
                )
            
            # Get the function
            func = namespace[function_name]
            
            # Execute with timeout
            result = self._execute_with_timeout(func, test_input)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if result.timeout:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Execution timeout after {self.timeout_sec}s",
                    execution_time_ms=execution_time,
                    timeout=True
                )
            
            if result.error:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=result.error,
                    execution_time_ms=execution_time
                )
            
            return ExecutionResult(
                success=True,
                output=result.output,
                execution_time_ms=execution_time
            )
            
        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Syntax error: {e}",
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Execution error: {e}",
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def _validate_code(self, tree: ast.AST):
        """Validate that code doesn't contain dangerous operations"""
        class DangerousNodeChecker(ast.NodeVisitor):
            def __init__(self):
                self.dangerous = []
            
            def visit_Import(self, node):
                for alias in node.names:
                    if alias.name not in self.allowed_modules:
                        self.dangerous.append(f"Import of '{alias.name}' is not allowed")
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module and node.module not in self.allowed_modules:
                    self.dangerous.append(f"Import from '{node.module}' is not allowed")
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'compile', 'open', '__import__']:
                        self.dangerous.append(f"Dangerous function call: {node.func.id}")
                self.generic_visit(node)
        
        checker = DangerousNodeChecker()
        checker.visit(tree)
        
        if checker.dangerous:
            raise ValueError(f"Code validation failed: {'; '.join(checker.dangerous)}")
    
    def _execute_with_timeout(self, func, *args) -> ExecutionResult:
        """Execute function with timeout using signal or manual check"""
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def target():
            try:
                output = func(*args)
                result_queue.put(('success', output))
            except Exception as e:
                result_queue.put(('error', str(e)))
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        thread.join(timeout=self.timeout_sec)
        
        if thread.is_alive():
            return ExecutionResult(
                success=False,
                output=None,
                timeout=True
            )
        
        try:
            status, output = result_queue.get_nowait()
            if status == 'success':
                return ExecutionResult(success=True, output=output)
            else:
                return ExecutionResult(success=False, output=None, error=output)
        except queue.Empty:
            return ExecutionResult(
                success=False,
                output=None,
                error="No result from execution"
            )
    
    @contextlib.contextmanager
    def _capture_output(self):
        """Capture stdout/stderr during execution"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class LongcodeEvaluator:
    """Evaluate miner code against longcode benchmark test cases"""
    
    def __init__(self, timeout_sec: int = 30):
        self.executor = SandboxExecutor(timeout_sec=timeout_sec)
    
    def evaluate_submission(
        self,
        miner_code: str,
        function_name: str,
        test_cases: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate miner code against multiple test cases
        
        Args:
            miner_code: Complete code submitted by miner
            function_name: Name of function to execute
            test_cases: List of (input, expected_output) tuples
        
        Returns:
            Dict with evaluation results
        """
        results = []
        passed = 0
        failed = 0
        timeouts = 0
        
        for i, (test_input, expected_output) in enumerate(test_cases):
            result = self.executor.execute_function(
                miner_code,
                function_name,
                test_input
            )
            
            # Check if output matches expected
            is_correct = False
            if result.success:
                is_correct = self._compare_outputs(result.output, expected_output)
            
            results.append({
                "test_case": i,
                "input": str(test_input)[:100] + "..." if len(str(test_input)) > 100 else str(test_input),
                "expected": str(expected_output)[:100] + "..." if len(str(expected_output)) > 100 else str(expected_output),
                "actual": str(result.output)[:100] + "..." if result.success and len(str(result.output)) > 100 else str(result.output) if result.success else None,
                "correct": is_correct,
                "success": result.success,
                "error": result.error,
                "timeout": result.timeout,
                "execution_time_ms": result.execution_time_ms
            })
            
            if result.timeout:
                timeouts += 1
            elif is_correct:
                passed += 1
            else:
                failed += 1
        
        total = len(test_cases)
        score = passed / total if total > 0 else 0.0
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "timeouts": timeouts,
            "score": score,
            "results": results
        }
    
    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare actual output with expected output"""
        # Handle None
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False
        
        # Handle lists
        if isinstance(expected, list) and isinstance(actual, list):
            if len(actual) != len(expected):
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        # Handle dicts
        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(actual.keys()) != set(expected.keys()):
                return False
            return all(self._compare_outputs(actual[k], expected[k]) for k in expected.keys())
        
        # Handle strings (case-insensitive for some cases)
        if isinstance(expected, str) and isinstance(actual, str):
            # Normalize whitespace
            expected_norm = ' '.join(expected.split())
            actual_norm = ' '.join(actual.split())
            return expected_norm == actual_norm
        
        # Handle numbers with tolerance
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(actual - expected) < 1e-6
        
        # Default: direct comparison
        return actual == expected
