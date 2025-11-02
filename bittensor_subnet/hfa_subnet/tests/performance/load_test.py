# The MIT License (MIT)
# Copyright © 2024 HFA Research Team

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import time
import statistics
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

import bittensor as bt


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    active_threads: int
    response_time: float
    throughput: float  # requests per second
    error_rate: float
    success_count: int
    error_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "active_threads": self.active_threads,
            "response_time": self.response_time,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "success_count": self.success_count,
            "error_count": self.error_count
        }


@dataclass
class LoadTestConfig:
    """Load test configuration"""
    name: str
    description: str
    duration_seconds: int = 300  # 5 minutes
    concurrent_users: int = 10
    ramp_up_seconds: int = 60
    target_rps: Optional[float] = None  # requests per second
    max_response_time: float = 30.0  # seconds
    success_threshold: float = 0.95  # 95% success rate
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "duration_seconds": self.duration_seconds,
            "concurrent_users": self.concurrent_users,
            "ramp_up_seconds": self.ramp_up_seconds,
            "target_rps": self.target_rps,
            "max_response_time": self.max_response_time,
            "success_threshold": self.success_threshold
        }


class LoadTestRunner:
    """
    Comprehensive load testing system for the unified HFA-SimpleMind subnet.
    
    Features:
    - Concurrent user simulation
    - Performance metrics collection
    - Resource utilization monitoring
    - Stress testing with gradual load increase
    - Detailed reporting and analysis
    """
    
    def __init__(self, config: LoadTestConfig):
        """
        Initialize load test runner.
        
        Args:
            config: Load test configuration
        """
        self.config = config
        
        # Test state
        self.is_running = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Metrics collection
        self.metrics: List[LoadTestMetrics] = []
        self.response_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        
        # Worker management
        self.workers: List[threading.Thread] = []
        self.worker_stop_event = threading.Event()
        
        # Monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        
        bt.logging.info(f"🏋️ LoadTestRunner initialized: {config.name}")
    
    async def run_load_test(self, test_function: Callable) -> Dict[str, Any]:
        """
        Run load test with specified test function.
        
        Args:
            test_function: Async function to test under load
            
        Returns:
            Load test results and analysis
        """
        bt.logging.info(f"🏋️ Starting load test: {self.config.name}")
        
        self.is_running = True
        self.start_time = time.time()
        self.worker_stop_event.clear()
        
        try:
            # Start monitoring
            self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitor_thread.start()
            
            # Start workers with ramp-up
            await self._start_workers_with_rampup(test_function)
            
            # Wait for test duration
            await asyncio.sleep(self.config.duration_seconds)
            
            # Stop workers
            self._stop_workers()
            
            self.end_time = time.time()
            self.is_running = False
            
            # Generate results
            results = self._generate_results()
            
            bt.logging.info(f"🏋️ Load test completed: {self.config.name}")
            
            return results
            
        except Exception as e:
            bt.logging.error(f"Load test failed: {e}")
            self._stop_workers()
            self.is_running = False
            raise
    
    async def _start_workers_with_rampup(self, test_function: Callable):
        """Start workers with gradual ramp-up"""
        ramp_up_interval = self.config.ramp_up_seconds / self.config.concurrent_users
        
        for i in range(self.config.concurrent_users):
            if not self.is_running:
                break
            
            # Create and start worker
            worker = threading.Thread(
                target=self._worker_loop,
                args=(test_function, f"worker_{i}"),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
            bt.logging.debug(f"🏋️ Started worker {i+1}/{self.config.concurrent_users}")
            
            # Wait for ramp-up interval
            if i < self.config.concurrent_users - 1:
                await asyncio.sleep(ramp_up_interval)
        
        bt.logging.info(f"🏋️ All {len(self.workers)} workers started")
    
    def _worker_loop(self, test_function: Callable, worker_id: str):
        """Worker loop that executes test function repeatedly"""
        while not self.worker_stop_event.is_set() and self.is_running:
            start_time = time.time()
            
            try:
                # Execute test function
                if asyncio.iscoroutinefunction(test_function):
                    # Run async function in new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(test_function())
                    finally:
                        loop.close()
                else:
                    test_function()
                
                # Record success
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                self.success_count += 1
                
                bt.logging.debug(f"🏋️ {worker_id} success: {response_time:.3f}s")
                
            except Exception as e:
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                self.error_count += 1
                
                bt.logging.debug(f"🏋️ {worker_id} error: {str(e)}")
            
            # Rate limiting if target RPS is set
            if self.config.target_rps:
                expected_interval = 1.0 / (self.config.target_rps / self.config.concurrent_users)
                actual_duration = time.time() - start_time
                sleep_time = max(0, expected_interval - actual_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    def _stop_workers(self):
        """Stop all worker threads"""
        self.worker_stop_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        bt.logging.info(f"🏋️ Stopped {len(self.workers)} workers")
    
    def _monitor_resources(self):
        """Monitor system resources during load test"""
        while self.is_running:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Calculate current throughput and error rate
                current_time = time.time()
                if self.start_time:
                    elapsed = current_time - self.start_time
                    total_requests = self.success_count + self.error_count
                    throughput = total_requests / elapsed if elapsed > 0 else 0
                    error_rate = self.error_count / total_requests if total_requests > 0 else 0
                else:
                    throughput = 0
                    error_rate = 0
                
                # Calculate average response time for recent requests
                recent_response_times = self.response_times[-100:]  # Last 100 requests
                avg_response_time = statistics.mean(recent_response_times) if recent_response_times else 0
                
                # Create metrics entry
                metrics = LoadTestMetrics(
                    timestamp=current_time,
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    active_threads=threading.active_count(),
                    response_time=avg_response_time,
                    throughput=throughput,
                    error_rate=error_rate,
                    success_count=self.success_count,
                    error_count=self.error_count
                )
                
                self.metrics.append(metrics)
                
                bt.logging.debug(f"🏋️ Metrics: CPU={cpu_percent:.1f}%, "
                               f"Memory={memory.percent:.1f}%, "
                               f"RPS={throughput:.1f}, "
                               f"Errors={error_rate:.2%}")
                
            except Exception as e:
                bt.logging.error(f"Error monitoring resources: {e}")
            
            time.sleep(5)  # Monitor every 5 seconds
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive load test results"""
        if not self.start_time or not self.end_time:
            return {"error": "Test timing not available"}
        
        total_duration = self.end_time - self.start_time
        total_requests = self.success_count + self.error_count
        
        # Calculate statistics
        response_time_stats = {}
        if self.response_times:
            response_time_stats = {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": self._percentile(self.response_times, 95),
                "p99": self._percentile(self.response_times, 99)
            }
        
        # Resource utilization stats
        resource_stats = {}
        if self.metrics:
            cpu_values = [m.cpu_percent for m in self.metrics]
            memory_values = [m.memory_percent for m in self.metrics]
            
            resource_stats = {
                "cpu": {
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "mean": statistics.mean(cpu_values)
                },
                "memory": {
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "mean": statistics.mean(memory_values)
                }
            }
        
        # Performance analysis
        success_rate = self.success_count / total_requests if total_requests > 0 else 0
        throughput = total_requests / total_duration if total_duration > 0 else 0
        
        # Determine test result
        test_passed = (
            success_rate >= self.config.success_threshold and
            (not response_time_stats or response_time_stats["p95"] <= self.config.max_response_time)
        )
        
        results = {
            "config": self.config.to_dict(),
            "execution": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration_seconds": total_duration,
                "test_passed": test_passed
            },
            "requests": {
                "total": total_requests,
                "successful": self.success_count,
                "failed": self.error_count,
                "success_rate": success_rate,
                "throughput_rps": throughput
            },
            "response_times": response_time_stats,
            "resource_utilization": resource_stats,
            "metrics_timeline": [m.to_dict() for m in self.metrics],
            "analysis": self._analyze_results(success_rate, throughput, response_time_stats, resource_stats)
        }
        
        return results
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _analyze_results(self, success_rate: float, throughput: float, 
                        response_time_stats: Dict[str, float], 
                        resource_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze load test results and provide insights"""
        analysis = {
            "performance_grade": "unknown",
            "bottlenecks": [],
            "recommendations": [],
            "alerts": []
        }
        
        # Performance grading
        if success_rate >= 0.99 and response_time_stats.get("p95", float('inf')) <= 5.0:
            analysis["performance_grade"] = "excellent"
        elif success_rate >= 0.95 and response_time_stats.get("p95", float('inf')) <= 10.0:
            analysis["performance_grade"] = "good"
        elif success_rate >= 0.90 and response_time_stats.get("p95", float('inf')) <= 20.0:
            analysis["performance_grade"] = "fair"
        else:
            analysis["performance_grade"] = "poor"
        
        # Identify bottlenecks
        if resource_stats.get("cpu", {}).get("mean", 0) > 80:
            analysis["bottlenecks"].append("High CPU utilization")
            analysis["recommendations"].append("Consider CPU optimization or scaling")
        
        if resource_stats.get("memory", {}).get("mean", 0) > 85:
            analysis["bottlenecks"].append("High memory utilization")
            analysis["recommendations"].append("Consider memory optimization or scaling")
        
        if response_time_stats.get("p95", 0) > self.config.max_response_time:
            analysis["bottlenecks"].append("High response times")
            analysis["recommendations"].append("Investigate response time bottlenecks")
        
        if success_rate < self.config.success_threshold:
            analysis["alerts"].append(f"Success rate ({success_rate:.2%}) below threshold ({self.config.success_threshold:.2%})")
        
        # Throughput analysis
        if self.config.target_rps and throughput < self.config.target_rps * 0.8:
            analysis["alerts"].append(f"Throughput ({throughput:.1f} RPS) significantly below target ({self.config.target_rps} RPS)")
        
        return analysis


class SubnetLoadTester:
    """Specialized load tester for subnet components"""
    
    def __init__(self):
        self.test_configs = self._create_test_configs()
    
    def _create_test_configs(self) -> List[LoadTestConfig]:
        """Create predefined load test configurations"""
        return [
            LoadTestConfig(
                name="light_load",
                description="Light load test with 5 concurrent users",
                duration_seconds=180,
                concurrent_users=5,
                ramp_up_seconds=30,
                max_response_time=10.0,
                success_threshold=0.98
            ),
            LoadTestConfig(
                name="moderate_load",
                description="Moderate load test with 15 concurrent users",
                duration_seconds=300,
                concurrent_users=15,
                ramp_up_seconds=60,
                max_response_time=15.0,
                success_threshold=0.95
            ),
            LoadTestConfig(
                name="heavy_load",
                description="Heavy load test with 30 concurrent users",
                duration_seconds=600,
                concurrent_users=30,
                ramp_up_seconds=120,
                max_response_time=30.0,
                success_threshold=0.90
            ),
            LoadTestConfig(
                name="stress_test",
                description="Stress test with 50 concurrent users",
                duration_seconds=300,
                concurrent_users=50,
                ramp_up_seconds=60,
                max_response_time=60.0,
                success_threshold=0.80
            )
        ]
    
    async def run_miner_load_test(self, miner, config_name: str = "moderate_load") -> Dict[str, Any]:
        """Run load test on miner component"""
        config = next((c for c in self.test_configs if c.name == config_name), self.test_configs[1])
        runner = LoadTestRunner(config)
        
        async def test_miner_inference():
            """Test function for miner inference"""
            # Mock inference request
            test_synapse = type('TestSynapse', (), {
                'context': 'Test context for load testing',
                'prompt': 'Test prompt',
                'max_tokens': 100
            })()
            
            # Call miner forward method
            response = await miner.forward(test_synapse)
            
            # Validate response
            assert response is not None, "Miner should return response"
            return response
        
        return await runner.run_load_test(test_miner_inference)
    
    async def run_validator_load_test(self, validator, config_name: str = "moderate_load") -> Dict[str, Any]:
        """Run load test on validator component"""
        config = next((c for c in self.test_configs if c.name == config_name), self.test_configs[1])
        runner = LoadTestRunner(config)
        
        async def test_validator_evaluation():
            """Test function for validator evaluation"""
            # Generate evaluation tasks
            tasks = validator.generate_evaluation_tasks()
            assert len(tasks) > 0, "Validator should generate tasks"
            
            # Mock miner responses
            mock_responses = {
                0: type('MockResponse', (), {
                    'response': 'Test response',
                    'exact_match_score': 0.8,
                    'f1_score': 0.75
                })(),
                1: type('MockResponse', (), {
                    'response': 'Another test response',
                    'exact_match_score': 0.85,
                    'f1_score': 0.80
                })()
            }
            
            # Score responses
            scores = validator.score_responses(mock_responses, tasks[0])
            assert len(scores) > 0, "Validator should return scores"
            
            return scores
        
        return await runner.run_load_test(test_validator_evaluation)
    
    async def run_full_subnet_load_test(self, miners: List, validators: List, 
                                      config_name: str = "moderate_load") -> Dict[str, Any]:
        """Run comprehensive load test on full subnet"""
        config = next((c for c in self.test_configs if c.name == config_name), self.test_configs[1])
        runner = LoadTestRunner(config)
        
        async def test_full_evaluation_cycle():
            """Test function for full evaluation cycle"""
            if not validators:
                raise ValueError("No validators available for testing")
            
            validator = validators[0]
            
            # Generate tasks
            tasks = validator.generate_evaluation_tasks()
            assert len(tasks) > 0, "Should generate evaluation tasks"
            
            # Mock miner queries and responses
            miner_uids = list(range(len(miners)))
            mock_responses = {}
            
            for uid in miner_uids:
                mock_responses[uid] = type('MockResponse', (), {
                    'response': f'Response from miner {uid}',
                    'exact_match_score': 0.8 + (uid * 0.05),
                    'f1_score': 0.75 + (uid * 0.05),
                    'tokens_per_second': 100.0,
                    'memory_usage_mb': 1024.0
                })()
            
            # Score responses
            scores = validator.score_responses(mock_responses, tasks[0])
            assert len(scores) == len(miner_uids), "Should score all miners"
            
            return scores
        
        return await runner.run_load_test(test_full_evaluation_cycle)
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save load test results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            bt.logging.info(f"🏋️ Load test results saved to {filename}")
            
        except Exception as e:
            bt.logging.error(f"Failed to save load test results: {e}")
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print load test results summary"""
        config = results.get("config", {})
        execution = results.get("execution", {})
        requests = results.get("requests", {})
        response_times = results.get("response_times", {})
        analysis = results.get("analysis", {})
        
        print("\n" + "="*80)
        print(f"LOAD TEST RESULTS: {config.get('name', 'Unknown')}")
        print("="*80)
        print(f"Duration: {execution.get('duration_seconds', 0):.1f}s")
        print(f"Concurrent Users: {config.get('concurrent_users', 0)}")
        print(f"Total Requests: {requests.get('total', 0)}")
        print(f"Successful: {requests.get('successful', 0)}")
        print(f"Failed: {requests.get('failed', 0)}")
        print(f"Success Rate: {requests.get('success_rate', 0):.2%}")
        print(f"Throughput: {requests.get('throughput_rps', 0):.1f} RPS")
        
        if response_times:
            print(f"Response Times:")
            print(f"  Mean: {response_times.get('mean', 0):.3f}s")
            print(f"  P95: {response_times.get('p95', 0):.3f}s")
            print(f"  P99: {response_times.get('p99', 0):.3f}s")
        
        print(f"Performance Grade: {analysis.get('performance_grade', 'unknown').upper()}")
        print(f"Test Passed: {'✅' if execution.get('test_passed', False) else '❌'}")
        
        if analysis.get("bottlenecks"):
            print(f"Bottlenecks: {', '.join(analysis['bottlenecks'])}")
        
        if analysis.get("alerts"):
            print("Alerts:")
            for alert in analysis["alerts"]:
                print(f"  ⚠️ {alert}")
        
        print("="*80)


async def main():
    """Main entry point for load testing"""
    
    # Create load tester
    load_tester = SubnetLoadTester()
    
    print("🏋️ Starting subnet load testing...")
    
    # This would normally test real components
    # For now, we'll demonstrate the framework
    
    print("Load testing framework ready!")
    print("Available test configurations:")
    for config in load_tester.test_configs:
        print(f"  - {config.name}: {config.description}")


if __name__ == "__main__":
    asyncio.run(main())