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

import time
import asyncio
import statistics
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import json
import tracemalloc

import bittensor as bt


@dataclass
class BenchmarkResult:
    """Result of a single benchmark"""
    name: str
    description: str
    duration: float
    iterations: int
    operations_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "duration": self.duration,
            "iterations": self.iterations,
            "operations_per_second": self.operations_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class PerformanceBenchmark:
    """Performance benchmark configuration"""
    name: str
    description: str
    benchmark_function: Callable
    iterations: int = 100
    warmup_iterations: int = 10
    timeout_seconds: int = 300
    measure_memory: bool = True
    measure_cpu: bool = True
    
    async def run(self) -> BenchmarkResult:
        """Run the benchmark"""
        bt.logging.info(f"🏃 Running benchmark: {self.name}")
        
        # Warmup
        if self.warmup_iterations > 0:
            bt.logging.debug(f"🏃 Warming up with {self.warmup_iterations} iterations")
            for _ in range(self.warmup_iterations):
                try:
                    if asyncio.iscoroutinefunction(self.benchmark_function):
                        await self.benchmark_function()
                    else:
                        self.benchmark_function()
                except Exception:
                    pass  # Ignore warmup errors
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Start memory tracking if enabled
        if self.measure_memory:
            tracemalloc.start()
        
        # Measure initial CPU usage
        initial_cpu = psutil.cpu_percent(interval=None) if self.measure_cpu else 0
        
        # Run benchmark
        start_time = time.time()
        success_count = 0
        error_message = None
        
        try:
            for i in range(self.iterations):
                try:
                    if asyncio.iscoroutinefunction(self.benchmark_function):
                        await asyncio.wait_for(
                            self.benchmark_function(),
                            timeout=self.timeout_seconds / self.iterations
                        )
                    else:
                        self.benchmark_function()
                    
                    success_count += 1
                    
                except asyncio.TimeoutError:
                    error_message = f"Timeout on iteration {i+1}"
                    break
                except Exception as e:
                    error_message = f"Error on iteration {i+1}: {str(e)}"
                    break
        
        except Exception as e:
            error_message = f"Benchmark setup error: {str(e)}"
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Measure final CPU usage
        final_cpu = psutil.cpu_percent(interval=1) if self.measure_cpu else 0
        avg_cpu = (initial_cpu + final_cpu) / 2
        
        # Measure memory usage
        memory_usage_mb = 0
        if self.measure_memory:
            try:
                current, peak = tracemalloc.get_traced_memory()
                memory_usage_mb = peak / (1024 * 1024)
                tracemalloc.stop()
            except Exception:
                memory_usage_mb = 0
        
        # Calculate operations per second
        ops_per_second = success_count / duration if duration > 0 else 0
        
        result = BenchmarkResult(
            name=self.name,
            description=self.description,
            duration=duration,
            iterations=success_count,
            operations_per_second=ops_per_second,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=avg_cpu,
            success=success_count == self.iterations,
            error_message=error_message
        )
        
        bt.logging.info(f"🏃 Benchmark completed: {self.name} - {ops_per_second:.1f} ops/sec")
        
        return result


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmarking suite for subnet components.
    
    Features:
    - Model inference benchmarking
    - Task generation performance
    - Scoring system benchmarking
    - Memory and CPU profiling
    - Comparative analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize benchmark suite.
        
        Args:
            config: Configuration dictionary with benchmark settings
        """
        self.config = config or {}
        self.benchmarks: List[PerformanceBenchmark] = []
        self.results: List[BenchmarkResult] = []
        
        # Default benchmark settings
        self.default_iterations = self.config.get("default_iterations", 100)
        self.default_warmup = self.config.get("default_warmup", 10)
        self.default_timeout = self.config.get("default_timeout", 300)
        
        bt.logging.info("🏃 PerformanceBenchmarkSuite initialized")
    
    def add_benchmark(self, benchmark: PerformanceBenchmark):
        """Add a benchmark to the suite"""
        self.benchmarks.append(benchmark)
        bt.logging.info(f"🏃 Added benchmark: {benchmark.name}")
    
    def create_model_inference_benchmark(self, model, test_inputs: List[str]) -> PerformanceBenchmark:
        """Create model inference benchmark"""
        
        async def inference_benchmark():
            """Benchmark model inference"""
            test_input = test_inputs[0] if test_inputs else "Test input for benchmarking"
            
            # Mock inference call
            if hasattr(model, 'forward'):
                if asyncio.iscoroutinefunction(model.forward):
                    result = await model.forward(test_input)
                else:
                    result = model.forward(test_input)
            else:
                # Fallback for mock models
                await asyncio.sleep(0.001)  # Simulate processing time
                result = "Mock inference result"
            
            return result
        
        return PerformanceBenchmark(
            name="model_inference",
            description="Model inference performance benchmark",
            benchmark_function=inference_benchmark,
            iterations=self.default_iterations,
            warmup_iterations=self.default_warmup
        )
    
    def create_task_generation_benchmark(self, validator) -> PerformanceBenchmark:
        """Create task generation benchmark"""
        
        def task_generation_benchmark():
            """Benchmark task generation"""
            tasks = validator.generate_evaluation_tasks()
            assert len(tasks) > 0, "Should generate tasks"
            return tasks
        
        return PerformanceBenchmark(
            name="task_generation",
            description="Evaluation task generation performance",
            benchmark_function=task_generation_benchmark,
            iterations=50,  # Fewer iterations for complex operations
            warmup_iterations=5
        )
    
    def create_scoring_benchmark(self, validator, mock_responses: Dict[int, Any], 
                               test_task: Dict[str, Any]) -> PerformanceBenchmark:
        """Create scoring system benchmark"""
        
        def scoring_benchmark():
            """Benchmark response scoring"""
            scores = validator.score_responses(mock_responses, test_task)
            assert len(scores) > 0, "Should return scores"
            return scores
        
        return PerformanceBenchmark(
            name="response_scoring",
            description="Response scoring performance",
            benchmark_function=scoring_benchmark,
            iterations=self.default_iterations,
            warmup_iterations=self.default_warmup
        )
    
    def create_diversity_tracking_benchmark(self, diversity_tracker, 
                                          test_responses: List[Tuple[int, str, Dict]]) -> PerformanceBenchmark:
        """Create diversity tracking benchmark"""
        
        def diversity_benchmark():
            """Benchmark diversity tracking"""
            miner_uid, response, model_info = test_responses[0]
            metrics = diversity_tracker.track_miner_response(miner_uid, response, model_info)
            assert metrics is not None, "Should return diversity metrics"
            return metrics
        
        return PerformanceBenchmark(
            name="diversity_tracking",
            description="Diversity tracking performance",
            benchmark_function=diversity_benchmark,
            iterations=self.default_iterations,
            warmup_iterations=self.default_warmup
        )
    
    def create_benchmark_loading_benchmark(self, benchmark_loader) -> PerformanceBenchmark:
        """Create benchmark loading performance test"""
        
        def benchmark_loading():
            """Benchmark loading performance"""
            tasks = benchmark_loader.load_benchmark_tasks(num_tasks=5)
            return tasks
        
        return PerformanceBenchmark(
            name="benchmark_loading",
            description="Benchmark data loading performance",
            benchmark_function=benchmark_loading,
            iterations=20,  # Fewer iterations for I/O operations
            warmup_iterations=2
        )
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks in the suite"""
        bt.logging.info(f"🏃 Running {len(self.benchmarks)} benchmarks")
        
        start_time = time.time()
        
        for benchmark in self.benchmarks:
            try:
                result = await benchmark.run()
                self.results.append(result)
            except Exception as e:
                bt.logging.error(f"Benchmark {benchmark.name} failed: {e}")
                
                # Create failed result
                failed_result = BenchmarkResult(
                    name=benchmark.name,
                    description=benchmark.description,
                    duration=0.0,
                    iterations=0,
                    operations_per_second=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    success=False,
                    error_message=str(e)
                )
                self.results.append(failed_result)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate comprehensive report
        report = self._generate_report(total_duration)
        
        bt.logging.info(f"🏃 Benchmark suite completed in {total_duration:.2f}s")
        
        return report
    
    def _generate_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        # Calculate summary statistics
        summary = {
            "total_benchmarks": len(self.results),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "total_duration": total_duration,
            "success_rate": len(successful_results) / len(self.results) if self.results else 0
        }
        
        # Performance statistics
        performance_stats = {}
        if successful_results:
            ops_per_sec = [r.operations_per_second for r in successful_results]
            memory_usage = [r.memory_usage_mb for r in successful_results]
            cpu_usage = [r.cpu_usage_percent for r in successful_results]
            
            performance_stats = {
                "operations_per_second": {
                    "min": min(ops_per_sec),
                    "max": max(ops_per_sec),
                    "mean": statistics.mean(ops_per_sec),
                    "median": statistics.median(ops_per_sec)
                },
                "memory_usage_mb": {
                    "min": min(memory_usage),
                    "max": max(memory_usage),
                    "mean": statistics.mean(memory_usage),
                    "median": statistics.median(memory_usage)
                },
                "cpu_usage_percent": {
                    "min": min(cpu_usage),
                    "max": max(cpu_usage),
                    "mean": statistics.mean(cpu_usage),
                    "median": statistics.median(cpu_usage)
                }
            }
        
        # Individual results
        individual_results = [result.to_dict() for result in self.results]
        
        # Performance analysis
        analysis = self._analyze_performance(successful_results)
        
        report = {
            "timestamp": time.time(),
            "config": self.config,
            "summary": summary,
            "performance_statistics": performance_stats,
            "individual_results": individual_results,
            "analysis": analysis,
            "recommendations": self._generate_recommendations(successful_results, failed_results)
        }
        
        return report
    
    def _analyze_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance results"""
        if not results:
            return {"status": "no_successful_results"}
        
        analysis = {
            "overall_performance": "unknown",
            "bottlenecks": [],
            "strengths": []
        }
        
        # Analyze operations per second
        ops_per_sec = [r.operations_per_second for r in results]
        avg_ops = statistics.mean(ops_per_sec)
        
        if avg_ops > 1000:
            analysis["strengths"].append("High throughput performance")
        elif avg_ops < 10:
            analysis["bottlenecks"].append("Low throughput performance")
        
        # Analyze memory usage
        memory_usage = [r.memory_usage_mb for r in results]
        avg_memory = statistics.mean(memory_usage)
        
        if avg_memory > 1000:  # > 1GB
            analysis["bottlenecks"].append("High memory usage")
        elif avg_memory < 100:  # < 100MB
            analysis["strengths"].append("Efficient memory usage")
        
        # Analyze CPU usage
        cpu_usage = [r.cpu_usage_percent for r in results]
        avg_cpu = statistics.mean(cpu_usage)
        
        if avg_cpu > 80:
            analysis["bottlenecks"].append("High CPU usage")
        elif avg_cpu < 20:
            analysis["strengths"].append("Efficient CPU usage")
        
        # Overall performance grade
        if len(analysis["bottlenecks"]) == 0:
            analysis["overall_performance"] = "excellent"
        elif len(analysis["bottlenecks"]) <= 1:
            analysis["overall_performance"] = "good"
        elif len(analysis["bottlenecks"]) <= 2:
            analysis["overall_performance"] = "fair"
        else:
            analysis["overall_performance"] = "poor"
        
        return analysis
    
    def _generate_recommendations(self, successful_results: List[BenchmarkResult], 
                                failed_results: List[BenchmarkResult]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if failed_results:
            recommendations.append(f"Investigate and fix {len(failed_results)} failed benchmarks")
        
        if successful_results:
            # Memory recommendations
            memory_usage = [r.memory_usage_mb for r in successful_results]
            avg_memory = statistics.mean(memory_usage)
            
            if avg_memory > 2000:  # > 2GB
                recommendations.append("Consider memory optimization - average usage is high")
            
            # CPU recommendations
            cpu_usage = [r.cpu_usage_percent for r in successful_results]
            avg_cpu = statistics.mean(cpu_usage)
            
            if avg_cpu > 70:
                recommendations.append("Consider CPU optimization - average usage is high")
            
            # Throughput recommendations
            ops_per_sec = [r.operations_per_second for r in successful_results]
            min_ops = min(ops_per_sec)
            
            if min_ops < 1:
                recommendations.append("Some operations are very slow - investigate bottlenecks")
        
        if not recommendations:
            recommendations.append("Performance looks good - consider adding more comprehensive benchmarks")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str):
        """Save benchmark report to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            bt.logging.info(f"🏃 Benchmark report saved to {filename}")
            
        except Exception as e:
            bt.logging.error(f"Failed to save benchmark report: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print benchmark summary"""
        summary = report.get("summary", {})
        performance_stats = report.get("performance_statistics", {})
        analysis = report.get("analysis", {})
        
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        print(f"Total Benchmarks: {summary.get('total_benchmarks', 0)}")
        print(f"Successful: {summary.get('successful', 0)} ✅")
        print(f"Failed: {summary.get('failed', 0)} ❌")
        print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"Total Duration: {summary.get('total_duration', 0):.2f}s")
        
        if performance_stats:
            ops_stats = performance_stats.get("operations_per_second", {})
            memory_stats = performance_stats.get("memory_usage_mb", {})
            
            print(f"\nPerformance Statistics:")
            print(f"  Operations/sec: {ops_stats.get('mean', 0):.1f} avg, {ops_stats.get('max', 0):.1f} max")
            print(f"  Memory Usage: {memory_stats.get('mean', 0):.1f} MB avg, {memory_stats.get('max', 0):.1f} MB max")
        
        print(f"\nOverall Performance: {analysis.get('overall_performance', 'unknown').upper()}")
        
        if analysis.get("strengths"):
            print("Strengths:")
            for strength in analysis["strengths"]:
                print(f"  ✅ {strength}")
        
        if analysis.get("bottlenecks"):
            print("Bottlenecks:")
            for bottleneck in analysis["bottlenecks"]:
                print(f"  ⚠️ {bottleneck}")
        
        print("="*80)


async def main():
    """Main entry point for performance benchmarking"""
    
    # Create benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite({
        "default_iterations": 50,
        "default_warmup": 5
    })
    
    print("🏃 Performance benchmarking framework ready!")
    print("This framework can benchmark:")
    print("  - Model inference performance")
    print("  - Task generation speed")
    print("  - Scoring system performance")
    print("  - Diversity tracking efficiency")
    print("  - Benchmark loading performance")


if __name__ == "__main__":
    asyncio.run(main())