#!/usr/bin/env python3
# The MIT License (MIT)
# Copyright © 2024 HFA Research Team

"""
Comprehensive test runner for the unified HFA-SimpleMind subnet.

This script runs all tests including:
- Integration tests
- Performance tests  
- Load tests
- System validation

Usage:
    python run_tests.py --all                    # Run all tests
    python run_tests.py --integration           # Run integration tests only
    python run_tests.py --performance           # Run performance tests only
    python run_tests.py --validation            # Run system validation only
    python run_tests.py --quick                 # Run quick test suite
"""

import asyncio
import argparse
import sys
import os
import time
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bittensor as bt


async def run_integration_tests() -> Dict[str, Any]:
    """Run integration tests"""
    print("🧪 Running integration tests...")
    
    try:
        from tests.integration.test_runner import IntegrationTestRunner
        
        config = {
            "parallel_execution": False,
            "verbose": True,
            "continue_on_failure": True,
            "default_timeout": 300
        }
        
        runner = IntegrationTestRunner(config)
        basic_suite = runner.create_basic_test_suite()
        runner.add_test_suite(basic_suite)
        
        results = await runner.run_all_tests()
        
        return {
            "status": "pass" if runner.stats["failed"] == 0 and runner.stats["errors"] == 0 else "fail",
            "results": results,
            "summary": runner.stats
        }
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "results": {},
            "summary": {}
        }


async def run_performance_tests() -> Dict[str, Any]:
    """Run performance tests"""
    print("🏃 Running performance tests...")
    
    try:
        from tests.performance.benchmark import PerformanceBenchmarkSuite
        
        # Create mock components for testing
        mock_model = type('MockModel', (), {
            'forward': lambda self, x: f"Mock response for {x}"
        })()
        
        mock_validator = type('MockValidator', (), {
            'generate_evaluation_tasks': lambda self: [{"type": "test", "context": "test"}],
            'score_responses': lambda self, responses, task: {0: 0.8, 1: 0.9}
        })()
        
        benchmark_suite = PerformanceBenchmarkSuite()
        
        # Add benchmarks
        model_benchmark = benchmark_suite.create_model_inference_benchmark(
            mock_model, ["test input"]
        )
        benchmark_suite.add_benchmark(model_benchmark)
        
        task_benchmark = benchmark_suite.create_task_generation_benchmark(mock_validator)
        benchmark_suite.add_benchmark(task_benchmark)
        
        # Run benchmarks
        results = await benchmark_suite.run_all_benchmarks()
        
        return {
            "status": "pass" if results["summary"]["failed"] == 0 else "fail",
            "results": results,
            "summary": results["summary"]
        }
        
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "results": {},
            "summary": {}
        }


async def run_system_validation() -> Dict[str, Any]:
    """Run system validation"""
    print("🔍 Running system validation...")
    
    try:
        from tests.validation.system_validator import SystemValidator
        
        validator = SystemValidator()
        results = await validator.run_full_validation()
        
        return {
            "status": results["overall_status"],
            "results": results,
            "summary": results["summary"]
        }
        
    except Exception as e:
        print(f"❌ System validation failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "results": {},
            "summary": {}
        }


async def run_quick_tests() -> Dict[str, Any]:
    """Run quick test suite"""
    print("⚡ Running quick test suite...")
    
    # Quick validation of core components
    results = {
        "status": "pass",
        "tests_run": [],
        "summary": {"passed": 0, "failed": 0, "errors": 0}
    }
    
    # Test 1: Model factory
    try:
        from template.model_factory import ModelArchitectureFactory
        factory = ModelArchitectureFactory()
        architectures = factory.get_available_architectures()
        
        if len(architectures) >= 3:  # Should have hfa, simplemind, hybrid
            results["tests_run"].append({"name": "model_factory", "status": "pass"})
            results["summary"]["passed"] += 1
        else:
            results["tests_run"].append({"name": "model_factory", "status": "fail", "message": "Missing architectures"})
            results["summary"]["failed"] += 1
            results["status"] = "fail"
            
    except Exception as e:
        results["tests_run"].append({"name": "model_factory", "status": "error", "error": str(e)})
        results["summary"]["errors"] += 1
        results["status"] = "fail"
    
    # Test 2: Benchmark loader
    try:
        from benchmarks.benchmark_loader import BenchmarkLoader
        loader = BenchmarkLoader({})
        
        results["tests_run"].append({"name": "benchmark_loader", "status": "pass"})
        results["summary"]["passed"] += 1
        
    except Exception as e:
        results["tests_run"].append({"name": "benchmark_loader", "status": "error", "error": str(e)})
        results["summary"]["errors"] += 1
        results["status"] = "fail"
    
    # Test 3: Diversity tracker
    try:
        from template.validator.diversity_tracker import DiversityTracker
        tracker = DiversityTracker()
        
        results["tests_run"].append({"name": "diversity_tracker", "status": "pass"})
        results["summary"]["passed"] += 1
        
    except Exception as e:
        results["tests_run"].append({"name": "diversity_tracker", "status": "error", "error": str(e)})
        results["summary"]["errors"] += 1
        results["status"] = "fail"
    
    # Test 4: Monitoring systems
    try:
        from template.monitoring import TelemetryCollector, HealthMonitor
        
        telemetry = TelemetryCollector()
        health_monitor = HealthMonitor()
        
        results["tests_run"].append({"name": "monitoring_systems", "status": "pass"})
        results["summary"]["passed"] += 1
        
    except Exception as e:
        results["tests_run"].append({"name": "monitoring_systems", "status": "error", "error": str(e)})
        results["summary"]["errors"] += 1
        results["status"] = "fail"
    
    return results


def print_final_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print final test summary"""
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    overall_status = "pass"
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    for test_type, results in all_results.items():
        status = results.get("status", "unknown")
        summary = results.get("summary", {})
        
        print(f"\n{test_type.upper()}:")
        print(f"  Status: {status.upper()}")
        
        if "passed" in summary:
            print(f"  Passed: {summary.get('passed', 0)}")
            print(f"  Failed: {summary.get('failed', 0)}")
            print(f"  Errors: {summary.get('errors', 0)}")
            
            total_tests += summary.get('total_tests', summary.get('passed', 0) + summary.get('failed', 0) + summary.get('errors', 0))
            total_passed += summary.get('passed', 0)
            total_failed += summary.get('failed', 0)
            total_errors += summary.get('errors', 0)
        
        if status in ["fail", "error"]:
            overall_status = "fail"
    
    print(f"\nOVERALL:")
    print(f"  Status: {overall_status.upper()}")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {total_passed} ✅")
    print(f"  Failed: {total_failed} ❌")
    print(f"  Errors: {total_errors} 💥")
    
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"  Success Rate: {success_rate:.1f}%")
    
    print("="*80)
    
    return overall_status


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Unified HFA-SimpleMind subnet test runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--validation", action="store_true", help="Run system validation")
    parser.add_argument("--quick", action="store_true", help="Run quick test suite")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # If no specific test type specified, run quick tests
    if not any([args.all, args.integration, args.performance, args.validation, args.quick]):
        args.quick = True
    
    print("🚀 Starting unified HFA-SimpleMind subnet tests")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    all_results = {}
    
    try:
        if args.quick:
            all_results["quick"] = await run_quick_tests()
        
        if args.integration or args.all:
            all_results["integration"] = await run_integration_tests()
        
        if args.performance or args.all:
            all_results["performance"] = await run_performance_tests()
        
        if args.validation or args.all:
            all_results["validation"] = await run_system_validation()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print(f"\n⏱️ Total test duration: {total_duration:.2f}s")
        
        # Print final summary
        overall_status = print_final_summary(all_results)
        
        # Save results
        import json
        with open("test_results.json", "w") as f:
            json.dump({
                "timestamp": start_time,
                "duration": total_duration,
                "overall_status": overall_status,
                "results": all_results
            }, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to test_results.json")
        
        # Exit with appropriate code
        if overall_status == "fail":
            print("\n❌ Some tests failed - see details above")
            return 1
        else:
            print("\n✅ All tests passed!")
            return 0
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)