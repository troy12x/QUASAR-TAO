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
import json
import sys
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import traceback

import bittensor as bt


@dataclass
class TestResult:
    """Result of a single test"""
    test_name: str
    status: str  # "pass", "fail", "skip", "error"
    duration: float
    message: str = ""
    error_details: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "status": self.status,
            "duration": duration,
            "message": self.message,
            "error_details": self.error_details,
            "timestamp": self.timestamp
        }


@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    description: str
    tests: List[Callable]
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    timeout: int = 300  # 5 minutes default timeout


class IntegrationTestRunner:
    """
    Comprehensive test runner for integration tests.
    
    Features:
    - Parallel and sequential test execution
    - Test result reporting and analysis
    - Performance benchmarking
    - Error handling and recovery
    - Test environment management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize test runner.
        
        Args:
            config: Configuration dictionary with test settings
        """
        self.config = config or {}
        
        # Test execution settings
        self.parallel_execution = self.config.get("parallel_execution", False)
        self.max_workers = self.config.get("max_workers", 4)
        self.default_timeout = self.config.get("default_timeout", 300)
        self.continue_on_failure = self.config.get("continue_on_failure", True)
        
        # Reporting settings
        self.report_format = self.config.get("report_format", "json")
        self.report_file = self.config.get("report_file", "integration_test_results.json")
        self.verbose = self.config.get("verbose", True)
        
        # Test suites
        self.test_suites: List[TestSuite] = []
        self.results: List[TestResult] = []
        
        # Statistics
        self.stats = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "total_duration": 0.0
        }
        
        bt.logging.info("🧪 IntegrationTestRunner initialized")
    
    def add_test_suite(self, suite: TestSuite):
        """Add a test suite to the runner"""
        self.test_suites.append(suite)
        bt.logging.info(f"🧪 Added test suite: {suite.name} ({len(suite.tests)} tests)")
    
    def create_basic_test_suite(self) -> TestSuite:
        """Create basic integration test suite"""
        from .test_end_to_end import (
            TestEndToEndIntegration,
            test_full_system_integration
        )
        
        test_instance = TestEndToEndIntegration()
        
        # Create test environment for all tests
        from .test_end_to_end import IntegrationTestEnvironment
        test_env = IntegrationTestEnvironment()
        test_env.setup_mocks()
        
        tests = [
            lambda: asyncio.run(test_instance.test_complete_evaluation_cycle(test_env)),
            lambda: asyncio.run(test_instance.test_multi_architecture_support(test_env)),
            lambda: asyncio.run(test_instance.test_benchmark_integration(test_env)),
            lambda: asyncio.run(test_instance.test_diversity_tracking_system(test_env)),
            lambda: asyncio.run(test_instance.test_scoring_harness_integration(test_env)),
            lambda: asyncio.run(test_instance.test_monitoring_system_integration(test_env)),
            lambda: asyncio.run(test_instance.test_protocol_communication(test_env)),
            lambda: asyncio.run(test_instance.test_error_handling_and_recovery(test_env)),
            lambda: asyncio.run(test_instance.test_performance_under_load(test_env)),
            lambda: asyncio.run(test_full_system_integration())
        ]
        
        return TestSuite(
            name="basic_integration",
            description="Basic integration tests for unified HFA-SimpleMind subnet",
            tests=tests,
            setup=lambda: test_env,
            teardown=lambda env: env.cleanup() if env else None
        )
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        bt.logging.info("🧪 Starting integration test execution")
        
        start_time = time.time()
        
        for suite in self.test_suites:
            await self.run_test_suite(suite)
        
        end_time = time.time()
        self.stats["total_duration"] = end_time - start_time
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        self.save_report(report)
        
        # Print summary
        self.print_summary()
        
        bt.logging.info(f"🧪 Integration test execution completed in {self.stats['total_duration']:.2f}s")
        
        return report
    
    async def run_test_suite(self, suite: TestSuite):
        """Run a single test suite"""
        bt.logging.info(f"🧪 Running test suite: {suite.name}")
        
        # Setup
        setup_result = None
        if suite.setup:
            try:
                setup_result = suite.setup()
                bt.logging.info(f"🧪 Setup completed for {suite.name}")
            except Exception as e:
                bt.logging.error(f"🧪 Setup failed for {suite.name}: {e}")
                return
        
        try:
            # Run tests
            if self.parallel_execution:
                await self.run_tests_parallel(suite.tests, suite.name)
            else:
                await self.run_tests_sequential(suite.tests, suite.name)
        
        finally:
            # Teardown
            if suite.teardown:
                try:
                    suite.teardown(setup_result)
                    bt.logging.info(f"🧪 Teardown completed for {suite.name}")
                except Exception as e:
                    bt.logging.error(f"🧪 Teardown failed for {suite.name}: {e}")
    
    async def run_tests_sequential(self, tests: List[Callable], suite_name: str):
        """Run tests sequentially"""
        for i, test in enumerate(tests):
            test_name = f"{suite_name}.test_{i+1}"
            await self.run_single_test(test, test_name)
    
    async def run_tests_parallel(self, tests: List[Callable], suite_name: str):
        """Run tests in parallel"""
        tasks = []
        
        for i, test in enumerate(tests):
            test_name = f"{suite_name}.test_{i+1}"
            task = asyncio.create_task(self.run_single_test(test, test_name))
            tasks.append(task)
        
        # Wait for all tests to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def run_single_test(self, test: Callable, test_name: str) -> TestResult:
        """Run a single test"""
        start_time = time.time()
        
        if self.verbose:
            bt.logging.info(f"🧪 Running test: {test_name}")
        
        try:
            # Run test with timeout
            await asyncio.wait_for(
                asyncio.create_task(self._execute_test(test)),
                timeout=self.default_timeout
            )
            
            duration = time.time() - start_time
            result = TestResult(
                test_name=test_name,
                status="pass",
                duration=duration,
                message="Test passed successfully"
            )
            
            self.stats["passed"] += 1
            
            if self.verbose:
                bt.logging.info(f"✅ Test passed: {test_name} ({duration:.2f}s)")
        
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            result = TestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                message="Test timed out",
                error_details=f"Test exceeded timeout of {self.default_timeout}s"
            )
            
            self.stats["errors"] += 1
            
            if self.verbose:
                bt.logging.error(f"⏰ Test timed out: {test_name} ({duration:.2f}s)")
        
        except AssertionError as e:
            duration = time.time() - start_time
            result = TestResult(
                test_name=test_name,
                status="fail",
                duration=duration,
                message="Test assertion failed",
                error_details=str(e)
            )
            
            self.stats["failed"] += 1
            
            if self.verbose:
                bt.logging.error(f"❌ Test failed: {test_name} - {str(e)}")
        
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                message="Test error",
                error_details=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )
            
            self.stats["errors"] += 1
            
            if self.verbose:
                bt.logging.error(f"💥 Test error: {test_name} - {str(e)}")
        
        self.results.append(result)
        self.stats["total_tests"] += 1
        
        return result
    
    async def _execute_test(self, test: Callable):
        """Execute a test function"""
        if asyncio.iscoroutinefunction(test):
            await test()
        else:
            # Run synchronous test in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, test)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            "test_run_info": {
                "timestamp": time.time(),
                "date": datetime.now().isoformat(),
                "total_duration": self.stats["total_duration"],
                "config": self.config
            },
            "summary": self.stats.copy(),
            "test_suites": [
                {
                    "name": suite.name,
                    "description": suite.description,
                    "test_count": len(suite.tests)
                }
                for suite in self.test_suites
            ],
            "results": [result.to_dict() for result in self.results],
            "performance_metrics": self._calculate_performance_metrics(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from test results"""
        if not self.results:
            return {}
        
        durations = [result.duration for result in self.results]
        
        return {
            "average_test_duration": sum(durations) / len(durations),
            "min_test_duration": min(durations),
            "max_test_duration": max(durations),
            "total_test_time": sum(durations),
            "success_rate": (self.stats["passed"] / self.stats["total_tests"]) * 100 if self.stats["total_tests"] > 0 else 0,
            "failure_rate": (self.stats["failed"] / self.stats["total_tests"]) * 100 if self.stats["total_tests"] > 0 else 0,
            "error_rate": (self.stats["errors"] / self.stats["total_tests"]) * 100 if self.stats["total_tests"] > 0 else 0
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Performance recommendations
        performance_metrics = self._calculate_performance_metrics()
        
        if performance_metrics.get("success_rate", 0) < 80:
            recommendations.append("Success rate is below 80%. Review failed tests and fix underlying issues.")
        
        if performance_metrics.get("average_test_duration", 0) > 30:
            recommendations.append("Average test duration is high. Consider optimizing test performance.")
        
        if performance_metrics.get("error_rate", 0) > 10:
            recommendations.append("Error rate is high. Review test environment and error handling.")
        
        # Specific test recommendations
        failed_tests = [result for result in self.results if result.status == "fail"]
        if failed_tests:
            recommendations.append(f"Review and fix {len(failed_tests)} failed tests.")
        
        error_tests = [result for result in self.results if result.status == "error"]
        if error_tests:
            recommendations.append(f"Investigate and resolve {len(error_tests)} test errors.")
        
        if not recommendations:
            recommendations.append("All tests are performing well. Consider adding more comprehensive tests.")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any]):
        """Save test report to file"""
        try:
            with open(self.report_file, 'w') as f:
                if self.report_format == "json":
                    json.dump(report, f, indent=2, default=str)
                else:
                    # Add other formats as needed
                    json.dump(report, f, indent=2, default=str)
            
            bt.logging.info(f"🧪 Test report saved to {self.report_file}")
            
        except Exception as e:
            bt.logging.error(f"Failed to save test report: {e}")
    
    def print_summary(self):
        """Print test execution summary"""
        print("\n" + "="*80)
        print("INTEGRATION TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.stats['total_tests']}")
        print(f"Passed: {self.stats['passed']} ✅")
        print(f"Failed: {self.stats['failed']} ❌")
        print(f"Errors: {self.stats['errors']} 💥")
        print(f"Skipped: {self.stats['skipped']} ⏭️")
        print(f"Total Duration: {self.stats['total_duration']:.2f}s")
        
        if self.stats['total_tests'] > 0:
            success_rate = (self.stats['passed'] / self.stats['total_tests']) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        print("="*80)
        
        # Print failed tests
        failed_tests = [result for result in self.results if result.status in ["fail", "error"]]
        if failed_tests:
            print("\nFAILED TESTS:")
            for result in failed_tests:
                print(f"  ❌ {result.test_name}: {result.message}")
                if result.error_details and self.verbose:
                    print(f"     {result.error_details}")
        
        print()


async def main():
    """Main test runner entry point"""
    
    # Configuration
    config = {
        "parallel_execution": False,  # Sequential for better debugging
        "verbose": True,
        "continue_on_failure": True,
        "default_timeout": 300,
        "report_file": "integration_test_results.json"
    }
    
    # Create test runner
    runner = IntegrationTestRunner(config)
    
    # Add test suites
    basic_suite = runner.create_basic_test_suite()
    runner.add_test_suite(basic_suite)
    
    # Run all tests
    try:
        report = await runner.run_all_tests()
        
        # Exit with appropriate code
        if runner.stats["failed"] > 0 or runner.stats["errors"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        bt.logging.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())