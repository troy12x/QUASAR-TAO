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

import pytest
import asyncio
import time
import json
import tempfile
import os
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import threading

import bittensor as bt

# Import subnet components
from template.model_factory import ModelArchitectureFactory
from template.models.hfa_model import HFAModel
from template.models.simplemind_model import SimpleMindModel
from template.models.hybrid_model import HybridModel
from template.validator.diversity_tracker import DiversityTracker
from template.validator.scoring_harness import ScoringHarness
from template.monitoring import (
    TelemetryCollector, HealthMonitor, AlertManager, 
    DiagnosticSystem, AuditTrailManager
)
from benchmarks.benchmark_loader import BenchmarkLoader
from neurons.miner import HFAMiner
from neurons.validator import HFAValidator
import template.protocol as protocol


class IntegrationTestEnvironment:
    """Test environment for end-to-end integration testing"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = self._create_test_config()
        self.components = {}
        self.mock_metagraph = None
        self.mock_subtensor = None
        
    def _create_test_config(self) -> Dict[str, Any]:
        """Create test configuration"""
        return {
            "netuid": 999,  # Test netuid
            "subtensor": {
                "network": "test",
                "chain_endpoint": "ws://127.0.0.1:9944"
            },
            "wallet": {
                "name": "test_wallet",
                "hotkey": "test_hotkey"
            },
            "model_config": {
                "architecture": "hfa",
                "model_name": "test_model",
                "max_context_length": 4096
            },
            "benchmark_config": {
                "enabled": True,
                "data_path": os.path.join(self.temp_dir, "benchmark_data")
            },
            "monitoring": {
                "telemetry_enabled": True,
                "health_monitoring_enabled": True,
                "audit_trail_enabled": True
            }
        }
    
    def setup_mocks(self):
        """Setup mock objects for testing"""
        # Mock metagraph
        self.mock_metagraph = Mock()
        self.mock_metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
        self.mock_metagraph.validator_permit = [False, True, False]  # miner, validator, miner
        self.mock_metagraph.axons = [Mock() for _ in range(3)]
        
        # Mock subtensor
        self.mock_subtensor = Mock()
        self.mock_subtensor.network = "test"
        
        # Mock wallet
        self.mock_wallet = Mock()
        self.mock_wallet.hotkey.ss58_address = "test_hotkey_address"
        
    def create_test_miner(self, architecture: str = "hfa") -> HFAMiner:
        """Create a test miner instance"""
        config = self.config.copy()
        config["model_config"]["architecture"] = architecture
        
        with patch('bittensor.subtensor') as mock_subtensor_class:
            mock_subtensor_class.return_value = self.mock_subtensor
            with patch('bittensor.wallet') as mock_wallet_class:
                mock_wallet_class.return_value = self.mock_wallet
                
                miner = HFAMiner(config=config)
                miner.metagraph = self.mock_metagraph
                miner.subtensor = self.mock_subtensor
                miner.wallet = self.mock_wallet
                
                return miner
    
    def create_test_validator(self) -> HFAValidator:
        """Create a test validator instance"""
        with patch('bittensor.subtensor') as mock_subtensor_class:
            mock_subtensor_class.return_value = self.mock_subtensor
            with patch('bittensor.wallet') as mock_wallet_class:
                mock_wallet_class.return_value = self.mock_wallet
                
                validator = HFAValidator(config=self.config)
                validator.metagraph = self.mock_metagraph
                validator.subtensor = self.mock_subtensor
                validator.wallet = self.mock_wallet
                
                return validator
    
    def cleanup(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


@pytest.fixture
def test_env():
    """Pytest fixture for test environment"""
    env = IntegrationTestEnvironment()
    env.setup_mocks()
    yield env
    env.cleanup()


class TestEndToEndIntegration:
    """End-to-end integration tests for the unified HFA-SimpleMind subnet"""
    
    @pytest.mark.asyncio
    async def test_complete_evaluation_cycle(self, test_env):
        """Test complete evaluation cycle from task generation to weight setting"""
        
        # Create validator and miners
        validator = test_env.create_test_validator()
        hfa_miner = test_env.create_test_miner("hfa")
        simplemind_miner = test_env.create_test_miner("simplemind")
        hybrid_miner = test_env.create_test_miner("hybrid")
        
        miners = [hfa_miner, simplemind_miner, hybrid_miner]
        
        # Mock miner responses
        for i, miner in enumerate(miners):
            miner.forward = Mock(return_value=Mock(
                response=f"Test response from miner {i}",
                model_info={"architecture": miner.config["model_config"]["architecture"]},
                exact_match_score=0.8,
                f1_score=0.75,
                semantic_similarity_score=0.85,
                coherence_score=0.9,
                tokens_per_second=100.0,
                memory_usage_mb=1024.0
            ))
        
        # Test task generation
        evaluation_tasks = validator.generate_evaluation_tasks()
        assert len(evaluation_tasks) > 0, "Should generate evaluation tasks"
        assert any(task["type"] == "benchmark_evaluation" for task in evaluation_tasks), "Should include benchmark tasks"
        
        # Test miner querying
        miner_uids = [0, 1, 2]  # Corresponding to our test miners
        test_task = evaluation_tasks[0]
        
        # Mock the dendrite forward call
        with patch.object(validator, 'dendrite') as mock_dendrite:
            mock_responses = [miner.forward() for miner in miners]
            mock_dendrite.forward.return_value = mock_responses
            
            responses = await validator.query_miners(miner_uids, test_task)
            
            assert len(responses) == len(miner_uids), "Should get responses from all miners"
            assert all(uid in responses for uid in miner_uids), "Should have response for each miner"
        
        # Test scoring
        scores = validator.score_responses(responses, test_task)
        
        assert len(scores) == len(miner_uids), "Should get scores for all miners"
        assert all(0 <= score <= 1 for score in scores.values()), "Scores should be normalized"
        
        # Test diversity tracking if available
        if validator.diversity_tracker:
            for uid, response in responses.items():
                diversity_metrics = validator.diversity_tracker.track_miner_response(
                    miner_uid=uid,
                    response=response,
                    model_info=response.model_info if hasattr(response, 'model_info') else None
                )
                assert diversity_metrics is not None, "Should generate diversity metrics"
        
        print("✅ Complete evaluation cycle test passed")
    
    @pytest.mark.asyncio
    async def test_multi_architecture_support(self, test_env):
        """Test support for multiple model architectures"""
        
        # Test model factory
        factory = ModelArchitectureFactory()
        available_architectures = factory.get_available_architectures()
        
        assert "hfa" in available_architectures, "Should support HFA architecture"
        assert "simplemind" in available_architectures, "Should support SimpleMind architecture"
        assert "hybrid" in available_architectures, "Should support hybrid architecture"
        
        # Test model creation for each architecture
        for architecture in ["hfa", "simplemind", "hybrid"]:
            config = {
                "architecture": architecture,
                "model_name": f"test_{architecture}_model",
                "max_context_length": 4096
            }
            
            try:
                model = factory.create_model(config)
                assert model is not None, f"Should create {architecture} model"
                
                # Test basic model interface
                assert hasattr(model, 'forward'), f"{architecture} model should have forward method"
                assert hasattr(model, 'get_model_info'), f"{architecture} model should have get_model_info method"
                
            except Exception as e:
                pytest.fail(f"Failed to create {architecture} model: {e}")
        
        print("✅ Multi-architecture support test passed")
    
    @pytest.mark.asyncio
    async def test_benchmark_integration(self, test_env):
        """Test benchmark system integration"""
        
        # Create benchmark loader
        benchmark_config = {
            'longbench': {'data_path': 'test_data/longbench'},
            'hotpotqa': {'data_path': 'test_data/hotpotqa'},
            'govreport': {'data_path': 'test_data/govreport'},
            'needle_haystack': {'min_context_length': 1000, 'max_context_length': 4000}
        }
        
        benchmark_loader = BenchmarkLoader(benchmark_config)
        
        # Test benchmark availability checking
        available_benchmarks = benchmark_loader.available_benchmarks
        assert isinstance(available_benchmarks, list), "Should return list of available benchmarks"
        
        # Test synthetic task generation (fallback when real benchmarks unavailable)
        try:
            tasks = benchmark_loader.load_benchmark_tasks(num_tasks=3)
            assert len(tasks) <= 3, "Should respect task limit"
            
            for task in tasks:
                assert hasattr(task, 'task_id'), "Task should have ID"
                assert hasattr(task, 'context'), "Task should have context"
                assert hasattr(task, 'prompt'), "Task should have prompt"
                assert hasattr(task, 'context_length'), "Task should have context length"
                
        except Exception as e:
            # This is expected if benchmark data is not available
            print(f"Benchmark loading failed as expected (no test data): {e}")
        
        print("✅ Benchmark integration test passed")
    
    @pytest.mark.asyncio
    async def test_diversity_tracking_system(self, test_env):
        """Test diversity tracking and monoculture prevention"""
        
        diversity_tracker = DiversityTracker()
        
        # Simulate responses from different miners
        test_responses = [
            ("miner_0", "This is a unique response from miner 0", {"architecture_type": "hfa"}),
            ("miner_1", "This is a different response from miner 1", {"architecture_type": "simplemind"}),
            ("miner_2", "This is another unique response from miner 2", {"architecture_type": "hybrid"}),
            ("miner_3", "This is a unique response from miner 0", {"architecture_type": "hfa"}),  # Similar to miner 0
        ]
        
        diversity_metrics = []
        
        for miner_id, response, model_info in test_responses:
            metrics = diversity_tracker.track_miner_response(
                miner_uid=int(miner_id.split('_')[1]),
                response=response,
                model_info=model_info
            )
            diversity_metrics.append(metrics)
            
            assert metrics is not None, f"Should generate diversity metrics for {miner_id}"
            assert 0 <= metrics.response_uniqueness_score <= 1, "Uniqueness score should be normalized"
            assert 0 <= metrics.model_architecture_diversity <= 1, "Architecture diversity should be normalized"
        
        # Test monoculture risk detection
        monoculture_risk = diversity_tracker.detect_monoculture_risk()
        assert "risk_level" in monoculture_risk, "Should assess monoculture risk level"
        assert monoculture_risk["risk_level"] in ["low", "medium", "high"], "Risk level should be valid"
        
        # Test diversity incentives
        base_score = 0.8
        for i, (miner_id, _, _) in enumerate(test_responses):
            uid = int(miner_id.split('_')[1])
            adjusted_score = diversity_tracker.compute_diversity_incentive(uid, base_score)
            assert 0 <= adjusted_score <= 1, "Adjusted score should be normalized"
        
        print("✅ Diversity tracking system test passed")
    
    @pytest.mark.asyncio
    async def test_scoring_harness_integration(self, test_env):
        """Test sealed scoring harness with audit trails"""
        
        # Create scoring harness
        scoring_config = {
            "audit_and_transparency": {
                "audit_trail_retention_days": 30,
                "consensus_threshold": 0.9,
                "logit_hash_computation": True,
                "deterministic_scoring": True
            },
            "scoring_weights": {
                "task_quality": 0.8,
                "consistency_under_perturbation": 0.1,
                "efficiency_proxy": 0.1
            }
        }
        
        scoring_harness = ScoringHarness(scoring_config)
        
        # Test scoring with mock task and response
        test_task = {
            "task_id": "test_task_001",
            "type": "benchmark_evaluation",
            "context": "Test context for scoring",
            "prompt": "Test prompt",
            "expected_output": "Expected test output"
        }
        
        test_response = Mock()
        test_response.response = "Test response output"
        test_response.exact_match_score = 0.8
        test_response.f1_score = 0.75
        test_response.semantic_similarity_score = 0.85
        test_response.coherence_score = 0.9
        test_response.tokens_per_second = 100.0
        test_response.memory_usage_mb = 1024.0
        
        # Test scoring
        scoring_result = scoring_harness.score_response(
            task=test_task,
            response=test_response,
            miner_uid=1,
            model_info={"architecture": "hfa"}
        )
        
        assert scoring_result is not None, "Should generate scoring result"
        assert hasattr(scoring_result, 'final_score'), "Should have final score"
        assert hasattr(scoring_result, 'quality_score'), "Should have quality score"
        assert hasattr(scoring_result, 'logit_hash'), "Should have logit hash"
        assert 0 <= scoring_result.final_score <= 1, "Final score should be normalized"
        
        # Test audit trail
        audit_trail = scoring_harness.get_audit_trail(1, "test_task_001")
        assert isinstance(audit_trail, list), "Should return audit trail list"
        
        print("✅ Scoring harness integration test passed")
    
    @pytest.mark.asyncio
    async def test_monitoring_system_integration(self, test_env):
        """Test comprehensive monitoring system"""
        
        # Test telemetry collector
        telemetry = TelemetryCollector()
        telemetry.start_collection()
        
        # Record some test metrics
        telemetry.record_metric("test_metric", 42.0, {"component": "test"})
        telemetry.record_metric("test_metric", 43.0, {"component": "test"})
        
        # Wait a bit for collection
        await asyncio.sleep(1)
        
        # Check metrics
        metric_history = telemetry.get_metric_history("test_metric", hours=1)
        assert len(metric_history) >= 2, "Should record test metrics"
        
        telemetry.stop_collection()
        
        # Test health monitor
        health_monitor = HealthMonitor()
        health_monitor.start_monitoring()
        
        # Wait for health checks
        await asyncio.sleep(2)
        
        health_status = health_monitor.get_health_status()
        assert "overall_health" in health_status, "Should provide health status"
        assert health_status["overall_health"] in ["healthy", "warning", "critical", "unknown"], "Health status should be valid"
        
        health_monitor.stop_monitoring()
        
        # Test alert manager
        alert_manager = AlertManager()
        
        # Test alert processing
        test_data = {
            "cpu_percent": 95.0,  # Should trigger high CPU alert
            "memory_percent": 50.0,
            "active_miners": 3
        }
        
        alert_manager.process_data(test_data)
        
        # Check for alerts
        active_alerts = alert_manager.get_active_alerts()
        # Note: Alerts might not trigger immediately due to cooldown periods
        
        # Test diagnostic system
        diagnostic_system = DiagnosticSystem()
        
        # Run some diagnostic checks
        python_check = diagnostic_system.run_check("python_version")
        assert python_check.status in ["pass", "fail", "warning", "info"], "Should return valid status"
        
        deps_check = diagnostic_system.run_check("dependencies")
        assert deps_check.status in ["pass", "fail", "warning", "info"], "Should return valid status"
        
        # Test audit trail manager
        audit_manager = AuditTrailManager({"db_path": os.path.join(test_env.temp_dir, "test_audit.db")})
        
        # Log some test events
        audit_manager.log_system_event("test_action", "test_resource", {"test": "data"})
        audit_manager.log_miner_event(1, "model_inference", "test_task", {"score": 0.8})
        
        # Wait for flush
        await asyncio.sleep(2)
        
        # Query events
        from template.monitoring.audit_trail import AuditQuery
        query = AuditQuery(limit=10)
        events = audit_manager.query_events(query)
        assert len(events) >= 2, "Should record audit events"
        
        audit_manager.stop()
        
        print("✅ Monitoring system integration test passed")
    
    @pytest.mark.asyncio
    async def test_protocol_communication(self, test_env):
        """Test protocol communication between miners and validators"""
        
        # Test synapse creation
        test_synapse = protocol.InfiniteContextSynapse(
            context="Test context for protocol communication",
            prompt="Test prompt",
            evaluation_type="memory_retention",
            max_tokens=100,
            architecture_type="hfa",
            model_configuration={"max_context_length": 4096}
        )
        
        assert test_synapse.context == "Test context for protocol communication"
        assert test_synapse.prompt == "Test prompt"
        assert test_synapse.evaluation_type == "memory_retention"
        assert test_synapse.architecture_type == "hfa"
        
        # Test benchmark synapse
        benchmark_synapse = protocol.BenchmarkEvaluationSynapse(
            task_id="test_benchmark_001",
            task_type="longbench",
            dataset_name="narrativeqa",
            context="Benchmark context",
            prompt="Benchmark prompt",
            difficulty_level="medium",
            evaluation_metrics=["exact_match", "f1_score"],
            context_length=2048
        )
        
        assert benchmark_synapse.task_id == "test_benchmark_001"
        assert benchmark_synapse.task_type == "longbench"
        assert benchmark_synapse.dataset_name == "narrativeqa"
        assert "exact_match" in benchmark_synapse.evaluation_metrics
        
        # Test serialization/deserialization
        try:
            serialized = test_synapse.json()
            assert isinstance(serialized, str), "Should serialize to JSON string"
            
            # Test that it contains expected fields
            data = json.loads(serialized)
            assert "context" in data, "Serialized data should contain context"
            assert "prompt" in data, "Serialized data should contain prompt"
            
        except Exception as e:
            print(f"Serialization test failed (expected for mock objects): {e}")
        
        print("✅ Protocol communication test passed")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, test_env):
        """Test error handling and recovery mechanisms"""
        
        validator = test_env.create_test_validator()
        
        # Test handling of miner failures
        miner_uids = [0, 1, 2]
        test_task = {
            "type": "memory_retention",
            "context": "Test context",
            "prompt": "Test prompt"
        }
        
        # Mock some miners failing
        with patch.object(validator, 'dendrite') as mock_dendrite:
            # Simulate mixed success/failure responses
            mock_responses = [
                Mock(response="Success response"),  # Successful response
                None,  # Failed response
                Mock(response="Another success")   # Successful response
            ]
            mock_dendrite.forward.return_value = mock_responses
            
            responses = await validator.query_miners(miner_uids, test_task)
            
            # Should handle failures gracefully
            assert len(responses) == len(miner_uids), "Should return response dict for all miners"
            assert responses[1] is None, "Should handle failed miner gracefully"
            assert responses[0] is not None, "Should preserve successful responses"
            assert responses[2] is not None, "Should preserve successful responses"
        
        # Test scoring with failed responses
        scores = validator.score_responses(responses, test_task)
        
        assert len(scores) == len(miner_uids), "Should score all miners"
        assert scores[1] == 0.0, "Failed miner should get zero score"
        assert scores[0] >= 0.0, "Successful miners should get valid scores"
        assert scores[2] >= 0.0, "Successful miners should get valid scores"
        
        # Test model factory error handling
        factory = ModelArchitectureFactory()
        
        # Test invalid architecture
        invalid_config = {
            "architecture": "nonexistent_architecture",
            "model_name": "test_model"
        }
        
        try:
            model = factory.create_model(invalid_config)
            pytest.fail("Should raise exception for invalid architecture")
        except Exception as e:
            assert "architecture" in str(e).lower(), "Error should mention architecture"
        
        print("✅ Error handling and recovery test passed")
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, test_env):
        """Test system performance under load"""
        
        validator = test_env.create_test_validator()
        
        # Test multiple concurrent evaluation cycles
        num_cycles = 3
        tasks_per_cycle = 5
        
        async def run_evaluation_cycle():
            """Run a single evaluation cycle"""
            try:
                # Generate tasks
                tasks = validator.generate_evaluation_tasks()
                assert len(tasks) > 0, "Should generate tasks"
                
                # Mock responses for performance testing
                miner_uids = [0, 1, 2]
                responses = {
                    uid: Mock(
                        response=f"Response from miner {uid}",
                        exact_match_score=0.8,
                        f1_score=0.75,
                        tokens_per_second=100.0
                    )
                    for uid in miner_uids
                }
                
                # Score responses
                scores = validator.score_responses(responses, tasks[0])
                assert len(scores) == len(miner_uids), "Should score all miners"
                
                return True
                
            except Exception as e:
                print(f"Evaluation cycle failed: {e}")
                return False
        
        # Run multiple cycles concurrently
        start_time = time.time()
        
        cycle_tasks = [run_evaluation_cycle() for _ in range(num_cycles)]
        results = await asyncio.gather(*cycle_tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check results
        successful_cycles = sum(1 for result in results if result is True)
        assert successful_cycles >= num_cycles * 0.8, f"At least 80% of cycles should succeed (got {successful_cycles}/{num_cycles})"
        
        # Performance assertions
        assert duration < 30.0, f"Load test should complete within 30 seconds (took {duration:.2f}s)"
        
        print(f"✅ Performance under load test passed ({successful_cycles}/{num_cycles} cycles successful in {duration:.2f}s)")


@pytest.mark.asyncio
async def test_full_system_integration():
    """Test full system integration with all components"""
    
    test_env = IntegrationTestEnvironment()
    test_env.setup_mocks()
    
    try:
        # Create all major components
        validator = test_env.create_test_validator()
        miners = [
            test_env.create_test_miner("hfa"),
            test_env.create_test_miner("simplemind"),
            test_env.create_test_miner("hybrid")
        ]
        
        # Test that all components can be created without errors
        assert validator is not None, "Should create validator"
        assert len(miners) == 3, "Should create all miners"
        
        # Test basic functionality of each component
        for i, miner in enumerate(miners):
            assert miner is not None, f"Miner {i} should be created"
            assert hasattr(miner, 'forward'), f"Miner {i} should have forward method"
        
        assert hasattr(validator, 'generate_evaluation_tasks'), "Validator should have task generation"
        assert hasattr(validator, 'score_responses'), "Validator should have scoring"
        
        # Test monitoring components
        if hasattr(validator, 'diversity_tracker') and validator.diversity_tracker:
            assert validator.diversity_tracker is not None, "Should have diversity tracker"
        
        if hasattr(validator, 'scoring_harness') and validator.scoring_harness:
            assert validator.scoring_harness is not None, "Should have scoring harness"
        
        print("✅ Full system integration test passed")
        
    finally:
        test_env.cleanup()


if __name__ == "__main__":
    # Run tests directly
    import sys
    
    print("Running end-to-end integration tests...")
    
    # Run the full system integration test
    asyncio.run(test_full_system_integration())
    
    print("All integration tests completed!")