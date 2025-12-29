# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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
import hashlib
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import bittensor as bt


@dataclass
class ScoringResult:
    """Complete scoring result with audit information"""
    task_id: str
    miner_uid: int
    task_type: str
    quality_score: float
    consistency_score: float
    efficiency_score: float
    final_score: float
    logit_hash: Optional[str] = None
    model_signature: Optional[str] = None
    audit_trail: List[str] = None
    timestamp: float = None
    perturbation_scores: Optional[Dict[str, float]] = None
    score_components: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.audit_trail is None:
            self.audit_trail = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def add_audit_entry(self, entry: str):
        """Add entry to audit trail"""
        timestamp_str = datetime.fromtimestamp(time.time()).isoformat()
        self.audit_trail.append(f"[{timestamp_str}] {entry}")


@dataclass
class PerturbationTest:
    """Configuration for perturbation testing"""
    perturbation_type: str  # "paraphrase", "reorder", "noise_injection"
    original_task: Dict[str, Any]
    perturbed_task: Dict[str, Any]
    expected_consistency_threshold: float = 0.85
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Handles audit logging for scoring decisions"""
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.audit_entries = []
        
    def log_scoring_decision(self, miner_uid: int, task_id: str, 
                           decision_type: str, details: Dict[str, Any]):
        """Log a scoring decision with full context"""
        entry = {
            "timestamp": time.time(),
            "miner_uid": miner_uid,
            "task_id": task_id,
            "decision_type": decision_type,
            "details": details,
            "hash": self._compute_entry_hash(miner_uid, task_id, decision_type, details)
        }
        self.audit_entries.append(entry)
        
        bt.logging.info(f"🔍 Audit: {decision_type} for miner {miner_uid}, task {task_id}")
        
    def _compute_entry_hash(self, miner_uid: int, task_id: str, 
                          decision_type: str, details: Dict[str, Any]) -> str:
        """Compute deterministic hash for audit entry"""
        content = f"{miner_uid}:{task_id}:{decision_type}:{json.dumps(details, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def cleanup_old_entries(self):
        """Remove audit entries older than retention period"""
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)
        self.audit_entries = [
            entry for entry in self.audit_entries 
            if entry["timestamp"] > cutoff_time
        ]
        
    def get_audit_trail(self, miner_uid: int, task_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for specific miner and task"""
        return [
            entry for entry in self.audit_entries
            if entry["miner_uid"] == miner_uid and entry["task_id"] == task_id
        ]


class ConsensusChecker:
    """Handles consensus validation across multiple validators"""
    
    def __init__(self, consensus_threshold: float = 0.9):
        self.consensus_threshold = consensus_threshold
        self.validator_scores = {}
        
    def add_validator_score(self, validator_id: str, miner_uid: int, 
                          task_id: str, score: float):
        """Add score from a validator for consensus checking"""
        key = f"{miner_uid}:{task_id}"
        if key not in self.validator_scores:
            self.validator_scores[key] = {}
        self.validator_scores[key][validator_id] = score
        
    def check_consensus(self, miner_uid: int, task_id: str) -> Tuple[bool, float, Dict[str, float]]:
        """Check if consensus exists among validators for a score"""
        key = f"{miner_uid}:{task_id}"
        if key not in self.validator_scores:
            return False, 0.0, {}
        
        scores = list(self.validator_scores[key].values())
        if len(scores) < 2:
            return True, scores[0] if scores else 0.0, self.validator_scores[key]
        
        # Calculate consensus metrics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Check if all scores are within acceptable range
        consensus_achieved = all(
            abs(score - mean_score) <= (1 - self.consensus_threshold) * mean_score
            for score in scores
        )
        
        return consensus_achieved, mean_score, self.validator_scores[key]
    
    def get_consensus_score(self, miner_uid: int, task_id: str) -> Optional[float]:
        """Get consensus score if available"""
        consensus_achieved, mean_score, _ = self.check_consensus(miner_uid, task_id)
        return mean_score if consensus_achieved else None


class PerturbationTester:
    """Handles perturbation testing for consistency evaluation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.perturbation_types = config.get("perturbation_types", ["paraphrase", "reorder", "noise_injection"])
        self.consistency_threshold = config.get("consistency_threshold", 0.85)
        self.perturbation_frequency = config.get("perturbation_frequency", 0.2)
        
    def should_apply_perturbation(self) -> bool:
        """Determine if perturbation should be applied to current task"""
        return np.random.random() < self.perturbation_frequency
    
    def generate_perturbation_tests(self, original_task: Dict[str, Any]) -> List[PerturbationTest]:
        """Generate perturbation variants of the original task"""
        perturbation_tests = []
        
        for perturbation_type in self.perturbation_types:
            try:
                perturbed_task = self._apply_perturbation(original_task, perturbation_type)
                if perturbed_task:
                    test = PerturbationTest(
                        perturbation_type=perturbation_type,
                        original_task=original_task,
                        perturbed_task=perturbed_task,
                        expected_consistency_threshold=self.consistency_threshold
                    )
                    perturbation_tests.append(test)
            except Exception as e:
                bt.logging.warning(f"Failed to generate {perturbation_type} perturbation: {e}")
        
        return perturbation_tests
    
    def _apply_perturbation(self, task: Dict[str, Any], perturbation_type: str) -> Optional[Dict[str, Any]]:
        """Apply specific perturbation to task"""
        perturbed_task = task.copy()
        
        if perturbation_type == "paraphrase":
            # Simple paraphrasing by synonym replacement
            perturbed_task["prompt"] = self._paraphrase_text(task["prompt"])
            
        elif perturbation_type == "reorder":
            # Reorder sentences in context while preserving meaning
            perturbed_task["context"] = self._reorder_sentences(task["context"])
            
        elif perturbation_type == "noise_injection":
            # Add minimal noise that shouldn't affect core meaning
            perturbed_task["context"] = self._inject_noise(task["context"])
            
        else:
            return None
        
        return perturbed_task
    
    def _paraphrase_text(self, text: str) -> str:
        """Simple paraphrasing by replacing common words"""
        replacements = {
            "What": "Which",
            "How": "In what way",
            "Where": "At what location",
            "When": "At what time",
            "the": "a",
            "is": "was",
            "are": "were"
        }
        
        paraphrased = text
        for original, replacement in replacements.items():
            if original in paraphrased:
                paraphrased = paraphrased.replace(original, replacement, 1)
                break  # Only apply one replacement to keep changes minimal
        
        return paraphrased
    
    def _reorder_sentences(self, context: str) -> str:
        """Reorder sentences while preserving overall meaning"""
        sentences = context.split('. ')
        if len(sentences) > 3:
            # Swap two adjacent sentences in the middle
            mid_idx = len(sentences) // 2
            if mid_idx > 0 and mid_idx < len(sentences) - 1:
                sentences[mid_idx], sentences[mid_idx + 1] = sentences[mid_idx + 1], sentences[mid_idx]
        
        return '. '.join(sentences)
    
    def _inject_noise(self, context: str) -> str:
        """Inject minimal noise that shouldn't affect meaning"""
        # Add a few extra spaces or punctuation
        words = context.split()
        if len(words) > 10:
            # Add extra space after a random word
            idx = np.random.randint(5, len(words) - 5)
            words[idx] = words[idx] + " "
        
        return ' '.join(words)
    
    def evaluate_consistency(self, original_response: Any, perturbed_responses: List[Any], 
                           perturbation_tests: List[PerturbationTest]) -> Dict[str, float]:
        """Evaluate consistency between original and perturbed responses"""
        consistency_scores = {}
        
        for i, (perturbed_response, test) in enumerate(zip(perturbed_responses, perturbation_tests)):
            try:
                consistency_score = self._compute_response_similarity(
                    original_response, perturbed_response
                )
                consistency_scores[test.perturbation_type] = consistency_score
                
                bt.logging.debug(f"Consistency for {test.perturbation_type}: {consistency_score:.3f}")
                
            except Exception as e:
                bt.logging.error(f"Error computing consistency for {test.perturbation_type}: {e}")
                consistency_scores[test.perturbation_type] = 0.0
        
        return consistency_scores
    
    def _compute_response_similarity(self, response1: Any, response2: Any) -> float:
        """Compute similarity between two responses"""
        # Extract response text
        text1 = self._extract_response_text(response1)
        text2 = self._extract_response_text(response2)
        
        if not text1 or not text2:
            return 0.0
        
        # Simple token-based similarity
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_response_text(self, response: Any) -> str:
        """Extract text from response object"""
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return response.get("response", "") or response.get("text", "")
        elif hasattr(response, "response"):
            return getattr(response, "response", "")
        else:
            return str(response)


class ScoringHarness:
    """
    Sealed, auditable scoring system for the unified HFA-SimpleMind subnet.
    
    This harness provides:
    - Deterministic scoring algorithms with audit trails
    - Logit hash computation for response verification
    - Perturbation testing for consistency evaluation
    - Consensus checking across multiple validators
    - Comprehensive audit logging for transparency
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        audit_config = config.get("audit_and_transparency", {})
        self.audit_logger = AuditLogger(
            retention_days=audit_config.get("audit_trail_retention_days", 30)
        )
        
        self.consensus_checker = ConsensusChecker(
            consensus_threshold=audit_config.get("consensus_threshold", 0.9)
        )
        
        perturbation_config = config.get("perturbation_testing", {})
        self.perturbation_tester = PerturbationTester(perturbation_config)
        
        # Scoring configuration
        self.scoring_weights = config.get("scoring_weights", {
            "task_quality": 0.8,
            "consistency_under_perturbation": 0.1,
            "efficiency_proxy": 0.1
        })
        
        self.detailed_weights = config.get("detailed_scoring_weights", {})
        
        # Enable features
        self.logit_hash_enabled = audit_config.get("logit_hash_computation", True)
        self.consensus_validation_enabled = audit_config.get("consensus_validation", True)
        self.deterministic_scoring_enabled = audit_config.get("deterministic_scoring", True)
        
        bt.logging.info("🔒 ScoringHarness initialized with sealed audit capabilities")
    
    def score_response(self, task: Dict[str, Any], response: Any, 
                      miner_uid: int, model_info: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score a single response with full audit trail and perturbation testing.
        
        Args:
            task: The evaluation task
            response: The miner's response
            miner_uid: Unique identifier for the miner
            model_info: Optional model information for audit
            
        Returns:
            ScoringResult with comprehensive scoring and audit information
        """
        task_id = task.get("task_id", f"task_{int(time.time())}")
        task_type = task.get("type", "unknown")
        
        # Initialize scoring result
        scoring_result = ScoringResult(
            task_id=task_id,
            miner_uid=miner_uid,
            task_type=task_type,
            quality_score=0.0,
            consistency_score=0.0,
            efficiency_score=0.0,
            final_score=0.0
        )
        
        try:
            # Log scoring initiation
            self.audit_logger.log_scoring_decision(
                miner_uid, task_id, "scoring_initiated",
                {"task_type": task_type, "model_info": model_info}
            )
            scoring_result.add_audit_entry("Scoring initiated")
            
            # Compute logit hash if enabled and available
            if self.logit_hash_enabled:
                logit_hash = self.compute_logit_hash(response)
                scoring_result.logit_hash = logit_hash
                scoring_result.add_audit_entry(f"Logit hash computed: {logit_hash}")
            
            # Compute model signature
            if model_info:
                model_signature = self.compute_model_signature(model_info)
                scoring_result.model_signature = model_signature
                scoring_result.add_audit_entry(f"Model signature: {model_signature}")
            
            # Score quality components
            quality_score, score_components = self._score_quality(task, response, task_type)
            scoring_result.quality_score = quality_score
            scoring_result.score_components = score_components
            scoring_result.add_audit_entry(f"Quality score computed: {quality_score:.4f}")
            
            # Score efficiency
            efficiency_score = self._score_efficiency(response)
            scoring_result.efficiency_score = efficiency_score
            scoring_result.add_audit_entry(f"Efficiency score computed: {efficiency_score:.4f}")
            
            # Apply perturbation testing if enabled
            consistency_score = 0.0
            if self.perturbation_tester.should_apply_perturbation():
                consistency_score, perturbation_scores = self._apply_perturbation_testing(
                    task, response, miner_uid
                )
                scoring_result.consistency_score = consistency_score
                scoring_result.perturbation_scores = perturbation_scores
                scoring_result.add_audit_entry(f"Perturbation testing applied: {consistency_score:.4f}")
            else:
                scoring_result.consistency_score = 1.0  # No perturbation applied, assume consistent
                scoring_result.add_audit_entry("No perturbation testing applied")
            
            # Calculate final score
            final_score = self._calculate_final_score(
                quality_score, consistency_score, efficiency_score
            )
            scoring_result.final_score = final_score
            scoring_result.add_audit_entry(f"Final score calculated: {final_score:.4f}")
            
            # Log final scoring decision
            self.audit_logger.log_scoring_decision(
                miner_uid, task_id, "scoring_completed",
                {
                    "quality_score": quality_score,
                    "consistency_score": consistency_score,
                    "efficiency_score": efficiency_score,
                    "final_score": final_score,
                    "score_components": score_components
                }
            )
            
        except Exception as e:
            bt.logging.error(f"Error in scoring harness for miner {miner_uid}: {e}")
            scoring_result.add_audit_entry(f"Scoring error: {str(e)}")
            self.audit_logger.log_scoring_decision(
                miner_uid, task_id, "scoring_error", {"error": str(e)}
            )
        
        return scoring_result
    
    def compute_logit_hash(self, response: Any) -> str:
        """
        Compute deterministic hash of model logits for audit trails.
        
        Args:
            response: The response object containing logits or response data
            
        Returns:
            Hexadecimal hash string for audit verification
        """
        try:
            # Extract logits or response content for hashing
            if hasattr(response, 'logits') and response.logits is not None:
                # Hash actual logits if available
                if isinstance(response.logits, torch.Tensor):
                    logits_data = response.logits.detach().cpu().numpy()
                else:
                    logits_data = np.array(response.logits)
                
                # Create deterministic hash of logits
                logits_bytes = logits_data.tobytes()
                hash_obj = hashlib.sha256(logits_bytes)
                
            else:
                # Fall back to hashing response content
                response_text = self._extract_response_content(response)
                hash_obj = hashlib.sha256(response_text.encode('utf-8'))
            
            # Add timestamp for uniqueness while maintaining determinism within evaluation
            evaluation_timestamp = int(time.time() / 3600) * 3600  # Hour-level granularity
            hash_obj.update(str(evaluation_timestamp).encode('utf-8'))
            
            return hash_obj.hexdigest()[:16]  # 16-character hash for readability
            
        except Exception as e:
            bt.logging.warning(f"Failed to compute logit hash: {e}")
            # Return hash of error message as fallback
            return hashlib.sha256(f"hash_error_{str(e)}".encode()).hexdigest()[:16]
    
    def compute_model_signature(self, model_info: Dict[str, Any]) -> str:
        """Compute signature for model configuration"""
        try:
            # Create deterministic signature from model info
            signature_data = {
                "architecture": model_info.get("architecture", "unknown"),
                "model_size": model_info.get("model_size", "unknown"),
                "config_hash": model_info.get("config_hash", "unknown")
            }
            
            signature_str = json.dumps(signature_data, sort_keys=True)
            return hashlib.sha256(signature_str.encode()).hexdigest()[:12]
            
        except Exception as e:
            bt.logging.warning(f"Failed to compute model signature: {e}")
            return "sig_error"
    
    def _extract_response_content(self, response: Any) -> str:
        """Extract content from response for hashing"""
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return json.dumps(response, sort_keys=True)
        elif hasattr(response, 'response'):
            return str(response.response)
        else:
            return str(response)
    
    def _score_quality(self, task: Dict[str, Any], response: Any, 
                      task_type: str) -> Tuple[float, Dict[str, float]]:
        """Score response quality based on task type"""
        score_components = {}
        
        try:
            if task_type == "benchmark_evaluation":
                return self._score_benchmark_quality(task, response, score_components)
            else:
                return self._score_synthetic_quality(task, response, score_components)
                
        except Exception as e:
            bt.logging.error(f"Error scoring quality: {e}")
            return 0.0, {"error": 0.0}
    
    def _score_benchmark_quality(self, task: Dict[str, Any], response: Any, 
                                score_components: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Score quality for benchmark tasks"""
        # Extract benchmark-specific metrics
        if isinstance(response, dict):
            exact_match = response.get('exact_match_score', 0.0)
            f1_score = response.get('f1_score', 0.0)
            semantic_sim = response.get('semantic_similarity_score', 0.0)
            coherence = response.get('coherence_score', 0.0)
        else:
            exact_match = getattr(response, 'exact_match_score', 0.0)
            f1_score = getattr(response, 'f1_score', 0.0)
            semantic_sim = getattr(response, 'semantic_similarity_score', 0.0)
            coherence = getattr(response, 'coherence_score', 0.0)
        
        # Weight components based on detailed weights
        score_components['exact_match'] = exact_match * self.detailed_weights.get('exact_match_score', 0.3)
        score_components['f1_score'] = f1_score * self.detailed_weights.get('f1_score', 0.3)
        score_components['semantic_similarity'] = semantic_sim * self.detailed_weights.get('semantic_similarity_score', 0.2)
        score_components['coherence'] = coherence * self.detailed_weights.get('coherence_score', 0.2)
        
        quality_score = sum(score_components.values())
        return min(1.0, max(0.0, quality_score)), score_components
    
    def _score_synthetic_quality(self, task: Dict[str, Any], response: Any, 
                                score_components: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Score quality for synthetic tasks"""
        # Extract synthetic task metrics
        if isinstance(response, dict):
            memory_score = response.get('memory_retention_score', 0.0)
            position_score = response.get('position_understanding_score', 0.0)
            coherence_score = response.get('coherence_score', 0.0)
        else:
            memory_score = getattr(response, 'memory_retention_score', 0.0)
            position_score = getattr(response, 'position_understanding_score', 0.0)
            coherence_score = getattr(response, 'coherence_score', 0.0)
        
        # Weight components
        score_components['memory_retention'] = memory_score * self.detailed_weights.get('memory_retention_score', 0.4)
        score_components['position_understanding'] = position_score * self.detailed_weights.get('position_understanding_score', 0.3)
        score_components['coherence'] = coherence_score * self.detailed_weights.get('coherence_score', 0.3)
        
        quality_score = sum(score_components.values())
        return min(1.0, max(0.0, quality_score)), score_components
    
    def _score_efficiency(self, response: Any) -> float:
        """Score response efficiency"""
        try:
            if isinstance(response, dict):
                tokens_per_sec = response.get('tokens_per_second', 0.0)
                memory_usage = response.get('memory_usage_mb', 0.0)
            else:
                tokens_per_sec = getattr(response, 'tokens_per_second', 0.0)
                memory_usage = getattr(response, 'memory_usage_mb', 0.0)
            
            # Normalize efficiency metrics
            speed_score = min(1.0, tokens_per_sec / 1000.0) if tokens_per_sec > 0 else 0.0
            memory_score = max(0.0, 1.0 - (memory_usage / 10000.0)) if memory_usage > 0 else 1.0
            
            return (speed_score + memory_score) / 2
            
        except Exception as e:
            bt.logging.warning(f"Error scoring efficiency: {e}")
            return 0.0
    
    def _apply_perturbation_testing(self, task: Dict[str, Any], response: Any, 
                                  miner_uid: int) -> Tuple[float, Dict[str, float]]:
        """Apply perturbation testing and evaluate consistency"""
        try:
            # Generate perturbation tests
            perturbation_tests = self.perturbation_tester.generate_perturbation_tests(task)
            
            if not perturbation_tests:
                bt.logging.warning("No perturbation tests generated")
                return 1.0, {}
            
            # For now, simulate perturbation responses (in real implementation, 
            # these would be obtained by querying the miner with perturbed tasks)
            perturbed_responses = [response] * len(perturbation_tests)  # Placeholder
            
            # Evaluate consistency
            consistency_scores = self.perturbation_tester.evaluate_consistency(
                response, perturbed_responses, perturbation_tests
            )
            
            # Calculate overall consistency score
            if consistency_scores:
                overall_consistency = sum(consistency_scores.values()) / len(consistency_scores)
            else:
                overall_consistency = 1.0
            
            return overall_consistency, consistency_scores
            
        except Exception as e:
            bt.logging.error(f"Error in perturbation testing: {e}")
            return 0.0, {}
    
    def _calculate_final_score(self, quality_score: float, consistency_score: float, 
                             efficiency_score: float) -> float:
        """Calculate final weighted score"""
        final_score = (
            quality_score * self.scoring_weights["task_quality"] +
            consistency_score * self.scoring_weights["consistency_under_perturbation"] +
            efficiency_score * self.scoring_weights["efficiency_proxy"]
        )
        
        return min(1.0, max(0.0, final_score))
    
    def get_audit_trail(self, miner_uid: int, task_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for a specific scoring decision"""
        return self.audit_logger.get_audit_trail(miner_uid, task_id)
    
    def cleanup_old_audit_data(self):
        """Clean up old audit data based on retention policy"""
        self.audit_logger.cleanup_old_entries()
        bt.logging.info("🧹 Audit data cleanup completed")
    
    def validate_consensus(self, miner_uid: int, task_id: str) -> Tuple[bool, float]:
        """Validate consensus for a scoring decision"""
        if not self.consensus_validation_enabled:
            return True, 1.0
        
        consensus_achieved, consensus_score, validator_scores = self.consensus_checker.check_consensus(
            miner_uid, task_id
        )
        
        if not consensus_achieved:
            bt.logging.warning(f"⚠️ Consensus not achieved for miner {miner_uid}, task {task_id}")
            bt.logging.warning(f"Validator scores: {validator_scores}")
        
        return consensus_achieved, consensus_score