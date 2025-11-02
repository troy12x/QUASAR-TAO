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
import numpy as np
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

import bittensor as bt


@dataclass
class DiversityMetrics:
    """Comprehensive diversity tracking metrics"""
    cosine_similarity_to_baseline: float
    response_uniqueness_score: float
    model_architecture_diversity: float
    behavioral_diversity_score: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cosine_similarity_to_baseline": self.cosine_similarity_to_baseline,
            "response_uniqueness_score": self.response_uniqueness_score,
            "model_architecture_diversity": self.model_architecture_diversity,
            "behavioral_diversity_score": self.behavioral_diversity_score,
            "timestamp": self.timestamp
        }


@dataclass
class MinerProfile:
    """Profile tracking for individual miners"""
    miner_uid: int
    model_id: Optional[str] = None
    architecture_type: Optional[str] = None
    response_patterns: List[str] = field(default_factory=list)
    behavior_signature: Optional[str] = None
    diversity_history: List[DiversityMetrics] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    
    def update_behavior_signature(self, responses: List[str]):
        """Update behavioral signature based on recent responses"""
        if not responses:
            return
            
        # Create signature from response patterns
        combined_text = " ".join(responses[-10:])  # Use last 10 responses
        signature_hash = hashlib.sha256(combined_text.encode()).hexdigest()[:16]
        self.behavior_signature = signature_hash
        self.last_updated = time.time()


class DiversityTracker:
    """
    Tracks and incentivizes diversity in miner responses and model architectures.
    
    This class implements diversity monitoring to prevent monoculture and encourage
    innovation in the unified HFA-SimpleMind subnet. It tracks:
    
    - Response diversity: Cosine similarity against baseline models
    - Model architecture diversity: Distribution of different architectures
    - Behavioral diversity: Unique response patterns and strategies
    - Temporal diversity: Changes in behavior over time
    
    The tracker integrates with the existing scoring system to provide diversity
    incentives and penalties for identical or near-identical responses.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize diversity tracker with configuration.
        
        Args:
            config: Configuration dictionary with diversity settings
        """
        self.config = config or {}
        
        # Diversity tracking settings
        self.similarity_threshold = self.config.get("similarity_threshold", 0.85)
        self.diversity_penalty_factor = self.config.get("diversity_penalty_factor", 0.2)
        self.diversity_bonus_factor = self.config.get("diversity_bonus_factor", 0.1)
        self.baseline_update_frequency = self.config.get("baseline_update_frequency", 100)
        self.history_window_size = self.config.get("history_window_size", 1000)
        
        # Tracking data structures
        self.miner_profiles: Dict[int, MinerProfile] = {}
        self.baseline_responses: deque = deque(maxlen=self.history_window_size)
        self.architecture_distribution: Dict[str, int] = defaultdict(int)
        self.response_history: deque = deque(maxlen=self.history_window_size)
        self.diversity_stats: Dict[str, Any] = {}
        
        # Baseline model tracking
        self.baseline_embeddings: Optional[np.ndarray] = None
        self.baseline_update_counter = 0
        
        bt.logging.info("🎯 DiversityTracker initialized for monoculture prevention")
    
    def track_miner_response(self, miner_uid: int, response: Any, 
                           model_info: Optional[Dict[str, Any]] = None,
                           task_type: Optional[str] = None) -> DiversityMetrics:
        """
        Track a miner's response and compute diversity metrics.
        
        Args:
            miner_uid: Unique identifier for the miner
            response: The miner's response to evaluate
            model_info: Optional model architecture information
            task_type: Type of task being evaluated
            
        Returns:
            DiversityMetrics with computed diversity scores
        """
        # Ensure miner profile exists
        if miner_uid not in self.miner_profiles:
            self.miner_profiles[miner_uid] = MinerProfile(miner_uid=miner_uid)
        
        miner_profile = self.miner_profiles[miner_uid]
        
        # Extract response text
        response_text = self._extract_response_text(response)
        
        # Update miner profile
        if model_info:
            miner_profile.model_id = model_info.get("model_id")
            miner_profile.architecture_type = model_info.get("architecture_type")
            
            # Update architecture distribution
            arch_type = model_info.get("architecture_type", "unknown")
            self.architecture_distribution[arch_type] += 1
        
        # Add response to history
        miner_profile.response_patterns.append(response_text)
        if len(miner_profile.response_patterns) > 50:  # Keep last 50 responses
            miner_profile.response_patterns.pop(0)
        
        # Update behavior signature
        miner_profile.update_behavior_signature(miner_profile.response_patterns)
        
        # Compute diversity metrics
        diversity_metrics = self._compute_diversity_metrics(
            miner_uid, response_text, miner_profile, task_type
        )
        
        # Store metrics in profile
        miner_profile.diversity_history.append(diversity_metrics)
        if len(miner_profile.diversity_history) > 100:  # Keep last 100 evaluations
            miner_profile.diversity_history.pop(0)
        
        # Update global tracking
        self.response_history.append({
            "miner_uid": miner_uid,
            "response": response_text,
            "timestamp": time.time(),
            "task_type": task_type
        })
        
        # Update baseline if needed
        self._update_baseline_if_needed()
        
        bt.logging.debug(f"🎯 Diversity metrics for miner {miner_uid}: "
                        f"uniqueness={diversity_metrics.response_uniqueness_score:.3f}, "
                        f"baseline_sim={diversity_metrics.cosine_similarity_to_baseline:.3f}")
        
        return diversity_metrics
    
    def _compute_diversity_metrics(self, miner_uid: int, response_text: str, 
                                 miner_profile: MinerProfile, 
                                 task_type: Optional[str] = None) -> DiversityMetrics:
        """Compute comprehensive diversity metrics for a response"""
        
        # 1. Cosine similarity to baseline
        baseline_similarity = self._compute_baseline_similarity(response_text)
        
        # 2. Response uniqueness score
        uniqueness_score = self._compute_response_uniqueness(response_text, miner_uid)
        
        # 3. Model architecture diversity
        arch_diversity = self._compute_architecture_diversity(miner_profile.architecture_type)
        
        # 4. Behavioral diversity score
        behavioral_diversity = self._compute_behavioral_diversity(miner_profile)
        
        return DiversityMetrics(
            cosine_similarity_to_baseline=baseline_similarity,
            response_uniqueness_score=uniqueness_score,
            model_architecture_diversity=arch_diversity,
            behavioral_diversity_score=behavioral_diversity
        )
    
    def _compute_baseline_similarity(self, response_text: str) -> float:
        """Compute cosine similarity to baseline responses"""
        try:
            if not self.baseline_responses:
                return 0.0  # No baseline yet, assume diverse
            
            # Simple token-based similarity (in production, use embeddings)
            response_tokens = set(response_text.lower().split())
            
            similarities = []
            for baseline_response in list(self.baseline_responses)[-10:]:  # Compare to last 10 baseline responses
                baseline_tokens = set(baseline_response.lower().split())
                
                if not response_tokens and not baseline_tokens:
                    similarity = 1.0
                elif not response_tokens or not baseline_tokens:
                    similarity = 0.0
                else:
                    intersection = response_tokens.intersection(baseline_tokens)
                    union = response_tokens.union(baseline_tokens)
                    similarity = len(intersection) / len(union) if union else 0.0
                
                similarities.append(similarity)
            
            return max(similarities) if similarities else 0.0
            
        except Exception as e:
            bt.logging.warning(f"Error computing baseline similarity: {e}")
            return 0.0
    
    def _compute_response_uniqueness(self, response_text: str, miner_uid: int) -> float:
        """Compute uniqueness of response compared to recent responses from all miners"""
        try:
            if len(self.response_history) < 2:
                return 1.0  # Assume unique if not enough history
            
            response_tokens = set(response_text.lower().split())
            similarities = []
            
            # Compare to recent responses from other miners
            for entry in list(self.response_history)[-20:]:  # Last 20 responses
                if entry["miner_uid"] == miner_uid:
                    continue  # Skip own responses
                
                other_tokens = set(entry["response"].lower().split())
                
                if not response_tokens and not other_tokens:
                    similarity = 1.0
                elif not response_tokens or not other_tokens:
                    similarity = 0.0
                else:
                    intersection = response_tokens.intersection(other_tokens)
                    union = response_tokens.union(other_tokens)
                    similarity = len(intersection) / len(union) if union else 0.0
                
                similarities.append(similarity)
            
            if not similarities:
                return 1.0
            
            # Uniqueness is inverse of maximum similarity
            max_similarity = max(similarities)
            return 1.0 - max_similarity
            
        except Exception as e:
            bt.logging.warning(f"Error computing response uniqueness: {e}")
            return 0.5  # Default to moderate uniqueness
    
    def _compute_architecture_diversity(self, architecture_type: Optional[str]) -> float:
        """Compute diversity bonus based on architecture distribution"""
        try:
            if not architecture_type or not self.architecture_distribution:
                return 0.5  # Default diversity score
            
            total_responses = sum(self.architecture_distribution.values())
            if total_responses == 0:
                return 1.0
            
            # Calculate inverse frequency - rarer architectures get higher scores
            arch_frequency = self.architecture_distribution[architecture_type] / total_responses
            diversity_score = 1.0 - arch_frequency
            
            # Normalize to reasonable range
            return max(0.1, min(1.0, diversity_score))
            
        except Exception as e:
            bt.logging.warning(f"Error computing architecture diversity: {e}")
            return 0.5
    
    def _compute_behavioral_diversity(self, miner_profile: MinerProfile) -> float:
        """Compute behavioral diversity based on response patterns"""
        try:
            if len(miner_profile.response_patterns) < 3:
                return 1.0  # Assume diverse if not enough history
            
            # Analyze response pattern consistency
            recent_responses = miner_profile.response_patterns[-10:]
            
            # Compute pairwise similarities
            similarities = []
            for i in range(len(recent_responses)):
                for j in range(i + 1, len(recent_responses)):
                    tokens_i = set(recent_responses[i].lower().split())
                    tokens_j = set(recent_responses[j].lower().split())
                    
                    if not tokens_i and not tokens_j:
                        similarity = 1.0
                    elif not tokens_i or not tokens_j:
                        similarity = 0.0
                    else:
                        intersection = tokens_i.intersection(tokens_j)
                        union = tokens_i.union(tokens_j)
                        similarity = len(intersection) / len(union) if union else 0.0
                    
                    similarities.append(similarity)
            
            if not similarities:
                return 1.0
            
            # Behavioral diversity is inverse of average self-similarity
            avg_similarity = sum(similarities) / len(similarities)
            return 1.0 - avg_similarity
            
        except Exception as e:
            bt.logging.warning(f"Error computing behavioral diversity: {e}")
            return 0.5
    
    def _update_baseline_if_needed(self):
        """Update baseline responses periodically"""
        self.baseline_update_counter += 1
        
        if self.baseline_update_counter >= self.baseline_update_frequency:
            self._update_baseline()
            self.baseline_update_counter = 0
    
    def _update_baseline(self):
        """Update baseline responses from recent high-quality responses"""
        try:
            if len(self.response_history) < 10:
                return
            
            # Select recent responses as baseline (in production, filter by quality)
            recent_responses = [entry["response"] for entry in list(self.response_history)[-50:]]
            
            # Update baseline responses
            self.baseline_responses.extend(recent_responses)
            
            bt.logging.info(f"🎯 Updated diversity baseline with {len(recent_responses)} responses")
            
        except Exception as e:
            bt.logging.error(f"Error updating baseline: {e}")
    
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
    
    def compute_diversity_incentive(self, miner_uid: int, base_score: float) -> float:
        """
        Compute diversity incentive/penalty to apply to base score.
        
        Args:
            miner_uid: Unique identifier for the miner
            base_score: Base quality score before diversity adjustment
            
        Returns:
            Adjusted score with diversity incentives applied
        """
        if miner_uid not in self.miner_profiles:
            return base_score  # No diversity data, return base score
        
        miner_profile = self.miner_profiles[miner_uid]
        
        if not miner_profile.diversity_history:
            return base_score  # No diversity history, return base score
        
        # Get latest diversity metrics
        latest_metrics = miner_profile.diversity_history[-1]
        
        # Calculate diversity incentive
        diversity_factors = [
            latest_metrics.response_uniqueness_score,
            latest_metrics.model_architecture_diversity,
            latest_metrics.behavioral_diversity_score
        ]
        
        avg_diversity = sum(diversity_factors) / len(diversity_factors)
        
        # Apply incentive/penalty
        if avg_diversity > 0.7:  # High diversity - bonus
            incentive = base_score * self.diversity_bonus_factor
            adjusted_score = base_score + incentive
            bt.logging.debug(f"🎯 Diversity bonus for miner {miner_uid}: +{incentive:.4f}")
        elif avg_diversity < 0.3:  # Low diversity - penalty
            penalty = base_score * self.diversity_penalty_factor
            adjusted_score = base_score - penalty
            bt.logging.debug(f"🎯 Diversity penalty for miner {miner_uid}: -{penalty:.4f}")
        else:
            adjusted_score = base_score  # Neutral diversity
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, adjusted_score))
    
    def detect_monoculture_risk(self) -> Dict[str, Any]:
        """Detect potential monoculture risks in the network"""
        risks = {
            "architecture_concentration": False,
            "response_similarity": False,
            "behavioral_homogeneity": False,
            "risk_level": "low",
            "recommendations": []
        }
        
        try:
            # Check architecture concentration
            if self.architecture_distribution:
                total_responses = sum(self.architecture_distribution.values())
                max_arch_share = max(self.architecture_distribution.values()) / total_responses
                
                if max_arch_share > 0.8:  # 80% concentration in one architecture
                    risks["architecture_concentration"] = True
                    risks["recommendations"].append("Incentivize alternative architectures")
            
            # Check response similarity
            if len(self.response_history) >= 10:
                recent_responses = [entry["response"] for entry in list(self.response_history)[-10:]]
                similarities = []
                
                for i in range(len(recent_responses)):
                    for j in range(i + 1, len(recent_responses)):
                        tokens_i = set(recent_responses[i].lower().split())
                        tokens_j = set(recent_responses[j].lower().split())
                        
                        if tokens_i and tokens_j:
                            intersection = tokens_i.intersection(tokens_j)
                            union = tokens_i.union(tokens_j)
                            similarity = len(intersection) / len(union) if union else 0.0
                            similarities.append(similarity)
                
                if similarities and sum(similarities) / len(similarities) > 0.8:
                    risks["response_similarity"] = True
                    risks["recommendations"].append("Increase task diversity")
            
            # Determine overall risk level
            risk_count = sum([
                risks["architecture_concentration"],
                risks["response_similarity"],
                risks["behavioral_homogeneity"]
            ])
            
            if risk_count >= 2:
                risks["risk_level"] = "high"
            elif risk_count == 1:
                risks["risk_level"] = "medium"
            
            if risks["risk_level"] != "low":
                bt.logging.warning(f"🚨 Monoculture risk detected: {risks['risk_level']} level")
            
        except Exception as e:
            bt.logging.error(f"Error detecting monoculture risk: {e}")
        
        return risks
    
    def get_diversity_stats(self) -> Dict[str, Any]:
        """Get comprehensive diversity statistics"""
        stats = {
            "total_miners": len(self.miner_profiles),
            "architecture_distribution": dict(self.architecture_distribution),
            "response_history_size": len(self.response_history),
            "baseline_responses_size": len(self.baseline_responses),
            "monoculture_risk": self.detect_monoculture_risk(),
            "diversity_trends": self._compute_diversity_trends()
        }
        
        return stats
    
    def _compute_diversity_trends(self) -> Dict[str, float]:
        """Compute diversity trends over time"""
        trends = {
            "avg_uniqueness": 0.0,
            "avg_architecture_diversity": 0.0,
            "avg_behavioral_diversity": 0.0
        }
        
        try:
            all_metrics = []
            for profile in self.miner_profiles.values():
                all_metrics.extend(profile.diversity_history[-10:])  # Last 10 evaluations per miner
            
            if all_metrics:
                trends["avg_uniqueness"] = sum(m.response_uniqueness_score for m in all_metrics) / len(all_metrics)
                trends["avg_architecture_diversity"] = sum(m.model_architecture_diversity for m in all_metrics) / len(all_metrics)
                trends["avg_behavioral_diversity"] = sum(m.behavioral_diversity_score for m in all_metrics) / len(all_metrics)
        
        except Exception as e:
            bt.logging.warning(f"Error computing diversity trends: {e}")
        
        return trends
    
    def cleanup_old_data(self, retention_days: int = 7):
        """Clean up old diversity tracking data"""
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        # Clean up miner profiles
        for miner_uid, profile in self.miner_profiles.items():
            profile.diversity_history = [
                metrics for metrics in profile.diversity_history
                if metrics.timestamp > cutoff_time
            ]
            
            # Remove profiles with no recent activity
            if profile.last_updated < cutoff_time:
                bt.logging.info(f"🧹 Removing inactive miner profile: {miner_uid}")
        
        # Remove inactive profiles
        inactive_miners = [
            uid for uid, profile in self.miner_profiles.items()
            if profile.last_updated < cutoff_time
        ]
        
        for uid in inactive_miners:
            del self.miner_profiles[uid]
        
        bt.logging.info(f"🧹 Diversity tracker cleanup completed, removed {len(inactive_miners)} inactive profiles")