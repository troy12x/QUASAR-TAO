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
import json
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from datetime import datetime, timedelta
import logging
import os

import bittensor as bt


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubnetMetrics:
    """Subnet-specific metrics"""
    active_miners: int
    active_validators: int
    evaluation_cycles_completed: int
    average_response_time: float
    benchmark_success_rate: float
    diversity_score: float
    monoculture_risk_level: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryCollector:
    """
    Comprehensive telemetry collection system for the unified HFA-SimpleMind subnet.
    
    Collects and aggregates:
    - System performance metrics (CPU, memory, disk, network, GPU)
    - Subnet-specific metrics (miners, validators, evaluation cycles)
    - Model performance metrics (inference time, accuracy, diversity)
    - Error rates and health indicators
    - Custom application metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize telemetry collector.
        
        Args:
            config: Configuration dictionary with telemetry settings
        """
        self.config = config or {}
        
        # Collection settings
        self.collection_interval = self.config.get("collection_interval", 30)  # seconds
        self.retention_hours = self.config.get("retention_hours", 24)
        self.max_points_per_metric = self.config.get("max_points_per_metric", 2880)  # 24h at 30s intervals
        
        # Storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_points_per_metric))
        self.performance_history: deque = deque(maxlen=self.max_points_per_metric)
        self.subnet_metrics_history: deque = deque(maxlen=self.max_points_per_metric)
        
        # Collection state
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        self.custom_collectors: List[Callable[[], Dict[str, float]]] = []
        
        # GPU monitoring
        self.gpu_available = self._check_gpu_availability()
        
        bt.logging.info("📊 TelemetryCollector initialized")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import GPUtil
            return len(GPUtil.getGPUs()) > 0
        except ImportError:
            bt.logging.warning("GPUtil not available, GPU monitoring disabled")
            return False
        except Exception as e:
            bt.logging.warning(f"GPU monitoring check failed: {e}")
            return False
    
    def start_collection(self):
        """Start automatic telemetry collection"""
        if self.is_collecting:
            bt.logging.warning("Telemetry collection already running")
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        bt.logging.info(f"📊 Started telemetry collection (interval: {self.collection_interval}s)")
    
    def stop_collection(self):
        """Stop automatic telemetry collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        bt.logging.info("📊 Stopped telemetry collection")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.is_collecting:
            try:
                # Collect system performance metrics
                perf_metrics = self._collect_performance_metrics()
                self.performance_history.append(perf_metrics)
                
                # Collect custom metrics
                for collector in self.custom_collectors:
                    try:
                        custom_metrics = collector()
                        for metric_name, value in custom_metrics.items():
                            self.record_metric(metric_name, value)
                    except Exception as e:
                        bt.logging.error(f"Error in custom collector: {e}")
                
                # Cleanup old data
                self._cleanup_old_data()
                
            except Exception as e:
                bt.logging.error(f"Error in telemetry collection: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect system performance metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network
            network = psutil.net_io_counters()
            
            # GPU metrics if available
            gpu_utilization = None
            gpu_memory_used = None
            
            if self.gpu_available:
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_utilization = gpu.load * 100
                        gpu_memory_used = gpu.memoryUsed
                except Exception as e:
                    bt.logging.debug(f"GPU metrics collection failed: {e}")
            
            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                gpu_utilization=gpu_utilization,
                gpu_memory_used=gpu_memory_used
            )
            
        except Exception as e:
            bt.logging.error(f"Error collecting performance metrics: {e}")
            return PerformanceMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_usage_percent=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0
            )
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a custom metric"""
        metric_point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        
        self.metrics[name].append(metric_point)
        
        bt.logging.debug(f"📊 Recorded metric {name}: {value}")
    
    def record_subnet_metrics(self, metrics: SubnetMetrics):
        """Record subnet-specific metrics"""
        self.subnet_metrics_history.append(metrics)
        
        # Also record individual metrics for easier querying
        self.record_metric("active_miners", metrics.active_miners)
        self.record_metric("active_validators", metrics.active_validators)
        self.record_metric("evaluation_cycles_completed", metrics.evaluation_cycles_completed)
        self.record_metric("average_response_time", metrics.average_response_time)
        self.record_metric("benchmark_success_rate", metrics.benchmark_success_rate)
        self.record_metric("diversity_score", metrics.diversity_score)
        
        bt.logging.debug(f"📊 Recorded subnet metrics: {metrics.active_miners} miners, {metrics.active_validators} validators")
    
    def add_custom_collector(self, collector: Callable[[], Dict[str, float]]):
        """Add a custom metric collector function"""
        self.custom_collectors.append(collector)
        bt.logging.info(f"📊 Added custom collector (total: {len(self.custom_collectors)})")
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricPoint]:
        """Get metric history for specified time period"""
        if name not in self.metrics:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        return [
            point for point in self.metrics[name]
            if point.timestamp > cutoff_time
        ]
    
    def get_performance_history(self, hours: int = 1) -> List[PerformanceMetrics]:
        """Get performance metrics history"""
        cutoff_time = time.time() - (hours * 3600)
        return [
            metrics for metrics in self.performance_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_subnet_metrics_history(self, hours: int = 1) -> List[SubnetMetrics]:
        """Get subnet metrics history"""
        cutoff_time = time.time() - (hours * 3600)
        return [
            metrics for metrics in self.subnet_metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_current_performance(self) -> Optional[PerformanceMetrics]:
        """Get most recent performance metrics"""
        return self.performance_history[-1] if self.performance_history else None
    
    def get_current_subnet_metrics(self) -> Optional[SubnetMetrics]:
        """Get most recent subnet metrics"""
        return self.subnet_metrics_history[-1] if self.subnet_metrics_history else None
    
    def get_metric_summary(self, name: str, hours: int = 1) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        history = self.get_metric_history(name, hours)
        
        if not history:
            return {"count": 0}
        
        values = [point.value for point in history]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1]
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format"""
        data = {
            "timestamp": time.time(),
            "performance_metrics": [m.to_dict() for m in self.performance_history],
            "subnet_metrics": [m.to_dict() for m in self.subnet_metrics_history],
            "custom_metrics": {
                name: [point.to_dict() for point in points]
                for name, points in self.metrics.items()
            }
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _cleanup_old_data(self):
        """Clean up old telemetry data"""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        # Clean up custom metrics
        for name, points in self.metrics.items():
            # deque with maxlen automatically handles size limits
            # but we can also clean by time if needed
            pass
        
        # Performance and subnet metrics are handled by deque maxlen
        
        bt.logging.debug("📊 Cleaned up old telemetry data")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        current_perf = self.get_current_performance()
        current_subnet = self.get_current_subnet_metrics()
        
        health_status = {
            "overall_health": "healthy",
            "timestamp": time.time(),
            "issues": []
        }
        
        if current_perf:
            # Check performance thresholds
            if current_perf.cpu_percent > 90:
                health_status["issues"].append("High CPU usage")
                health_status["overall_health"] = "warning"
            
            if current_perf.memory_percent > 90:
                health_status["issues"].append("High memory usage")
                health_status["overall_health"] = "warning"
            
            if current_perf.disk_usage_percent > 95:
                health_status["issues"].append("High disk usage")
                health_status["overall_health"] = "critical"
            
            if current_perf.gpu_utilization and current_perf.gpu_utilization > 95:
                health_status["issues"].append("High GPU utilization")
        
        if current_subnet:
            # Check subnet health
            if current_subnet.monoculture_risk_level == "high":
                health_status["issues"].append("High monoculture risk")
                health_status["overall_health"] = "warning"
            
            if current_subnet.benchmark_success_rate < 0.5:
                health_status["issues"].append("Low benchmark success rate")
                health_status["overall_health"] = "warning"
        
        # Set overall health based on issues
        if len(health_status["issues"]) > 3:
            health_status["overall_health"] = "critical"
        elif len(health_status["issues"]) > 0 and health_status["overall_health"] == "healthy":
            health_status["overall_health"] = "warning"
        
        return health_status


class StructuredLogger:
    """
    Structured logging system with context information and performance tracking.
    """
    
    def __init__(self, component_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize structured logger.
        
        Args:
            component_name: Name of the component (miner, validator, etc.)
            config: Configuration dictionary
        """
        self.component_name = component_name
        self.config = config or {}
        
        # Setup structured logging
        self.logger = logging.getLogger(f"hfa_subnet.{component_name}")
        
        # Context tracking
        self.context_stack: List[Dict[str, Any]] = []
        self.performance_tracking: Dict[str, List[float]] = defaultdict(list)
        
        # Log levels and filtering
        self.log_level = self.config.get("log_level", "INFO")
        self.enable_performance_logging = self.config.get("enable_performance_logging", True)
        
        bt.logging.info(f"📝 StructuredLogger initialized for {component_name}")
    
    def push_context(self, **context):
        """Push context information onto the stack"""
        self.context_stack.append(context)
    
    def pop_context(self):
        """Pop context information from the stack"""
        if self.context_stack:
            return self.context_stack.pop()
        return {}
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current context by merging all stack levels"""
        context = {"component": self.component_name}
        for ctx in self.context_stack:
            context.update(ctx)
        return context
    
    def log_structured(self, level: str, message: str, **extra_fields):
        """Log structured message with context"""
        context = self._get_current_context()
        context.update(extra_fields)
        context["timestamp"] = time.time()
        context["message"] = message
        
        # Format structured log entry
        log_entry = json.dumps(context, default=str)
        
        # Log using bittensor logger
        if level.upper() == "DEBUG":
            bt.logging.debug(log_entry)
        elif level.upper() == "INFO":
            bt.logging.info(log_entry)
        elif level.upper() == "WARNING":
            bt.logging.warning(log_entry)
        elif level.upper() == "ERROR":
            bt.logging.error(log_entry)
        elif level.upper() == "CRITICAL":
            bt.logging.critical(log_entry)
    
    def log_performance(self, operation: str, duration: float, **metadata):
        """Log performance metrics for an operation"""
        if not self.enable_performance_logging:
            return
        
        self.performance_tracking[operation].append(duration)
        
        # Keep only recent measurements
        if len(self.performance_tracking[operation]) > 1000:
            self.performance_tracking[operation] = self.performance_tracking[operation][-500:]
        
        self.log_structured(
            "INFO",
            f"Performance: {operation}",
            operation=operation,
            duration_ms=duration * 1000,
            **metadata
        )
    
    def log_error_with_context(self, error: Exception, operation: str, **context):
        """Log error with full context information"""
        self.log_structured(
            "ERROR",
            f"Error in {operation}: {str(error)}",
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            **context
        )
    
    def get_performance_stats(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation"""
        if operation not in self.performance_tracking:
            return {}
        
        durations = self.performance_tracking[operation]
        if not durations:
            return {}
        
        return {
            "count": len(durations),
            "avg_ms": (sum(durations) / len(durations)) * 1000,
            "min_ms": min(durations) * 1000,
            "max_ms": max(durations) * 1000,
            "latest_ms": durations[-1] * 1000
        }
    
    def context_manager(self, **context):
        """Context manager for automatic context push/pop"""
        return LoggingContext(self, context)


class LoggingContext:
    """Context manager for structured logging"""
    
    def __init__(self, logger: StructuredLogger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        self.logger.push_context(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.pop_context()
        
        if exc_type is not None:
            self.logger.log_error_with_context(
                exc_val, 
                self.context.get("operation", "unknown"),
                **self.context
            )


def create_telemetry_collector(config: Optional[Dict[str, Any]] = None) -> TelemetryCollector:
    """Factory function to create telemetry collector"""
    return TelemetryCollector(config)


def create_structured_logger(component_name: str, config: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """Factory function to create structured logger"""
    return StructuredLogger(component_name, config)