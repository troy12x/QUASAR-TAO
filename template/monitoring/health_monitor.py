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
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

import bittensor as bt


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check configuration"""
    name: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: int = 60
    timeout_seconds: int = 30
    enabled: bool = True
    last_run: Optional[float] = None
    last_result: Optional[Dict[str, Any]] = None
    consecutive_failures: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "interval_seconds": self.interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "last_result": self.last_result,
            "consecutive_failures": self.consecutive_failures
        }


@dataclass
class Alert:
    """System alert"""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    component: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "component": self.component,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolved_timestamp": self.resolved_timestamp,
            "metadata": self.metadata
        }


class HealthMonitor:
    """
    Comprehensive health monitoring system for the unified HFA-SimpleMind subnet.
    
    Monitors:
    - System resource health (CPU, memory, disk, GPU)
    - Subnet component health (miners, validators, protocol)
    - Model performance health (inference times, accuracy)
    - Network connectivity and communication health
    - Data integrity and consistency health
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize health monitor.
        
        Args:
            config: Configuration dictionary with monitoring settings
        """
        self.config = config or {}
        
        # Monitoring settings
        self.check_interval = self.config.get("check_interval", 30)  # seconds
        self.alert_retention_hours = self.config.get("alert_retention_hours", 24)
        self.max_consecutive_failures = self.config.get("max_consecutive_failures", 3)
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # State tracking
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Component health status
        self.component_health: Dict[str, HealthStatus] = {}
        self.overall_health: HealthStatus = HealthStatus.UNKNOWN
        
        # Initialize default health checks
        self._register_default_health_checks()
        
        bt.logging.info("🏥 HealthMonitor initialized")
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        
        # System resource checks
        self.register_health_check(
            "system_resources",
            self._check_system_resources,
            interval_seconds=30
        )
        
        # Memory usage check
        self.register_health_check(
            "memory_usage",
            self._check_memory_usage,
            interval_seconds=60
        )
        
        # Disk space check
        self.register_health_check(
            "disk_space",
            self._check_disk_space,
            interval_seconds=300  # 5 minutes
        )
        
        # GPU health check (if available)
        self.register_health_check(
            "gpu_health",
            self._check_gpu_health,
            interval_seconds=60
        )
    
    def register_health_check(self, name: str, check_function: Callable[[], Dict[str, Any]], 
                            interval_seconds: int = 60, timeout_seconds: int = 30):
        """Register a new health check"""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds
        )
        
        self.health_checks[name] = health_check
        bt.logging.info(f"🏥 Registered health check: {name}")
    
    def register_alert_handler(self, handler: Callable[[Alert], None]):
        """Register an alert handler function"""
        self.alert_handlers.append(handler)
        bt.logging.info(f"🚨 Registered alert handler (total: {len(self.alert_handlers)})")
    
    def start_monitoring(self):
        """Start health monitoring"""
        if self.is_monitoring:
            bt.logging.warning("Health monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        bt.logging.info(f"🏥 Started health monitoring (interval: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        bt.logging.info("🏥 Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Run health checks that are due
                for name, health_check in self.health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    # Check if it's time to run this health check
                    if (health_check.last_run is None or 
                        current_time - health_check.last_run >= health_check.interval_seconds):
                        
                        self._run_health_check(health_check)
                
                # Update overall health status
                self._update_overall_health()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
            except Exception as e:
                bt.logging.error(f"Error in health monitoring loop: {e}")
            
            time.sleep(self.check_interval)
    
    def _run_health_check(self, health_check: HealthCheck):
        """Run a single health check"""
        try:
            start_time = time.time()
            
            # Run the health check with timeout
            result = self._run_with_timeout(
                health_check.check_function,
                health_check.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            # Update health check state
            health_check.last_run = start_time
            health_check.last_result = result
            
            # Check if the health check passed
            if result.get("status") == "healthy":
                health_check.consecutive_failures = 0
                self._resolve_alerts_for_check(health_check.name)
            else:
                health_check.consecutive_failures += 1
                
                # Create alert if threshold exceeded
                if health_check.consecutive_failures >= self.max_consecutive_failures:
                    self._create_alert_for_check(health_check, result)
            
            bt.logging.debug(f"🏥 Health check {health_check.name}: {result.get('status')} ({duration:.2f}s)")
            
        except Exception as e:
            health_check.consecutive_failures += 1
            health_check.last_result = {
                "status": "critical",
                "error": str(e),
                "timestamp": time.time()
            }
            
            bt.logging.error(f"Health check {health_check.name} failed: {e}")
            
            # Create alert for check failure
            if health_check.consecutive_failures >= self.max_consecutive_failures:
                self._create_alert_for_check(health_check, health_check.last_result)
    
    def _run_with_timeout(self, func: Callable, timeout_seconds: int) -> Dict[str, Any]:
        """Run function with timeout"""
        try:
            # Simple timeout implementation
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Health check timed out after {timeout_seconds}s")
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func()
                signal.alarm(0)  # Cancel timeout
                return result
            except TimeoutError:
                raise
            finally:
                signal.alarm(0)  # Ensure timeout is cancelled
                
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _create_alert_for_check(self, health_check: HealthCheck, result: Dict[str, Any]):
        """Create alert for failed health check"""
        alert_id = f"health_check_{health_check.name}_{int(time.time())}"
        
        # Determine severity based on result
        if result.get("status") == "critical":
            severity = AlertSeverity.CRITICAL
        elif result.get("status") == "warning":
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.WARNING
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=f"Health Check Failed: {health_check.name}",
            description=result.get("error", f"Health check {health_check.name} failed"),
            component=health_check.name,
            metadata={
                "consecutive_failures": health_check.consecutive_failures,
                "check_result": result
            }
        )
        
        self._trigger_alert(alert)
    
    def _resolve_alerts_for_check(self, check_name: str):
        """Resolve alerts for a specific health check"""
        alerts_to_resolve = [
            alert for alert in self.active_alerts.values()
            if alert.component == check_name and not alert.resolved
        ]
        
        for alert in alerts_to_resolve:
            self._resolve_alert(alert.id)
    
    def _trigger_alert(self, alert: Alert):
        """Trigger a new alert"""
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        bt.logging.warning(f"🚨 Alert triggered: {alert.title} ({alert.severity.value})")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                bt.logging.error(f"Error in alert handler: {e}")
    
    def _resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_timestamp = time.time()
            
            bt.logging.info(f"✅ Alert resolved: {alert.title}")
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
    
    def _update_overall_health(self):
        """Update overall system health status"""
        if not self.health_checks:
            self.overall_health = HealthStatus.UNKNOWN
            return
        
        # Count health check statuses
        status_counts = {
            "healthy": 0,
            "warning": 0,
            "critical": 0,
            "unknown": 0
        }
        
        for health_check in self.health_checks.values():
            if not health_check.enabled or health_check.last_result is None:
                status_counts["unknown"] += 1
            else:
                status = health_check.last_result.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall health
        if status_counts["critical"] > 0:
            self.overall_health = HealthStatus.CRITICAL
        elif status_counts["warning"] > 0:
            self.overall_health = HealthStatus.WARNING
        elif status_counts["healthy"] > 0:
            self.overall_health = HealthStatus.HEALTHY
        else:
            self.overall_health = HealthStatus.UNKNOWN
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        cutoff_time = time.time() - (self.alert_retention_hours * 3600)
        
        # Remove old alerts from history
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        bt.logging.debug(f"🧹 Cleaned up old alerts, {len(self.alert_history)} remaining")
    
    # Default health check implementations
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            issues = []
            status = "healthy"
            
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                status = "critical"
            elif cpu_percent > 80:
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
                status = "warning"
            
            if memory.percent > 95:
                issues.append(f"Critical memory usage: {memory.percent:.1f}%")
                status = "critical"
            elif memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                status = "warning"
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "issues": issues,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check detailed memory usage"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            status = "healthy"
            issues = []
            
            if memory.percent > 95:
                status = "critical"
                issues.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent > 85:
                status = "warning"
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if swap.percent > 50:
                status = "warning"
                issues.append(f"High swap usage: {swap.percent:.1f}%")
            
            return {
                "status": status,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "swap_percent": swap.percent,
                "issues": issues,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage"""
        try:
            import psutil
            
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            issues = []
            
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent > 95:
                status = "critical"
                issues.append(f"Critical disk usage: {disk_percent:.1f}%")
            elif disk_percent > 85:
                status = "warning"
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            return {
                "status": status,
                "disk_percent": disk_percent,
                "disk_free_gb": disk.free / (1024**3),
                "issues": issues,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU health and utilization"""
        try:
            import GPUtil
            
            gpus = GPUtil.getGPUs()
            
            if not gpus:
                return {
                    "status": "healthy",
                    "message": "No GPUs detected",
                    "timestamp": time.time()
                }
            
            status = "healthy"
            issues = []
            gpu_info = []
            
            for gpu in gpus:
                gpu_data = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature
                }
                gpu_info.append(gpu_data)
                
                # Check GPU health
                if gpu.temperature > 85:
                    status = "critical"
                    issues.append(f"GPU {gpu.id} overheating: {gpu.temperature}°C")
                elif gpu.temperature > 80:
                    status = "warning"
                    issues.append(f"GPU {gpu.id} running hot: {gpu.temperature}°C")
                
                memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                if memory_percent > 95:
                    status = "warning"
                    issues.append(f"GPU {gpu.id} memory usage high: {memory_percent:.1f}%")
            
            return {
                "status": status,
                "gpu_count": len(gpus),
                "gpu_info": gpu_info,
                "issues": issues,
                "timestamp": time.time()
            }
            
        except ImportError:
            return {
                "status": "healthy",
                "message": "GPUtil not available, GPU monitoring disabled",
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "warning",
                "error": str(e),
                "timestamp": time.time()
            }
    
    # Public API methods
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            "overall_health": self.overall_health.value,
            "timestamp": time.time(),
            "active_alerts": len(self.active_alerts),
            "health_checks": {
                name: {
                    "status": check.last_result.get("status", "unknown") if check.last_result else "unknown",
                    "last_run": check.last_run,
                    "consecutive_failures": check.consecutive_failures
                }
                for name, check in self.health_checks.items()
            }
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        return [
            alert.to_dict() for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
    
    def force_health_check(self, check_name: str) -> Dict[str, Any]:
        """Force run a specific health check"""
        if check_name not in self.health_checks:
            return {"error": f"Health check {check_name} not found"}
        
        health_check = self.health_checks[check_name]
        self._run_health_check(health_check)
        
        return health_check.last_result or {"error": "No result available"}
    
    def enable_health_check(self, check_name: str):
        """Enable a health check"""
        if check_name in self.health_checks:
            self.health_checks[check_name].enabled = True
            bt.logging.info(f"🏥 Enabled health check: {check_name}")
    
    def disable_health_check(self, check_name: str):
        """Disable a health check"""
        if check_name in self.health_checks:
            self.health_checks[check_name].enabled = False
            bt.logging.info(f"🏥 Disabled health check: {check_name}")


def create_health_monitor(config: Optional[Dict[str, Any]] = None) -> HealthMonitor:
    """Factory function to create health monitor"""
    return HealthMonitor(config)