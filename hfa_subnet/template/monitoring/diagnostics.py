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
import traceback
import inspect
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import sys
import os

import bittensor as bt


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check"""
    name: str
    status: str  # "pass", "fail", "warning", "info"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms
        }


@dataclass
class DiagnosticCheck:
    """Diagnostic check configuration"""
    name: str
    description: str
    check_function: Callable[[], DiagnosticResult]
    category: str = "general"
    enabled: bool = True
    timeout_seconds: int = 30
    
    def run(self) -> DiagnosticResult:
        """Run the diagnostic check"""
        start_time = time.time()
        
        try:
            result = self.check_function()
            result.duration_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                name=self.name,
                status="fail",
                message=f"Diagnostic check failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                duration_ms=duration_ms
            )


class DiagnosticSystem:
    """
    Comprehensive diagnostic system for the unified HFA-SimpleMind subnet.
    
    Provides automated diagnostics for:
    - System configuration and dependencies
    - Model loading and inference capabilities
    - Network connectivity and communication
    - Data integrity and consistency
    - Performance benchmarks and bottlenecks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize diagnostic system.
        
        Args:
            config: Configuration dictionary with diagnostic settings
        """
        self.config = config or {}
        
        # Diagnostic checks registry
        self.checks: Dict[str, DiagnosticCheck] = {}
        self.categories: Dict[str, List[str]] = {}
        
        # Results tracking
        self.last_run_results: Dict[str, DiagnosticResult] = {}
        self.run_history: List[Dict[str, Any]] = []
        
        # Register default diagnostic checks
        self._register_default_checks()
        
        bt.logging.info("🔍 DiagnosticSystem initialized")
    
    def _register_default_checks(self):
        """Register default diagnostic checks"""
        
        # System checks
        self.register_check(
            "python_version",
            "Check Python version compatibility",
            self._check_python_version,
            "system"
        )
        
        self.register_check(
            "dependencies",
            "Check required dependencies",
            self._check_dependencies,
            "system"
        )
        
        self.register_check(
            "gpu_availability",
            "Check GPU availability and drivers",
            self._check_gpu_availability,
            "system"
        )
        
        # Configuration checks
        self.register_check(
            "config_files",
            "Check configuration file validity",
            self._check_config_files,
            "configuration"
        )
        
        self.register_check(
            "model_paths",
            "Check model file paths and accessibility",
            self._check_model_paths,
            "configuration"
        )
        
        # Network checks
        self.register_check(
            "bittensor_connection",
            "Check Bittensor network connectivity",
            self._check_bittensor_connection,
            "network"
        )
        
        self.register_check(
            "subnet_registration",
            "Check subnet registration status",
            self._check_subnet_registration,
            "network"
        )
        
        # Model checks
        self.register_check(
            "model_loading",
            "Test model loading capabilities",
            self._check_model_loading,
            "models"
        )
        
        self.register_check(
            "inference_test",
            "Test basic inference functionality",
            self._check_inference_test,
            "models"
        )
        
        # Performance checks
        self.register_check(
            "memory_usage",
            "Check memory usage patterns",
            self._check_memory_usage,
            "performance"
        )
        
        self.register_check(
            "disk_space",
            "Check available disk space",
            self._check_disk_space,
            "performance"
        )
    
    def register_check(self, name: str, description: str, 
                      check_function: Callable[[], DiagnosticResult],
                      category: str = "general", timeout_seconds: int = 30):
        """Register a new diagnostic check"""
        check = DiagnosticCheck(
            name=name,
            description=description,
            check_function=check_function,
            category=category,
            timeout_seconds=timeout_seconds
        )
        
        self.checks[name] = check
        
        # Add to category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)
        
        bt.logging.info(f"🔍 Registered diagnostic check: {name} ({category})")
    
    def run_check(self, check_name: str) -> DiagnosticResult:
        """Run a specific diagnostic check"""
        if check_name not in self.checks:
            return DiagnosticResult(
                name=check_name,
                status="fail",
                message=f"Diagnostic check '{check_name}' not found"
            )
        
        check = self.checks[check_name]
        if not check.enabled:
            return DiagnosticResult(
                name=check_name,
                status="info",
                message="Diagnostic check is disabled"
            )
        
        bt.logging.info(f"🔍 Running diagnostic check: {check_name}")
        
        result = check.run()
        self.last_run_results[check_name] = result
        
        return result
    
    def run_category(self, category: str) -> Dict[str, DiagnosticResult]:
        """Run all checks in a specific category"""
        if category not in self.categories:
            return {}
        
        results = {}
        for check_name in self.categories[category]:
            results[check_name] = self.run_check(check_name)
        
        return results
    
    def run_all_checks(self) -> Dict[str, DiagnosticResult]:
        """Run all diagnostic checks"""
        bt.logging.info("🔍 Running all diagnostic checks")
        
        start_time = time.time()
        results = {}
        
        for check_name in self.checks.keys():
            results[check_name] = self.run_check(check_name)
        
        duration = time.time() - start_time
        
        # Store run history
        run_summary = {
            "timestamp": start_time,
            "duration_seconds": duration,
            "total_checks": len(results),
            "passed": len([r for r in results.values() if r.status == "pass"]),
            "failed": len([r for r in results.values() if r.status == "fail"]),
            "warnings": len([r for r in results.values() if r.status == "warning"]),
            "results": {name: result.to_dict() for name, result in results.items()}
        }
        
        self.run_history.append(run_summary)
        
        # Keep only recent history
        if len(self.run_history) > 100:
            self.run_history = self.run_history[-50:]
        
        bt.logging.info(f"🔍 Diagnostic run completed in {duration:.2f}s: "
                       f"{run_summary['passed']} passed, {run_summary['failed']} failed, "
                       f"{run_summary['warnings']} warnings")
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        import platform
        import psutil
        
        system_info = {
            "timestamp": time.time(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation()
            },
            "resources": {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_total_gb": psutil.disk_usage('/').total / (1024**3)
            },
            "environment": {
                "python_path": sys.executable,
                "working_directory": os.getcwd(),
                "environment_variables": {
                    key: value for key, value in os.environ.items()
                    if key.startswith(('CUDA', 'TORCH', 'HF_', 'TRANSFORMERS_'))
                }
            }
        }
        
        # Add GPU information if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            system_info["gpu"] = {
                "count": len(gpus),
                "devices": [
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "driver_version": gpu.driver
                    }
                    for gpu in gpus
                ]
            }
        except ImportError:
            system_info["gpu"] = {"available": False, "reason": "GPUtil not installed"}
        except Exception as e:
            system_info["gpu"] = {"available": False, "reason": str(e)}
        
        return system_info
    
    def generate_report(self, format: str = "json") -> str:
        """Generate diagnostic report"""
        # Run all checks if no recent results
        if not self.last_run_results:
            self.run_all_checks()
        
        report_data = {
            "timestamp": time.time(),
            "system_info": self.get_system_info(),
            "diagnostic_results": {
                name: result.to_dict() 
                for name, result in self.last_run_results.items()
            },
            "summary": {
                "total_checks": len(self.last_run_results),
                "passed": len([r for r in self.last_run_results.values() if r.status == "pass"]),
                "failed": len([r for r in self.last_run_results.values() if r.status == "fail"]),
                "warnings": len([r for r in self.last_run_results.values() if r.status == "warning"]),
                "categories": list(self.categories.keys())
            }
        }
        
        if format == "json":
            return json.dumps(report_data, indent=2, default=str)
        elif format == "text":
            return self._format_text_report(report_data)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _format_text_report(self, report_data: Dict[str, Any]) -> str:
        """Format diagnostic report as text"""
        lines = []
        lines.append("=" * 80)
        lines.append("UNIFIED HFA-SIMPLEMIND SUBNET DIAGNOSTIC REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.fromtimestamp(report_data['timestamp']).isoformat()}")
        lines.append("")
        
        # Summary
        summary = report_data["summary"]
        lines.append("SUMMARY:")
        lines.append(f"  Total Checks: {summary['total_checks']}")
        lines.append(f"  Passed: {summary['passed']}")
        lines.append(f"  Failed: {summary['failed']}")
        lines.append(f"  Warnings: {summary['warnings']}")
        lines.append("")
        
        # System Info
        system_info = report_data["system_info"]
        lines.append("SYSTEM INFORMATION:")
        lines.append(f"  OS: {system_info['platform']['system']} {system_info['platform']['release']}")
        lines.append(f"  Python: {system_info['platform']['python_version']}")
        lines.append(f"  CPU: {system_info['resources']['cpu_count']} cores")
        lines.append(f"  Memory: {system_info['resources']['memory_total_gb']:.1f} GB")
        
        if system_info.get("gpu", {}).get("count", 0) > 0:
            lines.append(f"  GPU: {system_info['gpu']['count']} device(s)")
            for gpu in system_info["gpu"]["devices"]:
                lines.append(f"    - {gpu['name']} ({gpu['memory_total']} MB)")
        lines.append("")
        
        # Diagnostic Results by Category
        results_by_category = {}
        for name, result in report_data["diagnostic_results"].items():
            # Find category for this check
            category = "unknown"
            for cat, checks in self.categories.items():
                if name in checks:
                    category = cat
                    break
            
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append((name, result))
        
        for category, results in results_by_category.items():
            lines.append(f"{category.upper()} CHECKS:")
            for name, result in results:
                status_symbol = {
                    "pass": "✅",
                    "fail": "❌", 
                    "warning": "⚠️",
                    "info": "ℹ️"
                }.get(result["status"], "❓")
                
                lines.append(f"  {status_symbol} {name}: {result['message']}")
                
                if result["status"] == "fail" and result.get("details", {}).get("error"):
                    lines.append(f"    Error: {result['details']['error']}")
            lines.append("")
        
        return "\n".join(lines)
    
    # Default diagnostic check implementations
    
    def _check_python_version(self) -> DiagnosticResult:
        """Check Python version compatibility"""
        import sys
        
        version = sys.version_info
        min_version = (3, 8)
        
        if version >= min_version:
            return DiagnosticResult(
                name="python_version",
                status="pass",
                message=f"Python {version.major}.{version.minor}.{version.micro} is compatible",
                details={"version": f"{version.major}.{version.minor}.{version.micro}"}
            )
        else:
            return DiagnosticResult(
                name="python_version",
                status="fail",
                message=f"Python {version.major}.{version.minor} is too old (minimum: {min_version[0]}.{min_version[1]})",
                details={"version": f"{version.major}.{version.minor}.{version.micro}"}
            )
    
    def _check_dependencies(self) -> DiagnosticResult:
        """Check required dependencies"""
        required_packages = [
            "torch",
            "transformers", 
            "bittensor",
            "numpy",
            "psutil"
        ]
        
        missing_packages = []
        installed_packages = {}
        
        for package in required_packages:
            try:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
                installed_packages[package] = version
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return DiagnosticResult(
                name="dependencies",
                status="fail",
                message=f"Missing required packages: {', '.join(missing_packages)}",
                details={
                    "missing": missing_packages,
                    "installed": installed_packages
                }
            )
        else:
            return DiagnosticResult(
                name="dependencies",
                status="pass",
                message="All required dependencies are installed",
                details={"installed": installed_packages}
            )
    
    def _check_gpu_availability(self) -> DiagnosticResult:
        """Check GPU availability and drivers"""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                
                return DiagnosticResult(
                    name="gpu_availability",
                    status="pass",
                    message=f"CUDA available with {gpu_count} GPU(s)",
                    details={
                        "cuda_available": True,
                        "gpu_count": gpu_count,
                        "gpu_names": gpu_names,
                        "cuda_version": torch.version.cuda
                    }
                )
            else:
                return DiagnosticResult(
                    name="gpu_availability",
                    status="warning",
                    message="CUDA not available, using CPU only",
                    details={"cuda_available": False}
                )
                
        except ImportError:
            return DiagnosticResult(
                name="gpu_availability",
                status="warning",
                message="PyTorch not available, cannot check GPU",
                details={"torch_available": False}
            )
    
    def _check_config_files(self) -> DiagnosticResult:
        """Check configuration file validity"""
        config_files = [
            "hfa_config.json",
            "subnet_config.json"
        ]
        
        issues = []
        valid_configs = []
        
        for config_file in config_files:
            try:
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    valid_configs.append(config_file)
                else:
                    issues.append(f"Config file not found: {config_file}")
            except json.JSONDecodeError as e:
                issues.append(f"Invalid JSON in {config_file}: {str(e)}")
            except Exception as e:
                issues.append(f"Error reading {config_file}: {str(e)}")
        
        if issues:
            return DiagnosticResult(
                name="config_files",
                status="warning" if valid_configs else "fail",
                message=f"Configuration issues found: {'; '.join(issues)}",
                details={"issues": issues, "valid_configs": valid_configs}
            )
        else:
            return DiagnosticResult(
                name="config_files",
                status="pass",
                message="All configuration files are valid",
                details={"valid_configs": valid_configs}
            )
    
    def _check_model_paths(self) -> DiagnosticResult:
        """Check model file paths and accessibility"""
        # This would check if model files exist and are accessible
        # Implementation depends on specific model storage setup
        return DiagnosticResult(
            name="model_paths",
            status="info",
            message="Model path checking not implemented yet",
            details={}
        )
    
    def _check_bittensor_connection(self) -> DiagnosticResult:
        """Check Bittensor network connectivity"""
        try:
            # Basic connectivity test
            import bittensor as bt
            
            # Try to create a subtensor connection
            subtensor = bt.subtensor()
            
            return DiagnosticResult(
                name="bittensor_connection",
                status="pass",
                message="Bittensor network connection successful",
                details={"network": subtensor.network}
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="bittensor_connection",
                status="fail",
                message=f"Bittensor connection failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_subnet_registration(self) -> DiagnosticResult:
        """Check subnet registration status"""
        # This would check if the subnet is properly registered
        # Implementation depends on specific subnet setup
        return DiagnosticResult(
            name="subnet_registration",
            status="info",
            message="Subnet registration checking not implemented yet",
            details={}
        )
    
    def _check_model_loading(self) -> DiagnosticResult:
        """Test model loading capabilities"""
        try:
            # Try to import model factory
            from template.model_factory import ModelArchitectureFactory
            
            factory = ModelArchitectureFactory()
            
            return DiagnosticResult(
                name="model_loading",
                status="pass",
                message="Model factory initialized successfully",
                details={"available_architectures": factory.get_available_architectures()}
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="model_loading",
                status="fail",
                message=f"Model loading test failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_inference_test(self) -> DiagnosticResult:
        """Test basic inference functionality"""
        # This would test basic inference with a simple model
        # Implementation depends on specific model setup
        return DiagnosticResult(
            name="inference_test",
            status="info",
            message="Inference testing not implemented yet",
            details={}
        )
    
    def _check_memory_usage(self) -> DiagnosticResult:
        """Check memory usage patterns"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                status = "fail"
                message = f"Critical memory usage: {memory.percent:.1f}%"
            elif memory.percent > 80:
                status = "warning"
                message = f"High memory usage: {memory.percent:.1f}%"
            else:
                status = "pass"
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            return DiagnosticResult(
                name="memory_usage",
                status=status,
                message=message,
                details={
                    "percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="memory_usage",
                status="fail",
                message=f"Memory check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_disk_space(self) -> DiagnosticResult:
        """Check available disk space"""
        try:
            import psutil
            
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100
            
            if percent_used > 95:
                status = "fail"
                message = f"Critical disk usage: {percent_used:.1f}%"
            elif percent_used > 85:
                status = "warning"
                message = f"High disk usage: {percent_used:.1f}%"
            else:
                status = "pass"
                message = f"Disk usage normal: {percent_used:.1f}%"
            
            return DiagnosticResult(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "percent_used": percent_used,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3)
                }
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="disk_space",
                status="fail",
                message=f"Disk space check failed: {str(e)}",
                details={"error": str(e)}
            )


def create_diagnostic_system(config: Optional[Dict[str, Any]] = None) -> DiagnosticSystem:
    """Factory function to create diagnostic system"""
    return DiagnosticSystem(config)