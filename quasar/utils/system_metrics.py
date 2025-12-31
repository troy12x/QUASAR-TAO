import psutil
import bittensor as bt
from typing import Dict, Any

def get_system_metrics() -> Dict[str, Any]:
    """
    Fetches current system performance metrics.
    """
    metrics = {}
    try:
        # CPU
        metrics["system/cpu_percent"] = psutil.cpu_percent()
        
        # RAM
        memory = psutil.virtual_memory()
        metrics["system/ram_percent"] = memory.percent
        metrics["system/ram_total_gb"] = memory.total / (1024 ** 3)
        metrics["system/ram_available_gb"] = memory.available / (1024 ** 3)
        
        # Disk
        disk = psutil.disk_usage('/')
        metrics["system/disk_percent"] = disk.percent
        metrics["system/disk_total_gb"] = disk.total / (1024 ** 3)
        metrics["system/disk_free_gb"] = disk.free / (1024 ** 3)
        
    except Exception as e:
        bt.logging.warning(f"⚠️ Failed to fetch system metrics: {e}")
        
    return metrics
