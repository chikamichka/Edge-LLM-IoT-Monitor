from datetime import datetime
from collections import deque
from typing import Dict, List
import psutil
import time

class PerformanceMonitor:
    """Track API performance metrics"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.query_times = deque(maxlen=max_history)
        self.query_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
    def record_query(self, duration: float, success: bool = True):
        """Record a query execution"""
        self.query_times.append(duration)
        self.query_count += 1
        if not success:
            self.error_count += 1
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        uptime = time.time() - self.start_time
        
        avg_query_time = sum(self.query_times) / len(self.query_times) if self.query_times else 0
        
        return {
            "uptime_seconds": round(uptime, 2),
            "total_queries": self.query_count,
            "error_count": self.error_count,
            "success_rate": round((self.query_count - self.error_count) / self.query_count * 100, 2) if self.query_count > 0 else 100,
            "avg_query_time": round(avg_query_time, 2),
            "queries_per_minute": round(self.query_count / (uptime / 60), 2) if uptime > 0 else 0,
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_mb": round(psutil.virtual_memory().used / (1024**2), 2)
            }
        }


# Global monitor instance
monitor = PerformanceMonitor()