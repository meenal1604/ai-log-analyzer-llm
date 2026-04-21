# src/services/anomaly_detector.py

from typing import List

def detect_error_anomaly(structured_logs: List, threshold: int = 5) -> dict:
    """
    Simple anomaly detection based on error count
    """

    error_count = sum(
        1 for log in structured_logs
        if getattr(log, "log_level", "") == "ERROR"
    )

    is_anomaly = error_count > threshold

    return {
        "error_count": error_count,
        "threshold": threshold,
        "anomaly_detected": is_anomaly,
        "message": (
            "⚠️ Error spike detected"
            if is_anomaly
            else "✅ Error level is normal"
        )
    }

