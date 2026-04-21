from typing import List, Dict
from collections import Counter

class AnomalyDetector:
    """
    Basic rule-based anomaly detector.
    Detects unusual error spikes and abnormal patterns.
    """

    def __init__(self, error_threshold: int = 5):
        self.error_threshold = error_threshold

    def detect(self, structured_logs: List) -> Dict:
        """
        Detect anomalies from structured logs
        """

        if not structured_logs:
            return {
                "anomaly_detected": False,
                "reason": "No logs available"
            }

        # Count log levels
        levels = [log.log_level for log in structured_logs]
        level_counts = Counter(levels)

        error_count = level_counts.get("ERROR", 0)

        # Rule 1: Too many errors
        if error_count >= self.error_threshold:
            return {
                "anomaly_detected": True,
                "type": "ERROR_SPIKE",
                "error_count": error_count,
                "message": f"High number of ERROR logs detected ({error_count})"
            }

        # Rule 2: Error dominates logs
        total_logs = len(structured_logs)
        if total_logs > 0 and (error_count / total_logs) > 0.6:
            return {
                "anomaly_detected": True,
                "type": "ERROR_DOMINANCE",
                "error_count": error_count,
                "message": "ERROR logs dominate overall log volume"
            }

        # No anomaly
        return {
            "anomaly_detected": False,
            "message": "Log behaviour appears normal"
        }


        from src.services.ai_explainer import AIExplainer

        explainer = AIExplainer()
        ai_explanation = explainer.explain(
            logs=results["logs_summary"],
            rca_summary=results["rca_summary"],
            kb_fixes=results["kb_fixes"]
        )

        results["ai_explanation"] = ai_explanation
