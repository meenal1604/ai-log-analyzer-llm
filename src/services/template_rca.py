# src/services/template_rca.py

from collections import Counter

class TemplateRCA:
    def generate(self, structured_logs: list) -> str:
        if not structured_logs:
            return "No logs available for RCA."

        # Count error logs
        error_logs = [log for log in structured_logs if log.log_level == "ERROR"]

        if not error_logs:
            return "No ERROR logs found. System appears healthy."

        # Find top component
        components = [log.component for log in error_logs if log.component]
        top_component = Counter(components).most_common(1)[0][0] if components else "Unknown"

        # Build RCA template
        rca = f"""
Root Cause:
High number of ERROR logs detected in component: {top_component}

Impact:
Service degradation observed due to repeated failures.

Recommended Fix:
1. Check configuration of {top_component}
2. Restart the service
3. Monitor logs for further errors

Prevention:
Enable monitoring and alerts for early detection.
        """.strip()

        return rca
