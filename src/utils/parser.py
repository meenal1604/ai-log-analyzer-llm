from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class StructuredLog:
    timestamp: Optional[str]
    log_level: str
    message: str
    component: Optional[str]
    error_code: Optional[str]
    zone: str
    client: str
    app: str
    version: str

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "log_level": self.log_level,
            "message": self.message,
            "component": self.component,
            "error_code": self.error_code,
            "zone": self.zone,
            "client": self.client,
            "app": self.app,
            "version": self.version
        }


class LogParser:
    def parse_line(self, line: str, zone: str, client: str, app: str, version: str) -> Optional[StructuredLog]:
        line = line.strip()
        if not line:
            return None

        line_lower = line.lower()

        # ✅ FIX 1: Correct log level detection
        if "error" in line_lower:
            log_level = "ERROR"
        elif "warn" in line_lower:
            log_level = "WARN"
        elif "info" in line_lower:
            log_level = "INFO"
        else:
            log_level = "INFO"

        # ✅ Extract timestamp (basic)
        timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2})", line)
        timestamp = timestamp_match.group(1) if timestamp_match else None

        # ✅ Extract error code if present (E_TIMEOUT, ERR-001, etc.)
        error_code_match = re.search(r"\b(E_[A-Z_]+|ERR-\d+)\b", line)
        error_code = error_code_match.group(1) if error_code_match else None

        # ✅ Extract component (basic heuristic)
        component = None
        known_components = ["database", "sip", "api", "auth", "network"]
        for comp in known_components:
            if comp in line_lower:
                component = comp.capitalize()
                break

        return StructuredLog(
            timestamp=timestamp,
            log_level=log_level,
            message=line,
            component=component,
            error_code=error_code,
            zone=zone,
            client=client,
            app=app,
            version=version
        )

