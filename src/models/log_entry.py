from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class LogEntry:
    timestamp: str
    log_level: str
    component: str
    error_code: Optional[str]
    message: str
    zone: str
    client: str
    app: str
    version: str
    raw_line: str
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "log_level": self.log_level,
            "component": self.component,
            "error_code": self.error_code,
            "message": self.message,
            "zone": self.zone,
            "client": self.client,
            "app": self.app,
            "version": self.version,
            "raw": self.raw_line
        }