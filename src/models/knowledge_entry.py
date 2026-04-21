from dataclasses import dataclass

@dataclass
class KnowledgeEntry:
    issue: str
    root_cause: str
    solution: str
    affected_components: list
    tags: list
    confidence: float = 1.0

class KnowledgeEntry:
    def __init__(
        self,
        issue,
        root_cause,
        solution,
        affected_components=None,
        tags=None,
        severity="Medium",
        resolution_time="1 hour"
    ):
        self.issue = issue
        self.root_cause = root_cause
        self.solution = solution
        self.affected_components = affected_components or []
        self.tags = tags or []
        self.severity = severity
        self.resolution_time = resolution_time
