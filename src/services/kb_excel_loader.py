import pandas as pd
from src.models.knowledge_entry import KnowledgeEntry

def load_kb_from_excel(excel_path):
    df = pd.read_csv(excel_path)

    entries = []
    for _, row in df.iterrows():
        entry = KnowledgeEntry(
            issue=row.get("issue", ""),
            root_cause=row.get("root_cause", ""),
            solution=row.get("solution", ""),
            affected_components=[
                c.strip() for c in str(row.get("affected_components", "")).split(",") if c.strip()
            ],
            tags=[
                t.strip() for t in str(row.get("tags", "")).split(",") if t.strip()
            ],
            severity=row.get("severity", "Medium"),
            resolution_time=row.get("resolution_time", "1 hour")
        )
        entries.append(entry)

    return entries
