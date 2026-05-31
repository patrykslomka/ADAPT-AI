"""Domain data resource — serves structured data from PostgreSQL/JSON fallback."""
from adapt_ai.domain.db import DomainDB


async def get_domain_data(table: str) -> str:
    """Query domain data store for structured data."""
    import json
    db = DomainDB.get()
    if table == "patients":
        rows = await db.list_patients()
        return json.dumps(rows[:5], indent=2, default=str)
    if table == "metrics":
        rows = await db.get_metric_history("all", limit=10)
        return json.dumps(rows, indent=2, default=str)
    return f"Table '{table}' not found."
