"""Domain data store — PostgreSQL with JSON file fallback."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from adapt_ai.config import settings

logger = logging.getLogger(__name__)

# JSON fallback data representing structured domain data
_FALLBACK_PATIENTS: List[Dict] = []
_FALLBACK_METRICS: List[Dict] = []


def _load_fallback_patients() -> List[Dict]:
    """Load synthetic patient data from the existing JSON files."""
    global _FALLBACK_PATIENTS
    if _FALLBACK_PATIENTS:
        return _FALLBACK_PATIENTS
    patients_dir = settings.data_dir / "synthetic_patients"
    if not patients_dir.exists():
        # Try the original src location
        patients_dir = Path("./src/domain/synthetic_patients")
    if patients_dir.exists():
        for f in patients_dir.glob("*.json"):
            try:
                _FALLBACK_PATIENTS.append(json.loads(f.read_text()))
            except Exception:
                pass
    return _FALLBACK_PATIENTS


class DomainDB:
    """Thin data access layer — PostgreSQL when available, JSON files otherwise."""

    _instance: "DomainDB | None" = None

    def __init__(self) -> None:
        self._use_fallback = settings.postgres_fallback_json
        if not self._use_fallback:
            try:
                import asyncpg  # noqa: F401
                logger.info("PostgreSQL mode enabled")
            except ImportError:
                logger.warning("asyncpg not installed — using JSON fallback")
                self._use_fallback = True

    @classmethod
    def get(cls) -> "DomainDB":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        patients = _load_fallback_patients()
        for p in patients:
            if p.get("patient_id") == patient_id or p.get("id") == patient_id:
                return p
        return None

    async def list_patients(self) -> List[Dict[str, Any]]:
        return _load_fallback_patients()

    async def get_metric_history(
        self, metric_type: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return recent LLMOps metrics (reads from SQLite metrics.db if present)."""
        db_path = settings.data_dir / "metrics.db"
        if db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cur = conn.execute(
                    "SELECT * FROM query_metrics ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                conn.close()
                return rows
            except Exception as e:
                logger.warning("metrics.db read failed: %s", e)
        return []
