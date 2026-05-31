"""Rule-based validation tool — checks content against domain regulations."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from adapt_ai.config import settings


def _load_regulations(domain: str = "healthcare") -> Dict[str, Any]:
    path = settings.regulations_dir / f"{domain}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"regulations": []}


def _content_is_exempt(content: str, exceptions: List[str]) -> bool:
    content_lower = content.lower()
    return any(exc.lower() in content_lower for exc in exceptions)


async def validate_output(content: str, domain: str = "healthcare") -> Dict[str, Any]:
    """Rule-based validation against domain regulations.

    Returns {"passed": bool, "status": str, "issues": [...], "suggestions": [...]}.
    Critical issues → rejected. Warnings → approved with notes.
    """
    regs = _load_regulations(domain)
    issues: List[Dict] = []
    suggestions: List[str] = []

    for rule in regs.get("regulations", []):
        rule_id = rule["id"]
        severity = rule.get("severity", "low")
        exceptions = rule.get("exceptions", [])

        # Skip if content matches any exception
        if _content_is_exempt(content, exceptions):
            continue

        # Check patterns
        for pattern in rule.get("patterns", []):
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    "rule_id": rule_id,
                    "description": rule["description"],
                    "severity": severity,
                    "matched_pattern": pattern,
                })
                if severity == "critical":
                    suggestions.append(f"Remove or anonymise content matching '{pattern}' (rule {rule_id})")
                break  # one issue per rule is enough

        # Check required phrases (e.g. disclaimer)
        if "required_phrases" in rule and not _content_is_exempt(content, exceptions):
            if not any(
                phrase.lower() in content.lower()
                for phrase in rule["required_phrases"]
            ):
                issues.append({
                    "rule_id": rule_id,
                    "description": f"{rule['description']} — required phrase missing",
                    "severity": severity,
                })
                suggestions.append(
                    f"Add one of: {rule['required_phrases']} (rule {rule_id})"
                )

    critical_issues = [i for i in issues if i["severity"] == "critical"]
    high_issues = [i for i in issues if i["severity"] == "high"]

    if critical_issues:
        return {
            "passed": False,
            "status": "rejected",
            "issues": issues,
            "suggestions": suggestions,
        }
    if high_issues:
        return {
            "passed": True,
            "status": "warning",
            "issues": issues,
            "suggestions": suggestions,
        }
    return {
        "passed": True,
        "status": "approved",
        "issues": issues,
        "suggestions": suggestions,
    }
