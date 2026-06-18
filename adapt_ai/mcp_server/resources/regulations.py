"""Regulations resource - serves JSON schema regulations via MCP resource protocol."""
import json
from adapt_ai.config import settings


async def get_regulations(domain: str) -> str:
    """Load regulation JSON schema for the given domain."""
    path = settings.regulations_dir / f"{domain}.json"
    if path.exists():
        return path.read_text()
    return json.dumps({"error": f"No regulations found for domain '{domain}'"})
