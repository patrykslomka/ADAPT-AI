"""MCPClient - the central orchestration hub.

Wraps the FastMCP server for in-process tool/resource calls.
Agents interact exclusively through this client; they never import domain modules.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def _extract_tool_text(result: Any) -> str:
    """Extract text from FastMCP call_tool return value.

    FastMCP >= 2 returns list[ContentBlock] directly.
    Older versions returned (list[ContentBlock], dict).
    ContentBlock has a .text attribute for TextContent.
    """
    # Current FastMCP: returns list[ContentBlock]
    if isinstance(result, list):
        if result:
            return getattr(result[0], "text", str(result[0]))
        return ""
    # Legacy tuple format
    if isinstance(result, tuple) and len(result) == 2:
        content_blocks, structured = result
        if content_blocks:
            return getattr(content_blocks[0], "text", str(content_blocks[0]))
        if structured:
            return str(structured.get("result", ""))
    return str(result)


def _extract_resource_text(result: Any) -> str:
    """Extract text from FastMCP read_resource return value.

    read_resource returns list[ReadResourceContents].
    """
    items = list(result)
    if items:
        return getattr(items[0], "content", str(items[0]))
    return ""


class MCPClient:
    """In-process MCP client - the only interface between agents and domain resources."""

    def __init__(self, server: FastMCP) -> None:
        self._server = server

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        logger.debug("MCP call_tool: %s(%s)", name, list(arguments.keys()))
        result = await self._server.call_tool(name, arguments)
        return _extract_tool_text(result)

    async def read_resource(self, uri: str) -> str:
        logger.debug("MCP read_resource: %s", uri)
        result = await self._server.read_resource(uri)
        return _extract_resource_text(result)

    async def call_tool_dict(self, name: str, arguments: Dict[str, Any]) -> Any:
        import json
        logger.debug("MCP call_tool_dict: %s", name)
        result = await self._server.call_tool(name, arguments)
        text = _extract_tool_text(result)
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}


def build_mcp_client() -> MCPClient:
    from adapt_ai.mcp_server.server import mcp as _mcp_server
    return MCPClient(_mcp_server)
