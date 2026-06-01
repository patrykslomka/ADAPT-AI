"""FastMCP server — registers all tools and resources for the ADAPT-AI framework."""
from mcp.server.fastmcp import FastMCP

from adapt_ai.mcp_server.tools.rag import rag_retrieve
from adapt_ai.mcp_server.tools.rat import rat_reason
from adapt_ai.mcp_server.tools.validation import validate_output
from adapt_ai.mcp_server.resources.documents import get_domain_documents
from adapt_ai.mcp_server.resources.ontology import get_ontology
from adapt_ai.mcp_server.resources.data import get_domain_data
from adapt_ai.mcp_server.resources.regulations import get_regulations

mcp = FastMCP("adapt-ai")

# ── Tools (Building Blocks) ────────────────────────────────────────────────────

@mcp.tool()
async def rag_retrieve_tool(query: str, n_results: int = 5, domain: str = "healthcare") -> str:
    """Retrieve relevant domain documents using the RAG pipeline.

    Use for: facts check, simple Q&A, direct information retrieval.
    """
    return await rag_retrieve(query, n_results, domain)


@mcp.tool()
async def rat_reason_tool(query: str, context: str = "", domain: str = "healthcare") -> str:
    """Multi-step domain reasoning using the RAT pipeline with Chain-of-Thought.

    Use for: complex reasoning, multi-step questions, differential analysis.
    """
    return await rat_reason(query, context, domain)


@mcp.tool()
async def validate_output_tool(content: str, domain: str = "healthcare") -> dict:
    """Rule-based validation of content against domain regulations (HIPAA/FDA).

    Returns: {"passed": bool, "status": str, "issues": [...], "suggestions": [...]}.
    """
    return await validate_output(content, domain)


# ── Resources (Domain Configuration) ──────────────────────────────────────────

@mcp.resource("domain://documents/{query}")
async def documents_resource(query: str) -> str:
    """Retrieve domain documents from the ChromaDB vector store."""
    return await get_domain_documents(query)


@mcp.resource("domain://ontology/{concept}")
async def ontology_resource(concept: str) -> str:
    """Query the domain ontology graph for concept relationships."""
    return await get_ontology(concept)


@mcp.resource("domain://data/{table}")
async def data_resource(table: str) -> str:
    """Query structured domain data (patients, metrics) from the data store."""
    return await get_domain_data(table)


@mcp.resource("domain://regulations/{domain}")
async def regulations_resource(domain: str) -> str:
    """Load the regulation JSON schema for a given domain."""
    return await get_regulations(domain)
