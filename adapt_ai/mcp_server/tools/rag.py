"""RAG tool — 4-stage retrieval pipeline exposed as an MCP tool."""
from adapt_ai.domain.vector_store import VectorStore


async def rag_retrieve(query: str, n_results: int = 5) -> str:
    """Retrieve relevant clinical documents using RAG pipeline.

    Stages: query processing → ChromaDB retrieval → context assembly → formatted output.
    """
    store = VectorStore.get()

    # Stage 1: query processing — use the query as-is (already cleaned by caller)
    effective_query = query.strip()

    # Stage 2: ChromaDB retrieval
    docs = store.query(effective_query, n_results=n_results)

    # Stage 3 & 4: context assembly and formatted output
    return store.format_context(docs)
