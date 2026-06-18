"""Domain documents resource - serves ChromaDB documents via MCP resource protocol."""
from adapt_ai.domain.vector_store import VectorStore


async def get_domain_documents(query: str) -> str:
    """Retrieve domain documents from ChromaDB vector store."""
    store = VectorStore.get()
    docs = store.query(query, n_results=5)
    return store.format_context(docs)
