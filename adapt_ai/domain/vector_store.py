"""ChromaDB vector store — wraps the existing 'clinical_knowledge' collection."""
from typing import List, Dict, Any
import chromadb

from adapt_ai.config import settings


class VectorStore:
    """Singleton wrapper around ChromaDB for domain document retrieval."""

    _instance: "VectorStore | None" = None

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        # Do not pass an embedding_function — use whatever the collection was seeded with.
        # The existing 'clinical_knowledge' collection uses ChromaDB's default EF.
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
        )

    @classmethod
    def get(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Return top-n documents with their metadata and distances."""
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append({"content": doc, "metadata": meta, "distance": dist})
        return docs

    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string."""
        if not docs:
            return "No relevant clinical context found."
        parts = []
        for i, d in enumerate(docs, 1):
            source = d["metadata"].get("source", d["metadata"].get("type", "unknown"))
            parts.append(f"[{i}] ({source})\n{d['content']}")
        return "\n\n".join(parts)
