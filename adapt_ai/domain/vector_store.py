"""ChromaDB vector store — wraps per-domain collections."""
from typing import List, Dict, Any
import chromadb

from adapt_ai.config import settings


class VectorStore:
    """Wrapper around a single ChromaDB collection for domain document retrieval."""

    _instances: "dict[str, VectorStore]" = {}

    def __init__(self, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        # Do not pass an embedding_function — must match whatever seeded each collection.
        self._collection = self._client.get_or_create_collection(name=collection_name)

    @classmethod
    def for_collection(cls, collection_name: str) -> "VectorStore":
        """Return a cached VectorStore for the given collection name."""
        if collection_name not in cls._instances:
            cls._instances[collection_name] = cls(collection_name)
        return cls._instances[collection_name]

    @classmethod
    def get(cls) -> "VectorStore":
        """Return the default VectorStore (settings.chroma_collection). Back-compat."""
        return cls.for_collection(settings.chroma_collection)

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
            return "No relevant context found."
        parts = []
        for i, d in enumerate(docs, 1):
            source = d["metadata"].get("source", d["metadata"].get("type", "unknown"))
            parts.append(f"[{i}] ({source})\n{d['content']}")
        return "\n\n".join(parts)
