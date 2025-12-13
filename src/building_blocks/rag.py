"""RAG (Retrieval-Augmented Generation) Building Block."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RAGBlock:
    """Retrieval-Augmented Generation building block.

    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    """

    def __init__(
        self,
        collection_name: str = "clinical_knowledge",
        persist_directory: Path = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize RAG block.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Name of sentence-transformer model to use
        """
        self.collection_name = collection_name

        if persist_directory is None:
            persist_directory = Path("./data/chroma_db")

        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self._embedding_model = None

        # Initialize ChromaDB
        self._client = None
        self._collection = None

        logger.info(f"RAG block initialized with collection: {collection_name}")

    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    @property
    def client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            if not CHROMADB_AVAILABLE:
                raise ImportError(
                    "chromadb not installed. Run: pip install chromadb"
                )
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )
        return self._client

    @property
    def collection(self):
        """Get or create collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """Add documents to the collection.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        # Generate embeddings
        embeddings = self.embed_texts(documents)

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids
        )

        logger.info(f"Added {len(documents)} documents to collection")
        return len(documents)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        include_distances: bool = True
    ) -> Dict[str, Any]:
        """Query the collection for similar documents.

        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Optional filter conditions
            include_distances: Whether to include similarity distances

        Returns:
            Dict with documents, metadatas, distances
        """
        # Generate query embedding
        query_embedding = self.embed_text(query_text)

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"] if include_distances else ["documents", "metadatas"]
        )

        # Format results
        formatted = {
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "ids": results.get("ids", [[]])[0]
        }

        if include_distances:
            formatted["distances"] = results.get("distances", [[]])[0]

        logger.info(f"Query returned {len(formatted['documents'])} results")
        return formatted

    def process_query(
        self,
        query: str,
        n_results: int = 5,
        format_context: bool = True
    ) -> Dict[str, Any]:
        """Process a query and return formatted context.

        Args:
            query: User query
            n_results: Number of documents to retrieve
            format_context: Whether to format results as context string

        Returns:
            Dict with context, documents, and metadata
        """
        results = self.query(query, n_results=n_results)

        if format_context and results["documents"]:
            context_parts = []
            for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
                source = meta.get("source", "Unknown")
                context_parts.append(f"[{i+1}] ({source})\n{doc}")

            context = "\n\n".join(context_parts)
        else:
            context = ""

        return {
            "query": query,
            "context": context,
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "num_results": len(results["documents"])
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dict with collection stats
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": self.embedding_model_name
        }

    def clear_collection(self):
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.info(f"Cleared collection: {self.collection_name}")


def seed_clinical_knowledge(rag_block: RAGBlock, ontology_path: Path = None) -> int:
    """Seed the RAG database with clinical knowledge from ontology.

    Args:
        rag_block: RAGBlock instance
        ontology_path: Path to ontology directory

    Returns:
        Number of documents added
    """
    if ontology_path is None:
        ontology_path = Path(__file__).parent.parent / "domain" / "ontologies"

    total_added = 0

    # Load clinical ontology
    clinical_path = ontology_path / "clinical_ontology.json"
    if clinical_path.exists():
        with open(clinical_path, 'r') as f:
            ontology = json.load(f)

        # Add diseases
        documents = []
        metadatas = []
        ids = []

        for disease in ontology.get("diseases", []):
            doc = f"""Disease: {disease['name']}
Category: {disease.get('category', 'Unknown')}
ICD-10: {disease.get('icd10', 'N/A')}
Description: {disease.get('description', '')}

Typical Symptoms: {', '.join(disease.get('typical_symptoms', []))}
Risk Factors: {', '.join(disease.get('risk_factors', []))}
First Line Treatment: {', '.join(disease.get('first_line_treatment', []))}
Treatment Duration: {disease.get('treatment_duration', 'N/A')}

Guidelines Source: {disease.get('guidelines_source', 'N/A')}"""

            documents.append(doc)
            metadatas.append({
                "source": "clinical_ontology",
                "type": "disease",
                "id": disease["id"],
                "category": disease.get("category", "Unknown")
            })
            ids.append(f"disease_{disease['id']}")

        # Add symptoms
        for symptom in ontology.get("symptoms", []):
            doc = f"""Symptom: {symptom['name']}
Description: {symptom.get('description', '')}
Severity: {symptom.get('severity', 'variable')}
Red Flag: {'Yes - Requires immediate attention' if symptom.get('red_flag') else 'No'}

Associated Conditions: {', '.join(symptom.get('associated_conditions', []))}
Recommended Workup: {', '.join(symptom.get('workup', []))}"""

            documents.append(doc)
            metadatas.append({
                "source": "clinical_ontology",
                "type": "symptom",
                "id": symptom["id"],
                "red_flag": symptom.get("red_flag", False)
            })
            ids.append(f"symptom_{symptom['id']}")

        # Add treatments
        for treatment in ontology.get("treatments", []):
            doc = f"""Treatment: {treatment['name']}
Generic Name: {treatment.get('generic_name', '')}
Drug Class: {treatment.get('drug_class', '')}
Mechanism: {treatment.get('mechanism', '')}

Indications: {', '.join(treatment.get('indications', []))}
Typical Dose: {treatment.get('typical_dose', 'N/A')}
Route: {treatment.get('route', 'N/A')}

Contraindications: {', '.join(treatment.get('contraindications', []))}
Common Side Effects: {', '.join(treatment.get('common_side_effects', []))}
Monitoring: {', '.join(treatment.get('monitoring', []))}"""

            documents.append(doc)
            metadatas.append({
                "source": "clinical_ontology",
                "type": "treatment",
                "id": treatment["id"],
                "drug_class": treatment.get("drug_class", "Unknown")
            })
            ids.append(f"treatment_{treatment['id']}")

        if documents:
            added = rag_block.add_documents(documents, metadatas, ids)
            total_added += added
            logger.info(f"Added {added} clinical documents")

    # Load drug database
    drug_path = ontology_path / "drug_database.json"
    if drug_path.exists():
        with open(drug_path, 'r') as f:
            drug_db = json.load(f)

        documents = []
        metadatas = []
        ids = []

        for med in drug_db.get("medications", []):
            interactions_text = ""
            for interaction in med.get("drug_interactions", []):
                interactions_text += f"\n- {interaction['drug']}: {interaction.get('clinical_effect', '')} ({interaction.get('severity', 'unknown')} severity)"

            contraindications_text = ""
            for contra in med.get("contraindications", []):
                contraindications_text += f"\n- {contra['condition']}: {contra.get('reason', '')} ({contra.get('severity', 'unknown')})"

            doc = f"""Drug: {med['generic_name']}
Brand Names: {', '.join(med.get('brand_names', []))}
Drug Class: {med.get('drug_class', '')}

Indications: {', '.join(med.get('indications', []))}
Typical Dose: {med.get('typical_dose', 'N/A')}

Drug Interactions:{interactions_text if interactions_text else ' None documented'}

Contraindications:{contraindications_text if contraindications_text else ' None documented'}

Monitoring Parameters: {', '.join(med.get('monitoring_parameters', []))}
Serious Side Effects: {', '.join(med.get('serious_side_effects', []))}"""

            documents.append(doc)
            metadatas.append({
                "source": "drug_database",
                "type": "medication",
                "id": med["id"],
                "drug_class": med.get("drug_class", "Unknown")
            })
            ids.append(f"drug_{med['id']}")

        if documents:
            added = rag_block.add_documents(documents, metadatas, ids)
            total_added += added
            logger.info(f"Added {added} drug documents")

    return total_added
