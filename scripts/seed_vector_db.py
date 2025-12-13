#!/usr/bin/env python3
"""Seed the vector database with clinical knowledge."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.building_blocks.rag import RAGBlock, seed_clinical_knowledge


def main():
    """Seed the vector database."""
    print("🗄️  Seeding vector database with clinical knowledge...\n")

    # Initialize RAG block
    rag = RAGBlock(
        collection_name="clinical_knowledge",
        persist_directory=Path("./data/chroma_db")
    )

    # Check if already seeded
    stats = rag.get_collection_stats()
    if stats["document_count"] > 0:
        print(f"⚠️  Collection already has {stats['document_count']} documents.")
        response = input("Do you want to clear and re-seed? (y/N): ")
        if response.lower() == 'y':
            rag.clear_collection()
            print("   Cleared existing collection.")
        else:
            print("   Keeping existing data.")
            return

    # Seed with clinical knowledge
    ontology_path = Path(__file__).parent.parent / "src" / "domain" / "ontologies"
    total_added = seed_clinical_knowledge(rag, ontology_path)

    print(f"\n✅ Seeded {total_added} documents into vector database")

    # Show stats
    stats = rag.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   - Collection: {stats['collection_name']}")
    print(f"   - Documents: {stats['document_count']}")
    print(f"   - Embedding Model: {stats['embedding_model']}")

    # Test query
    print("\n🔍 Testing retrieval...")
    result = rag.process_query("What are the symptoms of tuberculosis?")
    print(f"   - Query: 'What are the symptoms of tuberculosis?'")
    print(f"   - Results found: {result['num_results']}")
    if result['documents']:
        print(f"   - Top result preview: {result['documents'][0][:100]}...")


if __name__ == "__main__":
    main()
