#!/usr/bin/env python3
"""Seed a domain's vector collection from its regulation corpus + ontology.

Usage:
    python3 scripts/seed_vector_db.py                 # default: healthcare
    python3 scripts/seed_vector_db.py --domain legal
    python3 scripts/seed_vector_db.py --domain finance --yes   # skip prompt

Each domain's collection (from its profile's ``vector_collection``) is seeded from:
  1. Regulation corpus markdown under data/regulations_corpus/<domain>/*.md  (chunked by section)
  2. Concept terms extracted from the profile's ``ontology_path`` (OWL/RDF/TTL), capped.

No embedding function is passed — Chroma's default must match whatever first seeded the collection.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rdflib
from rdflib.namespace import OWL, RDF, RDFS, SKOS
import chromadb

from adapt_ai.config import settings
from adapt_ai.domain.profiles import get_domain_profile

_ROOT = Path(__file__).parent.parent
_CORPUS_DIR = _ROOT / "data" / "regulations_corpus"


# ── Corpus (markdown) extraction ──────────────────────────────────────────────

def _extract_corpus_documents(domain: str):
    """Chunk each data/regulations_corpus/<domain>/*.md file by '## ' sections."""
    docs, metas, ids = [], [], []
    corpus_dir = _CORPUS_DIR / domain
    if not corpus_dir.exists():
        return docs, metas, ids

    for md in sorted(corpus_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        lines = text.splitlines()
        title = next((l[2:].strip() for l in lines if l.startswith("# ")), md.stem)

        # Split into sections on '## ' headers; keep the title as context.
        sections, current = [], []
        for line in lines:
            if line.startswith("## "):
                if current:
                    sections.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append("\n".join(current).strip())

        for i, section in enumerate(sections):
            if len(section) < 40:  # skip trivial fragments
                continue
            docs.append(f"{title}\n\n{section}")
            metas.append({"source": md.name, "domain": domain, "title": title})
            ids.append(f"{domain}_corpus_{md.stem}_{i}")

    return docs, metas, ids


# ── Ontology (OWL/RDF/TTL) extraction ─────────────────────────────────────────

def _extract_ontology_documents(owl_path: Path, domain: str, max_terms: int = 500):
    """Extract labelled concept entries from an OWL/RDF/TTL file as text documents."""
    docs, metas, ids = [], [], []
    if not owl_path.exists():
        print(f"   (ontology file not found, skipping: {owl_path})")
        return docs, metas, ids

    g = rdflib.Graph()
    try:
        g.parse(str(owl_path))
    except Exception as e:
        print(f"   (ontology parse failed, skipping: {e})")
        return docs, metas, ids
    print(f"   Loaded {len(g)} triples from {owl_path.name}")

    # Treat both owl:Class and skos:Concept as concept nodes.
    concept_nodes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, SKOS.Concept))
    for node in concept_nodes:
        if len(docs) >= max_terms:
            break
        if not isinstance(node, rdflib.URIRef):
            continue

        label = next(g.objects(node, SKOS.prefLabel), None) or next(g.objects(node, RDFS.label), None)
        if label is None:
            continue
        label = str(label)

        definition = str(next(g.objects(node, SKOS.definition), "") or "")
        synonyms = [str(s) for s in g.objects(node, SKOS.altLabel)]
        parents = []
        for parent in list(g.objects(node, RDFS.subClassOf)) + list(g.objects(node, SKOS.broader)):
            pl = next(g.objects(parent, SKOS.prefLabel), None) or next(g.objects(parent, RDFS.label), None)
            if pl:
                parents.append(str(pl))

        parts = [f"Term: {label}"]
        if definition:
            parts.append(f"Definition: {definition}")
        if synonyms:
            parts.append(f"Synonyms: {', '.join(synonyms[:5])}")
        if parents:
            parts.append(f"Category: {', '.join(parents[:3])}")

        docs.append("\n".join(parts))
        metas.append({"source": owl_path.name, "domain": domain, "term": label})
        ids.append(f"{domain}_onto_{node.split('/')[-1].split('#')[-1]}")

    return docs, metas, ids


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed a domain's vector collection.")
    parser.add_argument("--domain", default="healthcare", choices=["healthcare", "legal", "finance"])
    parser.add_argument("--yes", action="store_true", help="skip the clear/re-seed prompt")
    parser.add_argument("--max-terms", type=int, default=500, help="cap on ontology terms")
    args = parser.parse_args()

    profile = get_domain_profile(args.domain)
    collection_name = profile.vector_collection
    print(f"Seeding '{collection_name}' for domain '{args.domain}'\n")

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    count = collection.count()
    if count > 0:
        print(f"Collection '{collection_name}' already has {count} documents.")
        if not args.yes:
            if input("Clear and re-seed? (y/N): ").lower() != "y":
                print("Keeping existing data.")
                return
        collection.delete(ids=collection.get()["ids"])
        print("Cleared existing collection.")

    print("Extracting regulation corpus…")
    c_docs, c_metas, c_ids = _extract_corpus_documents(args.domain)
    print(f"   {len(c_docs)} corpus sections")

    print("Extracting ontology terms…")
    o_docs, o_metas, o_ids = _extract_ontology_documents(
        Path(profile.ontology_path) if not Path(profile.ontology_path).is_absolute()
        else Path(profile.ontology_path),
        args.domain, max_terms=args.max_terms,
    )
    print(f"   {len(o_docs)} ontology terms")

    docs = c_docs + o_docs
    metas = c_metas + o_metas
    ids = c_ids + o_ids
    if not docs:
        print("Nothing to seed — check the corpus directory and ontology path.")
        sys.exit(1)

    batch = 100
    for i in range(0, len(docs), batch):
        collection.upsert(
            documents=docs[i:i + batch],
            metadatas=metas[i:i + batch],
            ids=ids[i:i + batch],
        )

    print(f"\nSeeded {collection.count()} documents into '{collection_name}'")

    print("\nTesting retrieval…")
    probe = {"healthcare": "fever high temperature", "legal": "client confidentiality privilege",
             "finance": "guaranteed investment return risk"}[args.domain]
    results = collection.query(query_texts=[probe], n_results=3)
    for doc in results["documents"][0]:
        print(f"  - {doc[:80].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
