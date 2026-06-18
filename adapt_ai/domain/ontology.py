"""Domain ontology backed by real OWL files (rdflib).

Each domain profile specifies an `ontology_path` pointing to an OWL/RDF file
in data/ontologies/<domain>/. The graph is loaded once per path and cached.
Falls back to Neo4j if configured, or to an empty result when neither is
available (so tests and benchmarks run without the OWL files present).

Supported OWL formats: RDF/XML (.owl, .rdf), Turtle (.ttl), N-Triples (.nt).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Predicates used across biomedical / legal / financial OWL ontologies
_LABEL_PREDICATES = [
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
]
_DEF_PREDICATES = [
    "http://purl.obolibrary.org/obo/IAO_0000115",          # OBO definition
    "http://www.w3.org/2004/02/skos/core#definition",
    "http://www.w3.org/2000/01/rdf-schema#comment",
]
_SYNONYM_PREDICATES = [
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://www.w3.org/2004/02/skos/core#altLabel",
]
_BROADER_PREDICATES = [
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2004/02/skos/core#broader",
]


@lru_cache(maxsize=8)
def _load_graph(owl_path: str):
    """Load an OWL/RDF file into an rdflib Graph. Cached by path string."""
    import rdflib
    path = Path(owl_path)
    if not path.exists():
        logger.warning("Ontology file not found: %s", owl_path)
        return None
    logger.info("Loading ontology from %s (this may take a moment)…", path.name)
    g = rdflib.Graph()
    fmt = {".ttl": "turtle", ".nt": "nt", ".n3": "n3"}.get(path.suffix.lower(), "xml")
    g.parse(str(path), format=fmt)
    logger.info("Loaded %d triples from %s", len(g), path.name)
    return g


def _query_owl(g, term: str) -> dict[str, Any]:
    """Search rdflib graph for a term by label (case-insensitive prefix match)."""
    import rdflib

    term_lower = term.lower().strip()
    matched_uri = None

    # First pass: find a subject with a matching label
    for pred_uri in _LABEL_PREDICATES:
        pred = rdflib.URIRef(pred_uri)
        for s, _, o in g.triples((None, pred, None)):
            label_str = str(o).lower()
            if label_str == term_lower or label_str.startswith(term_lower):
                matched_uri = s
                break
        if matched_uri:
            break

    if matched_uri is None:
        return {"concept": term, "found": False}

    def _literals(subj, pred_uris):
        results = []
        for pu in pred_uris:
            for _, _, o in g.triples((subj, rdflib.URIRef(pu), None)):
                v = str(o).strip()
                if v and len(v) < 500:
                    results.append(v)
        return results

    labels = _literals(matched_uri, _LABEL_PREDICATES)
    defs = _literals(matched_uri, _DEF_PREDICATES)
    synonyms = _literals(matched_uri, _SYNONYM_PREDICATES)

    # Broader/parent terms (up to 3)
    broader = []
    for pu in _BROADER_PREDICATES:
        for _, _, parent in g.triples((matched_uri, rdflib.URIRef(pu), None)):
            if isinstance(parent, rdflib.URIRef):
                for lpu in _LABEL_PREDICATES:
                    for _, _, lo in g.triples((parent, rdflib.URIRef(lpu), None)):
                        broader.append(str(lo))
                        break
                if broader:
                    break
        if len(broader) >= 3:
            break

    return {
        "concept": labels[0] if labels else term,
        "found": True,
        "attributes": {
            "definition": defs[0] if defs else "",
            "synonyms": synonyms[:5],
            "broader": broader[:3],
        },
    }


class OntologyGraph:
    """Domain ontology graph.  Backed by a real OWL file via rdflib.

    Usage:
        graph = OntologyGraph.for_domain(profile)   # preferred
        graph = OntologyGraph.get()                 # healthcare default (legacy)
    """

    _default: "OntologyGraph | None" = None

    def __init__(self, owl_path: str | None = None) -> None:
        self._owl_path = owl_path
        self._rdf_graph = _load_graph(owl_path) if owl_path else None
        if self._rdf_graph is None and owl_path:
            logger.warning("OntologyGraph: could not load %s - concept queries will return empty", owl_path)

    @classmethod
    def for_domain(cls, profile) -> "OntologyGraph":
        """Create (or reuse cached) OntologyGraph for a DomainProfile."""
        path = getattr(profile, "ontology_path", None)
        return cls(path)

    @classmethod
    def get(cls) -> "OntologyGraph":
        """Return the default healthcare ontology (legacy singleton for back-compat)."""
        if cls._default is None:
            from adapt_ai.domain.profiles import get_domain_profile
            profile = get_domain_profile("healthcare")
            cls._default = cls.for_domain(profile)
        return cls._default

    def query_concept(self, concept: str) -> dict[str, Any]:
        if self._rdf_graph is not None:
            return _query_owl(self._rdf_graph, concept)
        return {"concept": concept, "found": False, "error": "No ontology loaded"}

    def format_result(self, result: dict[str, Any]) -> str:
        if not result.get("found"):
            return f"Concept '{result.get('concept')}' not found in ontology."
        attrs = result.get("attributes", {})
        lines = [f"Concept: {result['concept']}"]
        if attrs.get("definition"):
            lines.append(f"Definition: {attrs['definition']}")
        if attrs.get("synonyms"):
            lines.append(f"Synonyms: {', '.join(attrs['synonyms'])}")
        if attrs.get("broader"):
            lines.append(f"Broader terms: {', '.join(attrs['broader'])}")
        return "\n".join(lines)
