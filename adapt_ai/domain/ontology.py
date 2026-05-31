"""Healthcare domain ontology backed by Neo4j.

Tries to connect to Neo4j on first access. Falls back to a seeded in-memory
NetworkX graph when Neo4j is unavailable (no password configured, server not
running, etc.), so benchmarks and tests remain runnable without a running
database.

Neo4j schema
------------
(:Concept {name, category, symptoms_str, treatment})
-[:RELATED_TO {relation}]->(:Concept)

Cypher to seed the database (run once):
    python -m adapt_ai.domain.ontology --seed
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── In-memory fallback data ────────────────────────────────────────────────────

_CONCEPTS: List[tuple] = [
    ("tuberculosis",      "infectious_disease", ["cough", "fever", "night_sweats", "weight_loss"], "RIPE therapy"),
    ("hypertension",      "cardiovascular",     ["headache", "dizziness"],                          "ACE inhibitors, beta-blockers, lifestyle"),
    ("diabetes_type2",    "metabolic",          ["polyuria", "polydipsia", "fatigue"],              "metformin, insulin, lifestyle"),
    ("pneumonia",         "respiratory",        ["cough", "fever", "dyspnea"],                      "antibiotics"),
    ("myocardial_infarction", "cardiovascular", ["chest_pain", "diaphoresis", "dyspnea"],           "MONA, PCI"),
    ("dengue",            "infectious_disease", ["fever", "rash", "joint_pain", "leukopenia"],      "supportive"),
    ("malaria",           "infectious_disease", ["fever", "chills", "splenomegaly"],                "antimalarials"),
    ("sepsis",            "systemic",           ["fever", "tachycardia", "hypotension"],            "antibiotics, fluids"),
    ("hypothyroidism",    "endocrine",          ["fatigue", "weight_gain", "cold_intolerance"],     "levothyroxine"),
    ("hyperaldosteronism","endocrine",          ["hypertension", "hypokalemia", "muscle_cramps"],   "spironolactone, surgery"),
]

_EDGES: List[tuple] = [
    ("tuberculosis",        "pneumonia",            "differential"),
    ("hypertension",        "myocardial_infarction","risk_factor"),
    ("diabetes_type2",      "hypertension",         "comorbid"),
    ("dengue",              "malaria",              "differential"),
    ("hyperaldosteronism",  "hypertension",         "causes"),
]


# ── Seed / query helpers ───────────────────────────────────────────────────────

_SEED_CYPHER = """\
MERGE (c:Concept {name: $name})
SET c.category = $category,
    c.symptoms  = $symptoms,
    c.treatment = $treatment
"""

_EDGE_CYPHER = """\
MATCH (a:Concept {name: $src}), (b:Concept {name: $dst})
MERGE (a)-[r:RELATED_TO {relation: $relation}]->(b)
"""

_QUERY_CYPHER = """\
MATCH (c:Concept {name: $name})
OPTIONAL MATCH (c)-[r:RELATED_TO]->(n)
RETURN c.name       AS name,
       c.category   AS category,
       c.symptoms   AS symptoms,
       c.treatment  AS treatment,
       collect({name: n.name, relation: r.relation}) AS related
"""


# ── OntologyGraph ──────────────────────────────────────────────────────────────

class OntologyGraph:
    """Healthcare concept graph.  Uses Neo4j when available, NetworkX otherwise."""

    _instance: "OntologyGraph | None" = None

    def __init__(self) -> None:
        self._driver = None       # neo4j.GraphDatabase.driver instance
        self._nx_graph = None     # networkx.DiGraph fallback

        if self._try_neo4j():
            logger.info("OntologyGraph: connected to Neo4j")
        else:
            logger.info("OntologyGraph: Neo4j unavailable — using in-memory fallback")
            self._build_nx_graph()

    # ── singleton ──────────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "OntologyGraph":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── backend initialisation ─────────────────────────────────────────────────

    def _try_neo4j(self) -> bool:
        try:
            from neo4j import GraphDatabase
            from adapt_ai.config import settings
        except ImportError:
            return False

        if not settings.neo4j_password:
            return False

        try:
            auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
            driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
            driver.verify_connectivity()
            self._driver = driver
            self._db = settings.neo4j_database
            return True
        except Exception as exc:
            logger.debug("Neo4j connection failed: %s", exc)
            return False

    def _build_nx_graph(self) -> None:
        try:
            import networkx as nx
            g = nx.DiGraph()
            for name, cat, symptoms, treatment in _CONCEPTS:
                g.add_node(name, category=cat, symptoms=symptoms, treatment=treatment)
            for src, dst, rel in _EDGES:
                g.add_edge(src, dst, relation=rel)
            self._nx_graph = g
        except ImportError:
            logger.warning("networkx not installed — ontology queries will return empty results")

    # ── public API ─────────────────────────────────────────────────────────────

    def query_concept(self, concept: str) -> Dict[str, Any]:
        """Return concept attributes and related concepts."""
        concept_norm = concept.lower().replace(" ", "_")

        if self._driver is not None:
            return self._query_neo4j(concept_norm)
        return self._query_nx(concept_norm)

    def format_result(self, result: Dict[str, Any]) -> str:
        if not result.get("found"):
            return f"Concept '{result.get('concept')}' not found in ontology."
        attrs = result.get("attributes", {})
        lines = [f"Concept: {result['concept']}"]
        lines.append(f"Category: {attrs.get('category', 'unknown')}")
        symptoms = attrs.get("symptoms")
        if symptoms:
            sym_str = ", ".join(symptoms) if isinstance(symptoms, list) else symptoms
            lines.append(f"Symptoms: {sym_str}")
        if attrs.get("treatment"):
            lines.append(f"Treatment: {attrs['treatment']}")
        if result.get("related"):
            rels = ", ".join(
                f"{r['name']} ({r['relation']})"
                for r in result["related"]
                if r.get("name")
            )
            if rels:
                lines.append(f"Related: {rels}")
        return "\n".join(lines)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # ── Neo4j backend ──────────────────────────────────────────────────────────

    def _query_neo4j(self, concept: str) -> Dict[str, Any]:
        try:
            with self._driver.session(database=self._db) as session:
                record = session.run(_QUERY_CYPHER, name=concept).single()
            if record is None:
                return {"concept": concept, "found": False}
            symptoms = record["symptoms"]
            return {
                "concept": record["name"],
                "found": True,
                "attributes": {
                    "category":  record["category"],
                    "symptoms":  symptoms if isinstance(symptoms, list) else [symptoms],
                    "treatment": record["treatment"],
                },
                "related": [r for r in record["related"] if r.get("name")],
            }
        except Exception as exc:
            logger.error("Neo4j query error: %s", exc)
            return {"concept": concept, "found": False, "error": str(exc)}

    def seed_neo4j(self) -> None:
        """Write hardcoded concepts and edges into Neo4j (idempotent via MERGE)."""
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialised")
        with self._driver.session(database=self._db) as session:
            for name, cat, symptoms, treatment in _CONCEPTS:
                session.run(_SEED_CYPHER, name=name, category=cat,
                            symptoms=symptoms, treatment=treatment)
            for src, dst, rel in _EDGES:
                session.run(_EDGE_CYPHER, src=src, dst=dst, relation=rel)
        logger.info("Neo4j seeded with %d concepts and %d edges", len(_CONCEPTS), len(_EDGES))

    # ── NetworkX fallback backend ──────────────────────────────────────────────

    def _query_nx(self, concept: str) -> Dict[str, Any]:
        if self._nx_graph is None:
            return {"concept": concept, "found": False, "error": "No backend available"}

        if concept not in self._nx_graph:
            for node in self._nx_graph.nodes:
                if concept in node or node in concept:
                    concept = node
                    break
            else:
                return {"concept": concept, "found": False}

        attrs = dict(self._nx_graph.nodes[concept])
        related = [
            {"name": n, "relation": self._nx_graph.edges[concept, n].get("relation")}
            for n in self._nx_graph.successors(concept)
        ]
        return {
            "concept": concept,
            "found": True,
            "attributes": attrs,
            "related": related,
        }


# ── CLI seed helper ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--seed" in sys.argv:
        graph = OntologyGraph.get()
        if graph._driver is None:
            print("ERROR: Neo4j not connected.  Set NEO4J_PASSWORD in .env and ensure Neo4j is running.")
            sys.exit(1)
        graph.seed_neo4j()
        print(f"Seeded {len(_CONCEPTS)} concepts and {len(_EDGES)} edges into Neo4j.")
    else:
        print("Usage: python -m adapt_ai.domain.ontology --seed")
