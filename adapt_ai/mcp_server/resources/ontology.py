"""Ontology resource — serves concept graph data via MCP resource protocol."""
from adapt_ai.domain.ontology import OntologyGraph


async def get_ontology(concept: str) -> str:
    """Query domain ontology graph for concept relationships."""
    graph = OntologyGraph.get()
    result = graph.query_concept(concept)
    return graph.format_result(result)
