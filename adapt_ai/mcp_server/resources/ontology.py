"""Ontology resource - serves concept data from the domain's OWL file."""
from adapt_ai.domain.ontology import OntologyGraph
from adapt_ai.domain.profiles import get_domain_profile


async def get_ontology(concept: str, domain: str = "healthcare") -> str:
    """Query the domain ontology for a concept (backed by real OWL file)."""
    profile = get_domain_profile(domain)
    graph = OntologyGraph.for_domain(profile)
    result = graph.query_concept(concept)
    return graph.format_result(result)
