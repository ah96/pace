"""
PACE - Pathwise Affordance Counterfactual Explanations.

Explains why a robot cannot reach a goal, and what could be changed about the
environment so that it could. The pipeline is:

    ontology  ->  affordances  ->  navigation graph  ->  A* plan
                                                          |
                                    counterfactual search  |
                                    (what would need to change?)
                                                          v
                                                     explanation

Typical use::

    from pace import PaceNavigator, build_navigation_ontology

Module map:
    ontology.py        OWL 2 vocabulary and the typed property store
    domain.py          affordance/door/surface enumerations, ontology builder
    affordance.py      forward-chaining affordance rules and edge costs
    navigation.py      navigation graph, A* planner, Yen's k-shortest paths
    counterfactual.py  the counterfactual search (greedy and exhaustive)
    explanation.py     three-layer natural-language output
    orchestrator.py    PaceNavigator, the end-to-end entry point
    evaluation.py      ground truth, baselines and metrics
    visualization.py   matplotlib rendering of navigation graphs
"""

from .config        import DEFAULT_CONFIG, NavCostConfig
from .ontology      import Ontology, OntologyClass, OntologyIndividual
from .domain         import (
    AffordanceType,
    DoorState,
    SurfaceType,
    MobilityType,
    build_navigation_ontology,
)
from .affordance     import AffordanceReasoner, AffordanceResult
from .navigation     import (
    NavigationGraph, NavNode, NavEdge, NavPath, AStarPlanner, DijkstraPlanner,
)
from .counterfactual import CounterfactualEngine, Counterfactual, PropertyChange
from .explanation    import ExplanationGenerator, NavigationExplanation, ChatCompletionsNarrator
from .orchestrator   import PaceNavigator
from .visualization  import draw_navigation_graph

__all__ = [
    "NavCostConfig",
    "DEFAULT_CONFIG",
    "draw_navigation_graph",
    "PaceNavigator",
    "build_navigation_ontology",
    "Ontology",
    "OntologyClass",
    "OntologyIndividual",
    "AffordanceType",
    "DoorState",
    "SurfaceType",
    "MobilityType",
    "AffordanceReasoner",
    "AffordanceResult",
    "NavigationGraph",
    "NavNode",
    "NavEdge",
    "NavPath",
    "AStarPlanner",
    "DijkstraPlanner",
    "CounterfactualEngine",
    "Counterfactual",
    "PropertyChange",
    "ExplanationGenerator",
    "NavigationExplanation",
    "ChatCompletionsNarrator",
]
