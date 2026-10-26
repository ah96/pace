"""
scenarios/synthetic.py
======================
Procedurally generated facilities, for measuring how the method scales.

The hospital and warehouse scenarios are hand-authored to exercise particular
blocking conditions, which is what makes them useful for the results table and
useless for a scaling study: there are only nine waypoints, and any runtime
measured on them says nothing about a real building. This module generates
graphs of arbitrary size with the same property vocabulary, so runtime can be
plotted against the one variable that matters.

The layout is a lattice of corridors rather than a tree, because a tree has
exactly one route between any two waypoints and the counterfactual engine
would have no alternatives to question. Blockers are injected at a fixed rate
so the fraction of infeasible tasks does not drift as the graph grows.

Nothing here is claimed to be realistic. It is a stress geometry, and the only
number taken from it is time.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Tuple

from pace import (
    build_navigation_ontology,
    DoorState,
    MobilityType,
    NavEdge,
    NavigationGraph,
    NavNode,
    SurfaceType,
)

#: Metres between adjacent lattice waypoints.
_SPACING = 10.0

#: Fraction of segments carrying each kind of blocker. Chosen so a generated
#: facility has roughly the blocker density of the hand-authored ones.
_BLOCKER_RATE = 0.15


def build_synthetic_world(
    n_nodes:      int,
    seed:         int = 0,
    blocker_rate: float = _BLOCKER_RATE,
) -> Tuple[object, NavigationGraph, Dict[str, object]]:
    """
    A lattice facility with *n_nodes* waypoints and three robots.

    Returns ``(onto, graph, agents)``, matching the signature of the
    hand-authored scenario builders so the same harness runs on either.
    """
    rng   = random.Random(seed)
    onto  = build_navigation_ontology()
    graph = NavigationGraph()

    cols = max(2, int(round(math.sqrt(n_nodes))))
    rows = max(2, int(math.ceil(n_nodes / cols)))

    ids: Dict[Tuple[int, int], str] = {}
    placed = 0
    for r in range(rows):
        for c in range(cols):
            if placed >= n_nodes:
                break
            nid = f"wp_{r}_{c}"
            ids[(r, c)] = nid
            graph.add_node(NavNode(node_id=nid,
                                   position=(c * _SPACING, -r * _SPACING),
                                   label=f"Waypoint {r},{c}"))
            placed += 1

    def segment(name: str, wide: bool) -> object:
        """One passage, mostly benign, occasionally carrying a blocker."""
        props = dict(
            width            = round(rng.uniform(2.0, 4.0), 2) if wide else 1.2,
            height           = round(rng.uniform(2.6, 4.5), 2),
            length           = _SPACING,
            surface_type     = rng.choice([SurfaceType.SMOOTH.value,
                                           SurfaceType.TILED.value,
                                           SurfaceType.ROUGH.value]),
            illumination     = round(rng.uniform(0.4, 0.95), 2),
            obstacle_density = round(rng.uniform(0.0, 0.2), 2),
            crowd_density    = round(rng.uniform(0.0, 0.3), 2),
            slope_angle      = 0.0,
            is_accessible    = True,
            is_hazardous     = False,
            restricted       = False,
        )
        roll = rng.random()
        if roll < blocker_rate:
            # One blocker per affected segment, drawn across the same kinds the
            # hand-authored scenarios use, so no single repair type dominates.
            kind = rng.choice(["narrow", "door", "restricted",
                               "hazard", "slope", "dark"])
            if kind == "narrow":
                props["width"] = round(rng.uniform(0.7, 0.95), 2)
            elif kind == "door":
                props["door_state"] = DoorState.CLOSED.value
            elif kind == "restricted":
                props["restricted"] = True
            elif kind == "hazard":
                props["is_hazardous"] = True
            elif kind == "slope":
                props["slope_angle"] = round(rng.uniform(12.0, 20.0), 1)
            elif kind == "dark":
                props["illumination"] = round(rng.uniform(0.05, 0.2), 2)
        cls = "Doorway" if "door_state" in props else "Corridor"
        return onto.create(name, cls, **props)

    def dist(a: str, b: str) -> float:
        na, nb = graph.nodes[a], graph.nodes[b]
        return math.hypot(na.position[0] - nb.position[0],
                          na.position[1] - nb.position[1])

    # 4-neighbour lattice: every waypoint joins its right and lower neighbour,
    # which leaves a grid of cycles and so a supply of alternative routes.
    for (r, c), nid in sorted(ids.items()):
        for dr, dc in ((0, 1), (1, 0)):
            nbr = ids.get((r + dr, c + dc))
            if nbr is None:
                continue
            seg = segment(f"seg_{nid}_{nbr}", wide=rng.random() > 0.3)
            graph.add_edge(
                NavEdge(from_id=nid, to_id=nbr, space_individual=seg,
                        distance=dist(nid, nbr)),
                bidirectional=True,
            )

    # The same three body plans as the warehouse, so per-robot differentiation
    # is present without introducing a fourth set of numbers to justify.
    agents = {
        "small_bot": onto.create(
            "small_bot", "Robot", robot_width=0.6, robot_height=1.4,
            min_clearance=0.10, max_slope_angle=8.0,
            mobility_type=MobilityType.WHEELED.value,
            can_open_doors=True, has_arm=True),
        "wide_bot": onto.create(
            "wide_bot", "Robot", robot_width=1.2, robot_height=1.8,
            min_clearance=0.20, max_slope_angle=6.0,
            mobility_type=MobilityType.WHEELED.value,
            can_open_doors=False, has_arm=False),
        "climb_bot": onto.create(
            "climb_bot", "Robot", robot_width=0.8, robot_height=1.5,
            min_clearance=0.15, max_slope_angle=25.0,
            mobility_type=MobilityType.TRACKED.value,
            can_open_doors=False, has_arm=False),
    }
    return onto, graph, agents
