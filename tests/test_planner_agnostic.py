"""The explanation layer must not depend on which search produced the path.

The paper claims PACE is planner-agnostic, and that A* returns the optimal
route on these graphs. Both claims are checkable, so we check them rather
than asserting them:

  * ``DijkstraPlanner`` is uniform-cost search. It assumes nothing about the
    cost function beyond non-negativity, which ``cost_floor`` guarantees, so
    its answer is the certified optimum. If scaled-heuristic A* agrees with it
    on every benchmark task, A* has lost nothing to its heuristic.
  * Swapping the planner underneath ``PaceNavigator`` must leave the chosen
    route, its cost and the operator recommendations unchanged.

The scaling is not cosmetic: the emergency-route discount multiplies the whole
cost expression, so a raw Euclidean heuristic can overestimate the remaining
cost and break admissibility.
"""

from __future__ import annotations

import math

import pytest

from pace import AStarPlanner, DijkstraPlanner, PaceNavigator
from pace.affordance import AffordanceReasoner
from scenarios.plant import build_plant_world, build_random_plant

# Tasks of the plant that exercise every kind of constraint it has, plus a
# generated plant, so the check is not confined to the authored layout.
BENCHMARK = [
    ("plant", "tote_amr",           "dock",         "shipping"),       # doors, hazard
    ("plant", "forklift_agv",         "dock",         "shipping"),       # staged pallets, VNA, underpass
    ("plant", "mobile_manipulator", "dock",         "shipping"),       # feasible: opens the doors
    ("plant", "yard_ugv",  "dock",         "yard"),           # feasible: the ramp
    ("plant", "tote_amr",           "receiving",    "sorting"),        # feasible, crowded
    ("plant", "forklift_agv",         "receiving",    "shipping"),       # infeasible
    ("plant", "mobile_manipulator", "quarantine",   "sorting"),        # feasible via the wash
    ("plant", "yard_ugv",  "yard",         "sorting"),        # infeasible
    ("plant", "forklift_agv",         "module_store", "shipping"),       # conjunctive underpass
    ("random", "tote_amr",          "z00",          "z14"),
    ("random", "forklift_agv",        "z00",          "z14"),
]
TASK_IDS = [f"{s}/{r}/{a}->{b}" for s, r, a, b in BENCHMARK]

BUILDERS = {"plant": build_plant_world, "random": lambda: build_random_plant(7)}


def _world(scenario):
    """A fresh world per call: the counterfactual search mutates clones of it."""
    return BUILDERS[scenario]()


@pytest.mark.parametrize("scenario,robot,start,goal", BENCHMARK, ids=TASK_IDS)
def test_astar_matches_uniform_cost_optimum(scenario, robot, start, goal):
    """Scaled-heuristic A* returns Dijkstra's certified-optimal route and cost."""
    _, graph, agents = _world(scenario)
    reasoner = AffordanceReasoner()
    agent = agents[robot]

    astar = AStarPlanner(graph, reasoner).find_path(start, goal, agent)
    ucs = DijkstraPlanner(graph, reasoner).find_path(start, goal, agent)

    assert astar.is_feasible == ucs.is_feasible
    if astar.is_feasible:
        assert astar.total_cost == pytest.approx(ucs.total_cost)
        assert astar.nodes == ucs.nodes
    else:
        assert astar.total_cost == math.inf
        assert ucs.total_cost == math.inf


@pytest.mark.parametrize("scenario,robot,start,goal", BENCHMARK, ids=TASK_IDS)
def test_explanations_are_planner_independent(scenario, robot, start, goal):
    """Swapping the planner leaves route, cost and recommendations unchanged."""
    onto_a, graph_a, agents_a = _world(scenario)
    onto_b, graph_b, agents_b = _world(scenario)

    path_a, exp_a = PaceNavigator(
        onto_a, graph_a, planner_cls=AStarPlanner
    ).navigate(start, goal, agents_a[robot], k_alternatives=3)
    path_b, exp_b = PaceNavigator(
        onto_b, graph_b, planner_cls=DijkstraPlanner
    ).navigate(start, goal, agents_b[robot], k_alternatives=3)

    assert path_a.nodes == path_b.nodes
    assert path_a.is_feasible == path_b.is_feasible
    if path_a.is_feasible:
        assert path_a.total_cost == pytest.approx(path_b.total_cost)

    assert (exp_a is None) == (exp_b is None)
    if exp_a is not None:
        assert exp_a.recommendations == exp_b.recommendations


def test_the_plant_tells_the_four_robots_apart():
    """One question, four answers: which robots reach shipping from the dock."""
    _, graph, agents = _world("plant")
    reasoner = AffordanceReasoner()
    feasible = {name: AStarPlanner(graph, reasoner).find_path("dock", "shipping", a).is_feasible
                for name, a in agents.items()}
    assert feasible == {"tote_amr": False, "forklift_agv": False,
                        "mobile_manipulator": True, "yard_ugv": False}


def test_default_heuristic_scale_is_admissible():
    """The default scale is the largest discount an edge can receive.

    Any smaller multiplier would still be admissible but weaker; anything above
    the emergency discount could overestimate on a discounted edge.
    """
    _, graph, _ = _world("plant")
    reasoner = AffordanceReasoner()
    planner = AStarPlanner(graph, reasoner)

    assert planner.heuristic_scale == reasoner.config.emergency_discount
    assert planner.heuristic_scale < 1.0
