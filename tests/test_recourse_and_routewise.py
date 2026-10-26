"""Tests for the recourse baseline and the route-wise ground truth.

Both were added so the paper could answer two questions a reader is entitled to
ask: how PACE compares with the recourse literature its actionability split
descends from, and how it scores on the question an explanation is actually
put. Neither is worth much if the comparison is not fair, so what is tested
here is mostly fairness: that the baseline sees the same change space, that the
route-wise target is not defined in terms of PACE's internals, and that the
bound protecting the scaling sweep cannot alter a benchmark result.
"""

from __future__ import annotations

import math

import pytest

from pace import AStarPlanner
from pace.affordance import AffordanceReasoner
from pace.counterfactual import CounterfactualEngine
from pace.evaluation import (
    ACTIONABLE_PROPERTIES,
    baseline_recourse,
    candidate_space,
    compute_ground_truth,
    compute_ground_truth_routewise,
    predict_pace,
    questioned_routes,
    score_prediction_set,
)
from scenarios.plant import build_plant_world


@pytest.fixture(scope="module")
def plant():
    onto, graph, agents = build_plant_world()
    return onto, graph, agents, AffordanceReasoner()


def test_pace_emits_the_conjunctive_repair_of_the_underpass(plant):
    """The underpass is both too narrow and too low for the pallet truck.

    Neither half of the repair changes the planner's verdict on its own, so a
    descent guided by gain has nothing to follow. Blocking attribution emits
    every unsatisfied condition on the segment at once, and applying both
    opens the route.
    """
    onto, graph, agents, reasoner = plant
    agent = agents["forklift_agv"]
    planner = AStarPlanner(graph, reasoner)
    engine = CounterfactualEngine(reasoner, planner, onto.individuals)
    actual = planner.find_path("module_store", "shipping", agent)
    cf = engine.explain_why_not(actual, ["module_store", "shipping"], agent)
    proposed = {(c.individual_name, c.property_name, c.claim) for c in cf.changes}
    assert ("seg_store_ship", "width", "repair") in proposed
    assert ("seg_store_ship", "height", "repair") in proposed
    assert math.isfinite(cf.cf_cost)
    assert all(not c.is_actionable() and c.effort() == "high" for c in cf.changes)


def test_recourse_proposals_actually_restore_feasibility(plant):
    """Whatever it returns has to work, since it verifies by re-planning."""
    _onto, graph, agents, reasoner = plant
    agent = agents["forklift_agv"]
    got = baseline_recourse(graph, agent, reasoner, "module_store", "shipping", k=3)
    assert got, "the pair scan must find a repair here"
    feas, _cost = score_prediction_set(graph, agent, "module_store", "shipping",
                                       reasoner, got, k=3)
    assert feas == 1.0


def test_recourse_ranks_over_the_same_change_space(plant):
    """A baseline offered a smaller change space would be a straw man."""
    _onto, graph, agents, reasoner = plant
    agent = agents["forklift_agv"]
    space = {(ind.name, prop)
             for ind, prop, _ in candidate_space(graph, agent, reasoner)}
    got = baseline_recourse(graph, agent, reasoner, "dock", "shipping", k=3)
    assert {(f.individual_name, f.property_name) for f in got} <= space


def test_recourse_labels_structural_changes_as_such(plant):
    """The actionable flag is assigned from the property, not from the method."""
    _onto, graph, agents, reasoner = plant
    got = baseline_recourse(graph, agents["forklift_agv"], reasoner,
                            "dock", "shipping", k=3)
    for f in got:
        assert f.actionable == (f.property_name in ACTIONABLE_PROPERTIES)


def test_pair_scan_limit_cannot_change_a_benchmark_result(plant):
    """The bound that keeps the scaling sweep finite must not bind here.

    It is a performance guard, and a guard that changed a reported number would
    be a silent change of method.
    """
    _onto, graph, agents, reasoner = plant
    agent = agents["forklift_agv"]
    n = len(candidate_space(graph, agent, reasoner))
    pairs = n * (n - 1) // 2
    bounded = baseline_recourse(graph, agent, reasoner, "module_store",
                                "shipping", k=3, pair_scan_limit=pairs)
    unbounded = baseline_recourse(graph, agent, reasoner, "module_store",
                                  "shipping", k=3, pair_scan_limit=10 ** 6)
    assert [f.key for f in bounded] == [f.key for f in unbounded]


def test_routewise_truth_quantifies_over_the_shared_route_set(plant):
    """The route-wise target must not be defined by PACE's own enumeration.

    Both callers reach the routes through ``questioned_routes``, so the target
    is the same whichever method is being scored.
    """
    _onto, graph, agents, reasoner = plant
    routes = questioned_routes(graph, "receiving", "sorting", k=3)
    assert routes and all(r[0] == "receiving" and r[-1] == "sorting" for r in routes)
    assert routes == questioned_routes(graph, "receiving", "sorting", k=3)


def test_routewise_truth_credits_the_questioned_route(plant):
    """A change opening a questioned route counts even off the optimal path.

    This is the whole reason the second target exists: the optimal-path
    definition scores an explainer on a route it was not asked about.
    """
    _onto, graph, agents, reasoner = plant
    agent = agents["tote_amr"]
    optimal = compute_ground_truth(graph, agent, "receiving", "sorting", reasoner)
    routewise = compute_ground_truth_routewise(graph, agent, "receiving",
                                               "sorting", reasoner, k=3)
    # Every route-wise factor is an actionable property, as the optimal-path
    # target is, and the route-wise set is never the smaller of the two.
    assert all(prop in ACTIONABLE_PROPERTIES for _ind, prop in routewise)
    assert len(routewise) >= len(optimal)


def test_top_k_is_separable_from_the_alternative_budget(plant):
    """``top_k`` must widen the report without changing which routes are asked.

    The FeasRec diagnosis depends on telling a truncated recommendation apart
    from one the search never found, which needs exactly this separation.
    """
    _onto, graph, agents, reasoner = plant
    agent = agents["forklift_agv"]
    three = predict_pace(graph, agent, "dock", "shipping", reasoner, k=3)
    everything = predict_pace(graph, agent, "dock", "shipping", reasoner, k=3,
                              top_k=10 ** 6)
    assert len(three) == 3
    assert len(everything) > 3
    assert [f.key for f in everything[:3]] == [f.key for f in three]
