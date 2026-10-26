"""Staged goods narrow a passage without changing the building.

``blocked_width`` is the width temporary storage takes from a passage. It is
the commonest reason an AMR stops in an intralogistics hall, and it is
actionable: somebody moves the pallets. The rules read the *free* width
(structural minus staged); the candidate generator has to tell the two terms
apart, proposing to clear the staged goods when there are any and to widen
only when the passage is too narrow even empty.
"""

from __future__ import annotations

import math

from pace import AStarPlanner, NavEdge, NavigationGraph, NavNode
from pace.affordance import AffordanceReasoner
from pace.config import NavCostConfig
from pace.counterfactual import CounterfactualEngine, PropertyChange
from pace.domain import AffordanceType
from pace.evaluation import candidate_space, compute_ground_truth
from pace.explanation import _CF_PROPERTY_TEMPLATES

CLEAR = dict(width=3.0, height=4.0, surface_type="smooth", illumination=1.0,
             is_accessible=True)
# needs 1.2 + 2 * 0.25 = 1.7 m
PALLET = dict(robot_width=1.2, robot_height=2.2, min_clearance=0.25, max_slope_angle=3.0,
              mobility_type="wheeled", can_open_doors=False, has_arm=False)


def test_free_width_is_structural_minus_staged():
    cfg = NavCostConfig()
    e = dict(width=3.0, blocked_width=1.4)
    assert cfg.free_width(e) == 1.6
    assert not cfg.fits_width(e, PALLET)
    assert cfg.fits_width_if_cleared(e, PALLET)
    assert cfg.fits_width(dict(width=3.0), PALLET)


def test_blocking_reason_names_the_cause(onto, make_agent, make_space):
    """Narrowed by storage and too narrow as built are different reasons."""
    reasoner = AffordanceReasoner()
    pallet = make_agent("pallet", **PALLET)
    narrowed = reasoner.compute(make_space(width=3.0, blocked_width=1.4), pallet)
    built = reasoner.compute(make_space(width=1.5), pallet)
    assert AffordanceType.PASSABLE in narrowed.blocked_affordances
    assert AffordanceType.PASSABLE in built.blocked_affordances
    assert "staged" in narrowed.blocking_reasons[AffordanceType.PASSABLE]
    assert "staged" not in built.blocking_reasons[AffordanceType.PASSABLE]
    # a robot the free width admits is not blocked at all
    tote = make_agent("tote", robot_width=0.6, min_clearance=0.1)
    assert AffordanceType.PASSABLE in reasoner.compute(
        make_space(width=3.0, blocked_width=1.4), tote).affordances


def _world(onto, robot, direct_props):
    g = NavigationGraph()
    for nid, pos in [("s", (0.0, 0.0)), ("via", (6.0, 6.0)), ("g", (12.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    direct = onto.create("seg_direct", "Corridor", **direct_props)
    g.add_edge(NavEdge("s", "g", direct, distance=10.0))
    g.add_edge(NavEdge("s", "via", onto.create("seg_sv", "Corridor", **CLEAR), distance=12.0))
    g.add_edge(NavEdge("via", "g", onto.create("seg_vg", "Corridor", **CLEAR), distance=12.0))
    r = AffordanceReasoner()
    planner = AStarPlanner(g, r)
    return g, r, CounterfactualEngine(r, planner, onto.individuals), planner.find_path("s", "g", robot)


def test_clearing_is_proposed_alone_when_it_suffices(onto, make_agent):
    pallet = make_agent("pallet", **PALLET)
    _g, _r, engine, actual = _world(onto, pallet, dict(CLEAR, width=3.0, blocked_width=1.4))
    cf = engine.explain_why_not(actual, ["s", "g"], pallet)
    props = {c.property_name: c for c in cf.changes}
    assert set(props) == {"blocked_width"}
    change = props["blocked_width"]
    assert change.counterfactual_value == 0.0
    assert change.claim == "repair"
    assert change.is_actionable() and change.effort() == "low"
    assert math.isfinite(cf.cf_cost)


def test_clearing_and_widening_are_both_proposed_when_both_are_needed(onto, make_agent):
    """1.6 m of aisle with 0.5 m staged: empty it is still too narrow for 1.7 m."""
    pallet = make_agent("pallet", **PALLET)
    _g, _r, engine, actual = _world(onto, pallet, dict(CLEAR, width=1.6, blocked_width=0.5))
    cf = engine.explain_why_not(actual, ["s", "g"], pallet)
    props = {c.property_name: c for c in cf.changes}
    assert set(props) == {"blocked_width", "width"}
    assert props["width"].counterfactual_value == 1.8      # required + margin, once cleared
    assert props["blocked_width"].claim == props["width"].claim == "repair"
    assert math.isfinite(cf.cf_cost)
    # the exhaustive search needs both too: neither alone opens the segment
    engine.search = "min_cardinality"
    cf2 = engine.explain_why_not(actual, ["s", "g"], pallet)
    assert {c.property_name for c in cf2.changes} == {"blocked_width", "width"}


def test_ground_truth_and_change_space_see_the_staged_goods(onto, make_agent):
    pallet = make_agent("pallet", **PALLET)
    g, r, _engine, _actual = _world(onto, pallet, dict(CLEAR, width=3.0, blocked_width=1.4))
    truth = compute_ground_truth(g, pallet, "s", "g", r)
    assert ("seg_direct", "blocked_width") in truth
    space = {(ind.name, prop): val for ind, prop, val in candidate_space(g, pallet, r)}
    assert space[("seg_direct", "blocked_width")] == 0.0
    # the single structural change offered to the baselines works on its own:
    # it has to clear the robot past what stays staged
    assert space[("seg_direct", "width")] == 3.2


def test_recommendation_template_renders():
    text = _CF_PROPERTY_TEMPLATES["blocked_width"].format(
        entity="seg_disA_disB", cf_value=0.0, orig_value=1.4)
    assert "1.4" in text and "seg_disA_disB" in text
    assert PropertyChange("seg", "blocked_width", 1.4, 0.0).effort() == "low"
