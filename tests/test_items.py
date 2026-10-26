"""A mission that ends with picking something up is one more leg of the route.

The item is the space individual of a virtual passage from its zone to the
item; its affordances are reach, grasp and lift.  Everything else - the
planner, the ground truth, the baselines, the counterfactual search - is
unchanged, which is the point: one machinery explains navigation and
manipulation failures alike, and tells the operator which is which.
"""

from __future__ import annotations

import math

from pace import AStarPlanner, NavEdge, NavigationGraph, NavNode
from pace.affordance import AffordanceReasoner
from pace.counterfactual import CounterfactualEngine
from pace.domain import AffordanceType
from pace.evaluation import candidate_space, compute_ground_truth

CLEAR = dict(width=3.0, height=4.0, surface_type="smooth", illumination=1.0, is_accessible=True)
ARM = dict(has_arm=True, reach_height=1.35, gripper_opening=0.08, payload=3.0)


def _world(onto, item_props):
    g = NavigationGraph()
    g.add_node(NavNode("s", (0.0, 0.0)))
    g.add_node(NavNode("cell", (10.0, 0.0)))
    g.add_node(NavNode("pick", (10.0, 2.0)))
    g.add_edge(NavEdge("s", "cell", onto.create("seg", "Corridor", **CLEAR), distance=10.0))
    item = onto.create("module", "Item", located_at="cell", **item_props)
    g.add_edge(NavEdge("cell", "pick", item, distance=2.0), bidirectional=False)
    r = AffordanceReasoner()
    return g, r, AStarPlanner(g, r), item


def test_item_rules_are_the_manipulation_ones_only(onto, make_agent):
    reasoner = AffordanceReasoner()
    arm = make_agent("arm", **ARM)
    item = onto.create("m", "Item", item_height=0.8, item_width=0.03, item_mass=2.0)
    res = reasoner.compute(item, arm)
    assert set(res.affordances) == {AffordanceType.REACHABLE, AffordanceType.GRASPABLE, AffordanceType.LIFTABLE}
    assert AffordanceType.TRAVERSABLE not in res.affordances       # space rules do not run on items
    cost, _ = reasoner.navigation_cost(item, arm, base_distance=2.0)
    assert cost == 2.0 + reasoner.config.handling_cost


def test_no_arm_means_no_repair(onto, make_agent):
    """Nothing about the plant makes an armless robot able to pick."""
    g, r, planner, item = _world(onto, dict(item_height=0.8, item_width=0.03, item_mass=2.0))
    tote = make_agent("tote", has_arm=False, can_open_doors=False)
    assert not planner.find_path("s", "pick", tote).is_feasible
    engine = CounterfactualEngine(r, planner, onto.individuals)
    cf = engine.explain_why_not(planner.find_path("s", "pick", tote), ["s", "cell", "pick"], tote)
    assert cf.changes == []
    assert compute_ground_truth(g, tote, "s", "pick", r) == set()
    # the change space still offers the restock (it is a property of the plant,
    # not of the robot); no structural item change is offered to an armless robot
    assert not [c for c in candidate_space(g, tote, r) if c[0].name == "module" and c[1] != "item_height"]


def test_out_of_reach_is_an_actionable_restock(onto, make_agent):
    g, r, planner, item = _world(onto, dict(item_height=1.65, item_width=0.03, item_mass=2.4))
    arm = make_agent("arm", **ARM)
    actual = planner.find_path("s", "pick", arm)
    assert not actual.is_feasible
    engine = CounterfactualEngine(r, planner, onto.individuals)
    cf = engine.explain_why_not(actual, ["s", "cell", "pick"], arm)
    (c,) = cf.changes
    assert (c.individual_name, c.property_name, c.claim) == ("module", "item_height", "repair")
    assert c.counterfactual_value == 1.25 and c.is_actionable() and c.effort() == "low"
    assert math.isfinite(cf.cf_cost)
    assert ("module", "item_height") in compute_ground_truth(g, arm, "s", "pick", r)


def test_too_heavy_and_too_wide_are_structural(onto, make_agent):
    g, r, planner, item = _world(onto, dict(item_height=0.45, item_width=0.12, item_mass=24.0))
    arm = make_agent("arm", **ARM)
    actual = planner.find_path("s", "pick", arm)
    engine = CounterfactualEngine(r, planner, onto.individuals)
    cf = engine.explain_why_not(actual, ["s", "cell", "pick"], arm)
    props = {c.property_name: c for c in cf.changes}
    assert set(props) == {"item_width", "item_mass"}
    assert all(not c.is_actionable() and c.effort() == "high" and c.claim == "repair" for c in cf.changes)
    assert compute_ground_truth(g, arm, "s", "pick", r) == set()     # no single actionable change
    space = {(ind.name, prop): val for ind, prop, val in candidate_space(g, arm, r)}
    assert space[("module", "item_mass")] == 3.0 and space[("module", "item_width")] == 0.08


def test_navigation_and_manipulation_in_one_answer(onto, make_agent):
    """A closed door on the way and a module on the top shelf: two repairs, one list."""
    g = NavigationGraph()
    for nid, pos in (("s", (0.0, 0.0)), ("cell", (10.0, 0.0)), ("pick", (10.0, 2.0))):
        g.add_node(NavNode(nid, pos))
    g.add_edge(NavEdge("s", "cell", onto.create("door", "Doorway", **dict(CLEAR, door_state="closed")), distance=10.0))
    item = onto.create("module", "Item", item_height=1.65, item_width=0.03, item_mass=2.4, located_at="cell")
    g.add_edge(NavEdge("cell", "pick", item, distance=2.0), bidirectional=False)
    r = AffordanceReasoner()
    planner = AStarPlanner(g, r)
    arm = make_agent("arm", can_open_doors=False, **ARM)
    engine = CounterfactualEngine(r, planner, onto.individuals)
    cf = engine.explain_why_not(planner.find_path("s", "pick", arm), ["s", "cell", "pick"], arm)
    assert {(c.individual_name, c.property_name) for c in cf.changes} == {("door", "door_state"), ("module", "item_height")}
    assert all(c.claim == "repair" and c.is_actionable() for c in cf.changes)
    assert math.isfinite(cf.cf_cost)
