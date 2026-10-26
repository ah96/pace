"""Candidates read off the soft terms of the cost function, not only off blockers.

A blocker-driven generator is silent about a segment the robot can already
cross, which is exactly the segment a "why not this route?" question is often
about: the route is open, it simply costs more than the one taken. These tests
pin the three properties that make the generator worth having -- it finds those
segments, it does not propose savings too small to be credited, and it still
refuses to report a set that does nothing for the questioned route.

Every world below is built so the two-hop detour is the *chosen* path and the
direct edge is the one questioned; otherwise the engine short-circuits on an
alternative that is already preferred and the generator is never reached.
"""

from __future__ import annotations

import math

from pace import AStarPlanner, NavEdge, NavigationGraph, NavNode
from pace.affordance import AffordanceReasoner
from pace.counterfactual import CounterfactualEngine

CLEAR = dict(width=2.0, height=2.5, surface_type="smooth",
             illumination=1.0, is_accessible=True)

#: The direct edge is 10 m of bare distance; a detour half-leg shorter than this
#: makes the detour cheaper and so the one A* takes.
DIRECT_M = 10.0


def _world(onto, direct_props, detour_half, attribute_costs=True):
    g = NavigationGraph()
    for nid, pos in [("s", (0.0, 0.0)), ("via", (5.0, 5.0)), ("g", (10.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    g.add_edge(NavEdge("s", "g", onto.create("seg_direct", "Corridor",
                                             **direct_props), distance=DIRECT_M))
    g.add_edge(NavEdge("s", "via", onto.create("seg_sv", "Corridor", **CLEAR),
                       distance=detour_half))
    g.add_edge(NavEdge("via", "g", onto.create("seg_vg", "Corridor", **CLEAR),
                       distance=detour_half))
    reasoner = AffordanceReasoner()
    planner = AStarPlanner(g, reasoner)
    engine = CounterfactualEngine(reasoner, planner, onto.individuals,
                                  attribute_costs=attribute_costs)
    return engine, planner


def _question_direct(onto, direct_props, detour_half, attribute_costs=True):
    """Ask why the direct edge was not taken, and return the counterfactual."""
    engine, planner = _world(onto, direct_props, detour_half, attribute_costs)
    actual = planner.find_path("s", "g", robot_of(onto))
    assert actual.nodes == ["s", "via", "g"], "the detour must be the chosen path"
    return engine.explain_why_not(actual, ["s", "g"], robot_of(onto)), planner


def robot_of(onto):
    return onto.individuals["bot"]


def test_openable_door_is_found_though_nothing_blocks(onto, make_agent):
    """The case a blocker-driven generator misses entirely.

    The robot can open the door, so the segment is passable and no rule blocks
    it; it merely charges the 2.5 door-opening penalty.
    """
    make_agent("bot", can_open_doors=True)
    cf, _ = _question_direct(onto, dict(CLEAR, door_state="closed"), 5.5)
    assert "door_state" in {c.property_name for c in cf.changes}


def test_blockers_only_ablation_is_silent_on_the_same_case(onto, make_agent):
    make_agent("bot", can_open_doors=True)
    cf, _ = _question_direct(onto, dict(CLEAR, door_state="closed"), 5.5,
                             attribute_costs=False)
    assert "door_state" not in {c.property_name for c in cf.changes}


def test_saving_below_epsilon_is_not_proposed(onto, make_agent):
    """A carpet costs 0.3, under the 1.0 at which a saving is credited."""
    make_agent("bot")
    cf, _ = _question_direct(onto, dict(CLEAR, surface_type="carpeted"), 4.9)
    assert "surface_type" not in {c.property_name for c in cf.changes}


def test_wet_surface_clears_the_threshold(onto, make_agent):
    """A wet floor costs 1.4, so resurfacing is worth proposing."""
    make_agent("bot")
    cf, _ = _question_direct(onto, dict(CLEAR, surface_type="wet"), 5.5)
    assert "surface_type" in {c.property_name for c in cf.changes}


def test_nothing_is_reported_when_the_route_cannot_be_improved(onto, make_agent):
    """The gate that keeps PACE quiet where nothing needs changing."""
    make_agent("bot")
    cf, _ = _question_direct(onto, dict(CLEAR), 4.5)
    assert cf.changes == []


def test_reported_set_actually_improves_the_questioned_route(onto, make_agent):
    """Eq. (4) restated as a test: what comes back has to move that route."""
    make_agent("bot", can_open_doors=True)
    props = dict(CLEAR, door_state="closed", crowd_density=0.9)
    cf, planner = _question_direct(onto, props, 5.5)
    before = planner.evaluate_sequence(["s", "g"], robot_of(onto))

    assert {c.property_name for c in cf.changes} == {"door_state", "crowd_density"}
    assert math.isfinite(cf.cf_cost)
    assert before.total_cost - cf.cf_cost > 1.0


def test_savings_on_the_chosen_route_are_verified_and_ordered(onto, make_agent):
    """A deviation is also 'why is this way so slow?': the terms on the route
    the robot takes are attributed, each verified by re-evaluating the route."""
    from pace import AStarPlanner, NavEdge, NavigationGraph, NavNode
    from pace.affordance import AffordanceReasoner
    from pace.counterfactual import CounterfactualEngine
    g = NavigationGraph()
    for nid, pos in (("s", (0.0, 0.0)), ("m", (10.0, 0.0)), ("g", (20.0, 0.0))):
        g.add_node(NavNode(nid, pos))
    crowded = onto.create("seg_crowd", "Corridor", width=3.0, height=4.0, surface_type="smooth",
                          illumination=1.0, is_accessible=True, crowd_density=0.7)
    dark = onto.create("seg_dark", "Corridor", width=3.0, height=4.0, surface_type="smooth",
                       illumination=0.1, is_accessible=True)
    g.add_edge(NavEdge("s", "m", crowded, distance=10.0))
    g.add_edge(NavEdge("m", "g", dark, distance=10.0))
    r = AffordanceReasoner()
    planner = AStarPlanner(g, r)
    robot = make_agent("bot")
    path = planner.find_path("s", "g", robot)
    engine = CounterfactualEngine(r, planner, onto.individuals)
    savings = engine.savings_on(path, robot)
    props = [(c.individual_name, c.property_name) for _, c in savings]
    assert ("seg_crowd", "crowd_density") in props and ("seg_dark", "illumination") in props
    assert [s for s, _ in savings] == sorted([s for s, _ in savings], reverse=True)
    # the crowd term (0.5 * 4.0 = 2.0) outweighs the darkness (1.8), and both are verified savings
    assert savings[0][1].property_name == "crowd_density"
    assert all(s > r.config.cf_epsilon for s, _ in savings)
    # the blocker-only ablation attributes nothing on a route the robot drives
    engine.attribute_costs = False
    assert engine.savings_on(path, robot) == []
