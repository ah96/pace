"""
pace/evaluation.py
============================
Evaluation harness for PACE semantic navigation scenarios.

Ground truth definition
-----------------------
For a given (scenario, agent, start, goal) tuple:
  1. Run the planner → get actual_path (may be infeasible).
  2. For every space individual and every ACTIONABLE property:
       Apply the counterfactual value to a clone → re-evaluate.
       A (space, property) pair is ground-truth iff:
         - Restores feasibility (inf → finite cost), OR
         - Reduces actual path cost by > epsilon_cost.
  3. Ground truth = set of (individual_name, property_name) pairs.

Prediction (PACE counterfactuals)
----------------------------------
Uses a unit-weight (topological) graph to enumerate geometrically-short
candidates including currently-blocked routes, then explains each with
CounterfactualEngine.  Returns top-k PropertyChanges ranked by utility.

Baselines
---------
  random       - random shuffle of actionable change candidates
  distance     - rank by graph-edge distance to start node
  semantic     - rank by blocking-affordance severity
  occupancy    - rank by overlap with shortest path edges
  unconstrained - brute-force cost improvement ignoring actionability
  recourse     - DiCE-style diverse counterfactual search with the planner
                 standing in for the classifier (see baseline_recourse)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .affordance import AffordanceReasoner
from .config import NavCostConfig
from .counterfactual import CounterfactualEngine, PropertyChange
from .domain import AffordanceType, DoorState
from .navigation import AStarPlanner, NavPath, NavigationGraph
from .ontology import OntologyIndividual


# ---------------------------------------------------------------------------
# Prediction record: one ranked (individual, property) suggestion from a method.
# ---------------------------------------------------------------------------

@dataclass
class ExplanationFactor:
    """A ranked explanation factor in the semantic world model."""
    individual_name: str
    property_name:   str
    change:          PropertyChange
    utility:         float
    actionable:      bool   # True if change is actionable (not structural)
    feasible_recovery: bool = False
    cost_improvement:  float = 0.0

    @property
    def key(self) -> Tuple[str, str]:
        return (self.individual_name, self.property_name)


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------

ACTIONABLE_PROPERTIES = {
    "door_state", "is_accessible", "is_hazardous", "restricted",
    "blocked_width", "obstacle_density", "crowd_density", "illumination",
    "item_height",
}

# Counterfactual values to try for each actionable property
_CF_VALUES: Dict[str, object] = {
    "door_state":       DoorState.OPEN.value,
    "is_accessible":    True,
    "is_hazardous":     False,
    "restricted":       False,
    "blocked_width":    0.0,
    "obstacle_density": 0.0,
    "crowd_density":    0.2,
    "illumination":     0.8,
    "item_height":      0.3,     # restocked on a cart or the lowest shelf
}


def compute_ground_truth(
    graph:         NavigationGraph,
    agent:         OntologyIndividual,
    start:         str,
    goal:          str,
    reasoner:      AffordanceReasoner,
    epsilon_cost:  float = 1.0,
) -> Set[Tuple[str, str]]:
    """
    Compute operational ground-truth (individual, property) pairs.

    A pair is ground-truth iff applying the actionable change to the
    corresponding space individual restores feasibility or improves
    cost by more than epsilon_cost.
    """
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)

    truth: Set[Tuple[str, str]] = set()

    for ind in list(graph.all_space_individuals()):  # materialise - graph mutates during sweep
        for prop, cf_val in _CF_VALUES.items():
            orig_val = ind.get(prop)
            if orig_val is None:
                continue
            if orig_val == cf_val:
                continue

            # Apply change to a clone (never mutate the live individual).
            # clone() appends "_cf" to the name - restore must use clone.name.
            clone = ind.clone()
            clone.set(prop, cf_val)

            _swap_individual(graph, ind.name, clone)
            result = planner.find_path(start, goal, agent)
            _swap_individual(graph, clone.name, ind)  # restore: search by clone's name

            restores = (not baseline.is_feasible) and result.is_feasible
            improves = (
                baseline.is_feasible
                and result.is_feasible
                and (baseline.total_cost - result.total_cost) > epsilon_cost
            )
            if restores or improves:
                truth.add((ind.name, prop))

    return truth


def _swap_individual(
    graph:    NavigationGraph,
    name:     str,
    new_ind:  OntologyIndividual,
) -> None:
    """Replace every edge's space_individual that has the given name in-place."""
    for edge_list in graph.edges.values():
        for edge in edge_list:
            if edge.space_individual.name == name:
                edge.space_individual = new_ind


# ---------------------------------------------------------------------------
# The routes a question can be asked about
# ---------------------------------------------------------------------------

def questioned_routes(
    graph: NavigationGraph,
    start: str,
    goal:  str,
    k:     int = 3,
) -> List[List[str]]:
    """
    The candidate routes any method is asked to explain, as node sequences.

    Enumerated on the unit-weight (topological) graph so that currently blocked
    routes appear alongside traversable ones: a route that costs infinity is
    exactly the one a user is most likely to ask about, and an affordance-
    weighted enumeration would never surface it.

    This lives outside :func:`predict_pace` because the route-wise ground truth
    of :func:`compute_ground_truth_routewise` has to quantify over the same
    routes. Deriving both from one function is what stops the metric from being
    defined in terms of PACE's own internals.
    """
    import networkx as nx

    ug = nx.DiGraph()
    for nid in graph.nodes:
        ug.add_node(nid)
    for fid, elist in graph.edges.items():
        for edge in elist:
            if edge.to_id in graph.nodes:
                ug.add_edge(fid, edge.to_id, weight=1)

    seqs: List[List[str]] = []
    try:
        for seq in nx.shortest_simple_paths(ug, start, goal, weight="weight"):
            seqs.append(seq)
            if len(seqs) >= k + 2:
                break
    except Exception:
        pass
    return seqs


def compute_ground_truth_routewise(
    graph:        NavigationGraph,
    agent:        OntologyIndividual,
    start:        str,
    goal:         str,
    reasoner:     AffordanceReasoner,
    k:            int = 3,
    epsilon_cost: float = 1.0,
) -> Set[Tuple[str, str]]:
    """
    Ground truth for the question an explanation actually answers.

    :func:`compute_ground_truth` credits a change only when it improves the
    *optimal* path from start to goal. That is the right target for a route
    optimiser and the wrong one for an explainer, which is asked why some
    particular alternative was not taken and answers about that route. A change
    that opens the questioned route earns nothing under the optimal-path
    definition if the optimum lies elsewhere.

    Here a pair ``(i, p)`` is ground truth iff applying the canonical value
    ``v_cf(p)`` to a clone of ``i`` makes *some* questioned route feasible, or
    cheaper by more than *epsilon_cost*. The routes are those of
    :func:`questioned_routes`, which is method-independent, so every method is
    scored against the same target.
    """
    planner = AStarPlanner(graph, reasoner)
    routes  = questioned_routes(graph, start, goal, k)
    if not routes:
        return set()

    before = [planner.evaluate_sequence(seq, agent) for seq in routes]

    truth: Set[Tuple[str, str]] = set()
    for ind in list(graph.all_space_individuals()):
        for prop, cf_val in _CF_VALUES.items():
            orig_val = ind.get(prop)
            if orig_val is None or orig_val == cf_val:
                continue

            clone = ind.clone()
            clone.set(prop, cf_val)
            _swap_individual(graph, ind.name, clone)
            after = [planner.evaluate_sequence(seq, agent) for seq in routes]
            _swap_individual(graph, clone.name, ind)

            for pre, post in zip(before, after):
                restores = (not pre.is_feasible) and post.is_feasible
                improves = (
                    pre.is_feasible and post.is_feasible
                    and (pre.total_cost - post.total_cost) > epsilon_cost
                )
                if restores or improves:
                    truth.add((ind.name, prop))
                    break

    return truth


# ---------------------------------------------------------------------------
# PaceNavigator → ranked ExplanationFactor list
# ---------------------------------------------------------------------------

def predict_pace(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    start:    str,
    goal:     str,
    reasoner: AffordanceReasoner,
    k:        int = 3,
    search:   str = "greedy",
    stats:    Optional[Dict[str, float]] = None,
    top_k:    Optional[int] = None,
    attribute_costs: bool = True,
    priority_order:  Optional[List[str]] = None,
    separate_claims: bool = True,
    prefer_only:     bool = True,
) -> List[ExplanationFactor]:
    """
    Run the PaceNavigator counterfactual engine and return top-k factors.

    Uses a unit-weight (topological) graph to enumerate geometrically-short
    candidate alternatives - including currently-blocked routes that cost=inf
    in the affordance graph.  This allows the engine to explain:

    (a) Why a BLOCKED route is infeasible and what minimal changes would
        open it (e.g., door closed, area restricted, corridor too narrow).
    (b) Why a REACHABLE but expensive alternative isn't preferred and what
        changes would reduce its cost (e.g., high crowd, poor illumination).

    Candidates are ranked by counterfactual utility: feasibility-recovery
    scores (inf → finite) beat cost-improvement scores.

    *search* selects the counterfactual search over the (identical) candidate
    pool: "greedy" is the priority-ordered search; "min_cardinality" is the
    exhaustive optimal reference.  Passing a dict as *stats* collects the mean
    prescribed change-set size per alternative, before top-k truncation - the
    quantity that reveals whether the greedy search over-prescribes.

    *k* sets both the alternative budget and how many factors come back. Pass
    *top_k* to separate the two, which is what tells a recommendation the budget
    truncated apart from one the search never found.

    *separate_claims* keeps repair and saving claims apart on infeasible tasks:
    the recommendation list scored here then contains only the repair claims,
    and the savings stay in the per-route counterfactuals as claims conditional
    on the route being opened. Passing False pools both into one ranked list,
    the earlier design, in which a genuine actionable saving on a route only
    construction can open reads as something an operator could do about the
    task. On feasible tasks the flag is inert - every claim is a saving there,
    and the questioned-route repairs (a blocked shortcut's door) remain
    recommendations either way.
    """
    top_k = k if top_k is None else top_k
    planner   = AStarPlanner(graph, reasoner)
    baseline  = planner.find_path(start, goal, agent)
    cf_engine = CounterfactualEngine(
        reasoner         = reasoner,
        planner          = planner,
        onto_individuals = {ind.name: ind for ind in graph.all_space_individuals()},
        search           = search,
        attribute_costs  = attribute_costs,
        priority_order   = priority_order,
    )

    candidate_seqs = questioned_routes(graph, start, goal, k)

    if not candidate_seqs:
        return []

    # Skip any sequence identical to the actual chosen path (already explained)
    actual_seq = tuple(baseline.nodes)
    alt_seqs   = [s for s in candidate_seqs if tuple(s) != actual_seq]

    # Evaluate each alternative in the real affordance-weighted world
    alt_paths = [planner.evaluate_sequence(seq, agent) for seq in alt_seqs]

    # explain_why_not(baseline, alt_nodes) answers:
    #   "What minimal changes would make this alternative preferable?"
    cfs: List = []
    seen_seqs: Set[tuple] = {actual_seq}
    for alt_path in alt_paths:
        seq_key = tuple(alt_path.nodes)
        if seq_key in seen_seqs:
            continue
        seen_seqs.add(seq_key)
        cf = cf_engine.explain_why_not(baseline, alt_path.nodes, agent)
        cfs.append(cf)

    # Collect and deduplicate PropertyChanges; assign utility scores
    seen:    Set[Tuple[str, str]] = set()
    factors: List[ExplanationFactor] = []

    for cf in cfs:
        # On a task the robot completes, a change is advice only if the route
        # it opens or cheapens would then be *preferred*: "why not that way?"
        # is answered in the per-route counterfactual either way, but a repair
        # of a route that stays worse than the one taken is not a
        # recommendation, and reporting it would be noise on a task nothing is
        # wrong with.
        if baseline.is_feasible and prefer_only and not (cf.cost_delta > 0):
            continue
        for change in cf.changes:
            # On an infeasible task a saving is a conditional claim about a
            # route that cannot yet be driven; it is not advice about the task.
            if separate_claims and not baseline.is_feasible and change.claim == "saving":
                continue
            key = (change.individual_name, change.property_name)
            if key in seen:
                continue
            seen.add(key)
            feasible_recovery = (not baseline.is_feasible) and cf.cf_cost < math.inf
            # Utility: large score for feasibility recovery, otherwise cost delta
            if feasible_recovery:
                utility = 1000.0 + max(0.0, cf.cost_delta if cf.cost_delta < math.inf else 0.0)
            else:
                utility = max(0.0, cf.cost_delta if cf.cost_delta < math.inf else 0.0)
            factors.append(ExplanationFactor(
                individual_name  = change.individual_name,
                property_name    = change.property_name,
                change           = change,
                utility          = utility,
                actionable       = change.is_actionable(),
                feasible_recovery= feasible_recovery,
                cost_improvement = max(0.0, cf.cost_delta if cf.cost_delta < math.inf else 0.0),
            ))

    # On a task the robot completes, the route it takes is questioned too:
    # what would make it cheaper is a saving on the optimal route, and it is
    # exactly what the deviation target credits.
    if baseline.is_feasible:
        for saving, change in cf_engine.savings_on(baseline, agent):
            key = (change.individual_name, change.property_name)
            if key in seen:
                continue
            seen.add(key)
            factors.append(ExplanationFactor(
                individual_name  = change.individual_name,
                property_name    = change.property_name,
                change           = change,
                utility          = saving,
                actionable       = change.is_actionable(),
                feasible_recovery= False,
                cost_improvement = saving,
            ))

    if stats is not None:
        sizes = [len(cf.changes) for cf in cfs]
        stats["mean_changeset_size"] = (sum(sizes) / len(sizes)) if sizes else 0.0
        stats["total_changes"]       = float(sum(sizes))

    factors.sort(key=lambda f: f.utility, reverse=True)
    return factors[:top_k]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

# Structural properties. Their counterfactual value depends on the robot, so
# unlike _CF_VALUES they cannot be a fixed table: the width a corridor needs is
# the width of the robot that has to fit through it. The targets are derived
# from the same NavCostConfig predicates the counterfactual engine uses, so the
# baselines are offered exactly the repairs PACE could propose.
STRUCTURAL_PROPERTIES = {"width", "height", "slope_angle", "item_width", "item_mass"}


def _structural_cf_value(
    prop:  str,
    ind:   OntologyIndividual,
    agent: OntologyIndividual,
    cfg:   NavCostConfig,
) -> Optional[float]:
    """Target value that would clear a structural blocker, or None if it is not blocking."""
    e, a = ind.properties, agent.properties
    if cfg.is_item(e):
        if not a.get("has_arm", False):
            return None                     # no change to the item helps an armless robot
        if prop == "item_width" and not cfg.fits_gripper(e, a):
            return round(a.get("gripper_opening", cfg.default_gripper_opening), 3)
        if prop == "item_mass" and not cfg.within_payload(e, a):
            return round(a.get("payload", cfg.default_payload), 2)
        return None
    if prop in ("item_width", "item_mass"):
        return None
    if prop == "width" and not cfg.fits_width(e, a):
        # A single structural change that repairs the segment on its own has
        # to clear the robot past whatever stays staged in the passage.
        return round(cfg.required_width(a) + cfg.blocked_width(e)
                     + cfg.cf_width_safety_margin, 2)
    if prop == "height" and ind.get("height") is not None and not cfg.fits_height(e, a):
        return round(cfg.required_height(a) + cfg.cf_height_safety_margin, 2)
    if prop == "slope_angle" and not cfg.slope_within_limit(e, a):
        return round(a.get("max_slope_angle", cfg.default_max_slope_angle)
                     - cfg.cf_slope_margin, 1)
    return None


def candidate_space(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
) -> List[Tuple[OntologyIndividual, str, object]]:
    """
    Every (individual, property, counterfactual value) a method may propose.

    This is the full change space: the actionable properties an operator can
    alter, plus the structural repairs that need construction. Baselines rank
    over the same space PACE searches, so a baseline that never recommends a
    structural fix is making a choice rather than obeying the harness.
    """
    cfg = reasoner.config
    out: List[Tuple[OntologyIndividual, str, object]] = []
    for ind in graph.all_space_individuals():
        for prop, cf_val in _CF_VALUES.items():
            orig = ind.get(prop)
            if orig is not None and orig != cf_val:
                out.append((ind, prop, cf_val))
        for prop in STRUCTURAL_PROPERTIES:
            cf_val = _structural_cf_value(prop, ind, agent, cfg)
            if cf_val is not None:
                out.append((ind, prop, cf_val))
    return out


def repairable(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    start:    str,
    goal:     str,
    reasoner: AffordanceReasoner,
) -> bool:
    """Whether *any* change to the plant makes the task feasible: the whole
    change space (actionable and structural) applied at once, then re-planned.
    False means the cause is on the robot's side - no arm for a pick, say -
    and the right answer is to say so rather than to recommend anything."""
    planner = AStarPlanner(graph, reasoner)
    if planner.find_path(start, goal, agent).is_feasible:
        return True
    clones: Dict[str, OntologyIndividual] = {}
    originals: Dict[str, OntologyIndividual] = {}
    for ind, prop, val in candidate_space(graph, agent, reasoner):
        if ind.name not in clones:
            originals[ind.name] = ind
            clones[ind.name] = ind.clone()
        clones[ind.name].set(prop, val)
    for name, clone in clones.items():
        _swap_individual(graph, name, clone)
    result = planner.find_path(start, goal, agent)
    for name, clone in clones.items():
        _swap_individual(graph, clone.name, originals[name])
    return result.is_feasible


def _all_actionable_candidates(
    graph:    NavigationGraph,
    reasoner: AffordanceReasoner,
    agent:    OntologyIndividual,
) -> List[Tuple[OntologyIndividual, str]]:
    """Backwards-compatible view of the full space as (individual, property) pairs."""
    return [(ind, prop) for ind, prop, _ in candidate_space(graph, agent, reasoner)]


def _factor(
    ind:     OntologyIndividual,
    prop:    str,
    cf_val:  object,
    utility: float,
) -> ExplanationFactor:
    """Build one ranked recommendation, labelled actionable or structural."""
    change = PropertyChange(
        individual_name=ind.name, property_name=prop,
        original_value=ind.get(prop), counterfactual_value=cf_val,
    )
    return ExplanationFactor(
        individual_name=ind.name, property_name=prop, change=change,
        utility=utility, actionable=prop in ACTIONABLE_PROPERTIES,
    )


def baseline_random(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    k:        int,
    seed:     int,
) -> List[ExplanationFactor]:
    rng = random.Random(seed)
    candidates = candidate_space(graph, agent, reasoner)
    rng.shuffle(candidates)
    return [_factor(ind, prop, val, utility=float(len(candidates) - i))
            for i, (ind, prop, val) in enumerate(candidates[:k])]


def baseline_distance(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    k:        int,
) -> List[ExplanationFactor]:
    """Rank candidates by minimum edge distance from start."""
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, list(graph.nodes.keys())[-1], agent)

    # BFS distance from start (in terms of graph hops)
    from collections import deque
    dist: Dict[str, float] = {start: 0.0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for edge in graph.neighbors(cur):
            if edge.to_id not in dist:
                dist[edge.to_id] = dist[cur] + edge.distance
                queue.append(edge.to_id)

    candidates = candidate_space(graph, agent, reasoner)

    def min_dist_to_start(ind: OntologyIndividual) -> float:
        for edge_list in graph.edges.values():
            for edge in edge_list:
                if edge.space_individual.name == ind.name:
                    d = min(
                        dist.get(edge.from_id, math.inf),
                        dist.get(edge.to_id,   math.inf),
                    )
                    return d
        return math.inf

    ranked = sorted(candidates, key=lambda x: min_dist_to_start(x[0]))
    return [_factor(ind, prop, val, utility=-min_dist_to_start(ind))
            for ind, prop, val in ranked[:k]]


def baseline_semantic(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    goal:     str,
    k:        int,
) -> List[ExplanationFactor]:
    """Rank by affordance-blocking severity on the shortest path."""
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)

    # Collect spaces on the actual path (or all if infeasible)
    if baseline.is_feasible:
        path_spaces = {e.space_individual.name for e in baseline.edges}
    else:
        path_spaces = {
            ind.name for ind in graph.all_space_individuals()
        }

    def score(ind: OntologyIndividual) -> float:
        af = reasoner.compute(ind, agent)
        # Penalise based on which hard-blocking affordances are absent
        s = 0.0
        if AffordanceType.TRAVERSABLE not in af.affordances:
            s += 4.0
        if AffordanceType.PASSABLE not in af.affordances:
            s += 3.0
        if AffordanceType.CLIMBABLE not in af.affordances:
            s += 2.0
        if AffordanceType.OPENABLE not in af.affordances and ind.get("door_state"):
            s += 1.5
        if ind.name in path_spaces:
            s += 2.0
        return s

    candidates = candidate_space(graph, agent, reasoner)
    ranked = sorted(candidates, key=lambda x: score(x[0]), reverse=True)
    return [_factor(ind, prop, val, utility=score(ind))
            for ind, prop, val in ranked[:k]]


def baseline_occupancy(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    goal:     str,
    k:        int,
) -> List[ExplanationFactor]:
    """Rank by overlap between the individual's edges and the shortest path."""
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)
    path_spaces = {e.space_individual.name for e in baseline.edges}

    def overlap(ind: OntologyIndividual) -> int:
        return 1 if ind.name in path_spaces else 0

    candidates = candidate_space(graph, agent, reasoner)
    ranked = sorted(candidates, key=lambda x: overlap(x[0]), reverse=True)
    return [_factor(ind, prop, val, utility=float(overlap(ind)))
            for ind, prop, val in ranked[:k]]


def baseline_unconstrained(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    goal:     str,
    k:        int,
) -> List[ExplanationFactor]:
    """Rank all candidates by raw cost improvement (no actionability filter)."""
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)

    candidates_raw = []
    for ind, prop, cf_val in candidate_space(graph, agent, reasoner):
            orig = ind.get(prop)

            clone = ind.clone()
            clone.set(prop, cf_val)
            _swap_individual(graph, ind.name, clone)
            result = planner.find_path(start, goal, agent)
            _swap_individual(graph, clone.name, ind)  # restore by clone.name

            restores = (not baseline.is_feasible) and result.is_feasible
            improvement = 1000.0 * float(restores)
            if baseline.is_feasible and result.is_feasible:
                improvement += max(0.0, baseline.total_cost - result.total_cost)

            change = PropertyChange(
                individual_name=ind.name, property_name=prop,
                original_value=orig, counterfactual_value=cf_val,
            )
            candidates_raw.append(ExplanationFactor(
                individual_name=ind.name, property_name=prop,
                change=change,
                utility=improvement,
                actionable=prop in ACTIONABLE_PROPERTIES,
                feasible_recovery=restores,
                cost_improvement=max(0.0, (baseline.total_cost - result.total_cost)
                                    if baseline.is_feasible and result.is_feasible else 0.0),
            ))

    candidates_raw.sort(key=lambda f: f.utility, reverse=True)
    return candidates_raw[:k]


def baseline_unconstrained_sets(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    goal:     str,
    k:        int,
    max_cardinality: int = 3,
) -> List[ExplanationFactor]:
    """Excuse search over change *sets*: exhaustive ascending-cardinality
    enumeration of the full change space, each set verified by re-planning.

    This is the fair form of the excuse baseline for conjunctive repairs, where
    :func:`baseline_unconstrained` is handicapped by construction: excuse
    generation in the literature is not restricted to single changes, so a
    comparison that restricts it cannot say whether PACE's advantage is the
    formulation or the handicap. The search embeds no domain insight -- no
    attribution, no restriction to the questioned routes -- and pays full price
    for it: every subset of the change space up to ``max_cardinality`` is a
    candidate, so the cost is O(|space|^cardinality) re-plans.

    Sets pay off only where no single change restores feasibility. On a
    feasible task this reduces to the single-change search (enumerating subsets
    for cost savings squares an already-quadratic bill for no new claim), and
    where no set within the budget restores the route it returns nothing,
    exactly as an excuse generator with a budget must.
    """
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)
    if baseline.is_feasible:
        return baseline_unconstrained(graph, agent, reasoner, start, goal, k)

    candidates = candidate_space(graph, agent, reasoner)
    if any_restoring_single(graph, agent, reasoner, start, goal, candidates,
                            planner):
        return baseline_unconstrained(graph, agent, reasoner, start, goal, k)

    def replan_with(combo) -> NavPath:
        # One clone per touched individual, all its mutations applied to it.
        originals: Dict[str, OntologyIndividual] = {}
        clones:    Dict[str, OntologyIndividual] = {}
        for ind, prop, val in combo:
            if ind.name not in clones:
                originals[ind.name] = ind
                clones[ind.name]    = ind.clone()
            clones[ind.name].set(prop, val)
        for name, clone in clones.items():
            _swap_individual(graph, name, clone)
        result = planner.find_path(start, goal, agent)
        for name, clone in clones.items():
            _swap_individual(graph, clone.name, originals[name])
        return result

    from itertools import combinations
    for cardinality in range(2, max_cardinality + 1):
        best, best_cost = None, math.inf
        for combo in combinations(candidates, cardinality):
            result = replan_with(combo)
            if result.is_feasible and result.total_cost < best_cost:
                best, best_cost = combo, result.total_cost
        if best is not None:
            return [_factor(ind, prop, val, utility=1000.0 - i)
                    for i, (ind, prop, val) in enumerate(best)]
    return []


def any_restoring_single(
    graph, agent, reasoner, start, goal, candidates, planner,
) -> bool:
    """Whether any single change in ``candidates`` makes the task planable."""
    for ind, prop, val in candidates:
        clone = ind.clone()
        clone.set(prop, val)
        _swap_individual(graph, ind.name, clone)
        result = planner.find_path(start, goal, agent)
        _swap_individual(graph, clone.name, ind)
        if result.is_feasible:
            return True
    return False


# ---------------------------------------------------------------------------
# Recourse baseline: the closest thing in the literature, adapted
# ---------------------------------------------------------------------------

#: Properties whose counterfactual distance is a flip rather than a magnitude.
#: A door is open or shut; there is no "slightly open" to be closer to.
_CATEGORICAL_PROPERTIES = {
    "door_state", "restricted", "is_accessible", "is_hazardous", "surface_type",
}

#: Floor on the MAD scale, so a property that happens to be constant across the
#: graph does not divide a distance by zero.
_MAD_FLOOR = 1e-3


def _property_scales(graph: NavigationGraph) -> Dict[str, float]:
    """
    Median absolute deviation of each numeric property over the space segments.

    Wachter et al. normalise each feature by its MAD so that a counterfactual
    distance is not dominated by whichever feature happens to be measured in
    large units. The segments of the graph play the role of their dataset.
    """
    values: Dict[str, List[float]] = {}
    for ind in graph.all_space_individuals():
        for prop, val in ind.properties.items():
            if prop in _CATEGORICAL_PROPERTIES or not isinstance(val, (int, float)):
                continue
            if isinstance(val, bool):
                continue
            values.setdefault(prop, []).append(float(val))

    scales: Dict[str, float] = {}
    for prop, vals in values.items():
        med = sorted(vals)[len(vals) // 2]
        devs = sorted(abs(v - med) for v in vals)
        scales[prop] = max(devs[len(devs) // 2], _MAD_FLOOR)
    return scales


def _proximity(prop: str, orig: object, new: object, scales: Dict[str, float]) -> float:
    """Normalised L1 distance between the actual and counterfactual value."""
    if prop in _CATEGORICAL_PROPERTIES or isinstance(orig, bool) or orig is None:
        return 1.0
    if isinstance(orig, (int, float)) and isinstance(new, (int, float)):
        return abs(float(new) - float(orig)) / scales.get(prop, _MAD_FLOOR)
    return 1.0


def baseline_recourse(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    goal:     str,
    k:        int,
    max_cardinality: int = 3,
    max_sets:        int = 6,
    pair_scan_limit: int = 60,
    epsilon:         float = 1.0,
) -> List[ExplanationFactor]:
    """
    Algorithmic recourse over the world model, with the planner as the classifier.

    This is the strongest baseline available without granting a method PACE's
    actionable/structural split, and it is the one a reader who knows the
    recourse literature will ask for. The construction follows DiCE: the vector
    of mutable segment properties is the input, the planner's verdict on the
    task stands in for the classifier output, and the search looks for *diverse*
    valid counterfactuals that are *proximate* under a MAD-normalised L1
    distance.

    Validity means what it means for the task: on an infeasible task a
    counterfactual is valid when the goal becomes reachable; on a feasible one
    when the optimal cost drops by more than the same epsilon the ground truth
    uses.

    Proximity is minimised by greedy forward selection scored on gain per unit
    of distance, which is the practical stand-in for DiCE's gradient descent -
    the world model is discrete and the planner is not differentiable, so no
    gradient exists to follow. Diversity comes from re-running the selection
    with the previously used (individual, property) pairs withheld, which is
    DiCE's diversity term applied to a discrete space.

    What it cannot do is say which of its proposals a person could carry out,
    since nothing in the recourse formulation distinguishes a door from a wall.
    That is the gap the actionability labels fill, and this baseline is here to
    show the size of it.
    """
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)
    scales   = _property_scales(graph)

    pool = candidate_space(graph, agent, reasoner)
    if not pool:
        return []

    by_name = {ind.name: ind for ind in graph.all_space_individuals()}

    def replan(selected: List[Tuple[OntologyIndividual, str, object]]) -> NavPath:
        """Cost of the optimal path once every selected change is applied to clones."""
        clones: Dict[str, OntologyIndividual] = {}
        for ind, prop, val in selected:
            if ind.name not in clones:
                clones[ind.name] = by_name[ind.name].clone()
            clones[ind.name].set(prop, val)
        for name, clone in clones.items():
            _swap_individual(graph, name, clone)
        result = planner.find_path(start, goal, agent)
        for name, clone in clones.items():
            _swap_individual(graph, clone.name, by_name[name])
        return result

    def valid(result: NavPath) -> bool:
        if not baseline.is_feasible:
            return result.is_feasible
        return (result.is_feasible
                and (baseline.total_cost - result.total_cost) > epsilon)

    def gain(result: NavPath) -> float:
        """How much closer to validity the world has moved."""
        if not baseline.is_feasible:
            return 1000.0 if result.is_feasible else 0.0
        if not result.is_feasible:
            return 0.0
        return max(0.0, baseline.total_cost - result.total_cost)

    def _dist(cand: Tuple[OntologyIndividual, str, object]) -> float:
        return max(_proximity(cand[1], cand[0].get(cand[1]), cand[2], scales), _MAD_FLOOR)

    def search(banned: Set[Tuple[str, str]]
               ) -> List[Tuple[OntologyIndividual, str, object, float]]:
        """One diverse counterfactual, nearest first under the proximity metric."""
        selected: List[Tuple[OntologyIndividual, str, object]] = []
        chosen:   List[Tuple[OntologyIndividual, str, object, float]] = []
        remaining = [c for c in pool if (c[0].name, c[1]) not in banned]
        best_gain = gain(baseline)

        # Greedy forward selection on progress bought per unit of distance moved.
        while remaining and len(selected) < max_cardinality:
            scored = []
            for cand in remaining:
                result = replan(selected + [cand])
                dist   = _dist(cand)
                scored.append((valid(result), gain(result) / dist,
                               gain(result), dist, cand))
            scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
            is_valid, _ratio, g, dist, cand = scored[0]
            if not is_valid and g <= best_gain:
                break                      # plateau: no single change buys anything
            selected.append(cand)
            chosen.append((cand[0], cand[1], cand[2], dist))
            remaining = [c for c in remaining
                         if (c[0].name, c[1]) != (cand[0].name, cand[1])]
            best_gain = max(best_gain, g)
            if is_valid:
                return chosen

        # A conjunctive repair defeats greedy by construction: neither half moves
        # the planner's verdict, so the plateau above ends the descent at the
        # first step. Fall back to scanning pairs in ascending total distance,
        # which is what a discrete recourse search does when the objective is
        # flat, and take the nearest valid one. Without this the baseline would
        # lose the conjunctive cases to its optimiser rather than to its
        # formulation, which is not the comparison this baseline is here to make.
        # The scan is quadratic in the pool, so it is restricted to the
        # *pair_scan_limit* nearest candidates. That costs nothing on the
        # benchmark, where the whole pool is smaller than the limit, and keeps
        # the scaling sweep finite on graphs where it is not. A proximity-
        # ordered search examines nearest-first anyway, so the bound truncates
        # the tail rather than changing what the method looks for.
        if not selected:
            avail = sorted([c for c in pool if (c[0].name, c[1]) not in banned],
                           key=_dist)[:pair_scan_limit]
            best: Optional[Tuple[float, list]] = None
            for i, first in enumerate(avail):
                for second in avail[i + 1:]:
                    if (first[0].name, first[1]) == (second[0].name, second[1]):
                        continue
                    total = _dist(first) + _dist(second)
                    if best is not None and total >= best[0]:
                        continue
                    if valid(replan([first, second])):
                        best = (total, [first, second])
            if best is not None:
                return [(c[0], c[1], c[2], _dist(c)) for c in best[1]]

        return chosen

    # Diverse counterfactuals: each search runs without the pairs already used.
    # *max_sets* bounds how many are looked for, which matters only for the
    # actionability-filtered variant, where k is effectively unbounded so that
    # filtering happens before truncation rather than after.
    factors: List[ExplanationFactor] = []
    banned:  Set[Tuple[str, str]] = set()
    for _ in range(max_sets):
        if len(factors) >= k:
            break
        found = search(banned)
        if not found:
            break
        found.sort(key=lambda f: f[3])     # nearest change first
        for ind, prop, val, dist in found:
            banned.add((ind.name, prop))
            if len(factors) < k:
                # Utility descends with distance, so the ranking the metrics see
                # is the proximity ordering recourse actually prescribes.
                factors.append(_factor(ind, prop, val, utility=1.0 / (1.0 + dist)))

    return factors


#: Suffix marking an actionability-filtered variant of a baseline, e.g.
#: ``semantic-act``. See :func:`run_baseline`.
ACTIONABLE_SUFFIX = "-act"

#: Effectively "rank the whole space", so filtering happens before truncation
#: rather than after. Filtering a top-k list would leave fewer than k
#: recommendations for reasons that have nothing to do with the method.
_ALL = 10 ** 6


def run_baseline(
    method:   str,
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    reasoner: AffordanceReasoner,
    start:    str,
    goal:     str,
    k:        int,
    seed:     int,
    epsilon:  float = 1.0,
) -> List[ExplanationFactor]:
    """Dispatch a baseline by name.

    A ``-act`` suffix runs the same ranking restricted to
    :data:`ACTIONABLE_PROPERTIES`. This is the ablation that separates two
    explanations of PACE's low actionable fraction on structural tasks: whether
    it comes from the search, or merely from being the only method allowed to
    know which properties a user can change. The filtered variant ranks the full space
    first and truncates after filtering, so it still returns k suggestions
    wherever k actionable candidates exist.
    """
    filtered = method.endswith(ACTIONABLE_SUFFIX)
    base = method[: -len(ACTIONABLE_SUFFIX)] if filtered else method
    want = _ALL if filtered else k

    if base == "random":
        out = baseline_random(graph, agent, reasoner, want, seed)
    elif base == "distance":
        out = baseline_distance(graph, agent, reasoner, start, want)
    elif base == "semantic":
        out = baseline_semantic(graph, agent, reasoner, start, goal, want)
    elif base == "occupancy":
        out = baseline_occupancy(graph, agent, reasoner, start, goal, want)
    elif base == "unconstrained":
        out = baseline_unconstrained(graph, agent, reasoner, start, goal, want)
    elif base == "unconstrained-sets":
        out = baseline_unconstrained_sets(graph, agent, reasoner, start, goal,
                                          want if want is not _ALL else 3)
    elif base == "recourse":
        out = baseline_recourse(graph, agent, reasoner, start, goal, want,
                               epsilon=epsilon)
    else:
        raise ValueError(f"Unknown baseline: {method}")

    if filtered:
        out = [f for f in out if f.actionable][:k]
    return out



# ---------------------------------------------------------------------------
# Outcome scoring, applied identically to every method
# ---------------------------------------------------------------------------

def score_prediction_set(
    graph:    NavigationGraph,
    agent:    OntologyIndividual,
    start:    str,
    goal:     str,
    reasoner: AffordanceReasoner,
    factors:  List["ExplanationFactor"],
    k:        int,
) -> Tuple[float, float]:
    """
    Apply a method's top-k recommendations together and measure what they achieve.

    Returns ``(feasibility_recovery, cost_improvement)``. Feasibility recovery is
    1.0 when the task was infeasible and the recommended set makes it planable.
    Cost improvement is how much the re-planned optimal path drops, and is 0.0
    when the task was already infeasible.

    Every method is scored through this one function. Scoring a method on what
    its own engine predicted, rather than on what its advice actually does,
    would compare methods on different quantities.
    """
    planner  = AStarPlanner(graph, reasoner)
    baseline = planner.find_path(start, goal, agent)

    selected = factors[:k]
    if not selected:
        return (0.0, 0.0)

    # Swap a mutated clone in for each affected individual, re-plan, restore.
    originals: Dict[str, OntologyIndividual] = {}
    clones:    Dict[str, OntologyIndividual] = {}
    for f in selected:
        name = f.individual_name
        if name not in clones:
            src = None
            for ind in graph.all_space_individuals():
                if ind.name == name:
                    src = ind
                    break
            if src is None:
                continue
            originals[name] = src
            clones[name]    = src.clone()
        clones[name].set(f.change.property_name, f.change.counterfactual_value)

    for name, clone in clones.items():
        _swap_individual(graph, name, clone)
    result = planner.find_path(start, goal, agent)
    for name, clone in clones.items():
        _swap_individual(graph, clone.name, originals[name])

    restored = float((not baseline.is_feasible) and result.is_feasible)
    improved = 0.0
    if baseline.is_feasible and result.is_feasible:
        improved = max(0.0, baseline.total_cost - result.total_cost)
    return (restored, improved)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(
    predicted: List[ExplanationFactor],
    truth:     Set[Tuple[str, str]],
    k:         int,
) -> float:
    if k <= 0:
        return 0.0
    pred_keys = {f.key for f in predicted[:k]}
    return len(pred_keys & truth) / float(k)


def recall_at_k(
    predicted: List[ExplanationFactor],
    truth:     Set[Tuple[str, str]],
    k:         int,
) -> float:
    if not truth:
        return 0.0
    pred_keys = {f.key for f in predicted[:k]}
    return len(pred_keys & truth) / float(len(truth))


def actionable_fraction(predicted: List[ExplanationFactor], k: int) -> float:
    """
    Fraction of the returned recommendations that an operator could act on.

    The denominator is how many recommendations the method actually returned,
    not k. Dividing by k would penalise a method for being concise, which is
    the opposite of what this metric is meant to reward.
    """
    selected = predicted[:k]
    if not selected:
        return 0.0
    return sum(float(f.actionable) for f in selected) / float(len(selected))


def feasibility_recovery_at_k(predicted: List[ExplanationFactor], k: int) -> float:
    """1.0 if at least one top-k factor restores feasibility."""
    return float(any(f.feasible_recovery for f in predicted[:k]))


def cost_improvement_at_k(predicted: List[ExplanationFactor], k: int) -> float:
    """Sum of cost improvements from top-k factors."""
    return sum(f.cost_improvement for f in predicted[:k])


def compute_metrics(
    predicted:    List[ExplanationFactor],
    truth:        Set[Tuple[str, str]],
    k:            int,
    feas_rec:     Optional[float] = None,
    cost_improve: Optional[float] = None,
) -> Dict[str, float]:
    """
    Metrics for one method on one task.

    Pass *feas_rec* and *cost_improve* from :func:`score_prediction_set` so that
    every method is judged on what its advice actually achieves. Omitting them
    falls back to whatever the method recorded about itself, which is only
    meaningful for engines that verify their own proposals.
    """
    return {
        "precision_at_k":      precision_at_k(predicted, truth, k),
        "recall_at_k":         recall_at_k(predicted, truth, k),
        "actionable_fraction":  actionable_fraction(predicted, k),
        "feasibility_recovery": (feasibility_recovery_at_k(predicted, k)
                                 if feas_rec is None else feas_rec),
        "cost_improvement":    (cost_improvement_at_k(predicted, k)
                                if cost_improve is None else cost_improve),
        "num_returned":        float(len(predicted[:k])),
    }
