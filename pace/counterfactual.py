"""
pace/counterfactual.py
==============================
Counterfactual affordance reasoning engine.

What is a counterfactual explanation?
--------------------------------------
A counterfactual answers: "What would need to be different for outcome Y
to have occurred instead of X?"

Applied to navigation: given that the robot chose path A, we answer:
  "Why didn't it take path B?"
  → "Because property P on corridor C has value V, making C impassable.
     If P were changed to V', the robot would prefer path B."

This implements the *minimal-change principle* from counterfactual logic
(Lewis, 1973; Pearl, 2000): among all possible world states in which the
alternative path is preferred, find the one that differs least from the
actual world.

Algorithm
---------
For each edge on the alternative path:
  1. If cost == inf (hard-blocked) or cost is disproportionately high:
       → call _minimal_changes() to find the smallest set of property
         mutations that would make the edge traversable / cheaper.
  2. Apply those mutations to deep copies of the relevant individuals
     (counterfactual world - does not mutate the live world model).
  3. Re-evaluate the alternative path cost in the counterfactual world.
  4. Compute cost delta = actual_cost − cf_cost.
  5. Filter changes by actionability (can an operator do this today?).

PropertyChange.is_actionable() separates:
  - Actionable (low effort): open a door, clear a crowd, fix lighting
  - Structural (high effort): widen a corridor, reduce a slope

This distinction is critical for generating *useful* recommendations -
telling a user to "widen the corridor" is less useful than "open the door."
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .affordance import AffordanceReasoner, AffordanceResult
from .config import DEFAULT_CONFIG, NavCostConfig
from .domain import AffordanceType, DoorState, MobilityType, SurfaceType
from .navigation import NavEdge, NavPath, AStarPlanner
from .ontology import OntologyIndividual


# Ordinal effort ranking, used only to break ties between equal-cardinality
# change sets (prefer the set an operator can actually carry out).
_EFFORT_RANK = {"low": 0, "medium": 1, "high": 2}

#: The order the greedy search prescribes fixes in, most severe constraint
#: first. It is a declared policy rather than a derived quantity, which is why
#: the evaluation permutes it and reports what changes (Section: ablations).
DEFAULT_PRIORITY: List[str] = [
    "is_hazardous", "restricted", "is_accessible", "door_state", "blocked_width",
    "width", "height", "slope_angle",
    "item_height", "item_width", "item_mass",
    "crowd_density", "obstacle_density", "illumination", "surface_type",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PropertyChange:
    """
    A single property mutation in a counterfactual scenario.

    Represents the statement:
      "IF {individual}.{property_name} were {counterfactual_value}
       (instead of {original_value}) THEN ..."

    Attributes
    ----------
    individual_name      : name of the space individual to be modified
    property_name        : the property key to change
    original_value       : current value in the actual world
    counterfactual_value : hypothetical value in the counterfactual world
    change_type          : "enable" | "disable" | "improve" | "relax"
                           purely informational - used by the explanation layer
    rationale            : one-sentence human explanation of why this change helps
    claim                : what the change asserts. "repair" - it clears a
                           condition that makes a segment impassable for this
                           robot, so it is part of restoring the route's
                           feasibility. "saving" - it removes a finite charge,
                           so it makes the route cheaper *given* that the route
                           is (or is made) feasible. The two are reported as
                           separate claims: pooling them into one ranked list is
                           what lets a true saving mislead on a task only
                           construction can repair.
    """
    individual_name:      str
    property_name:        str
    original_value:       Any
    counterfactual_value: Any
    change_type:          str = "modify"
    rationale:            str = ""
    claim:                str = "saving"

    def description(self) -> str:
        """Compact one-line description for report output."""
        return (
            f"'{self.individual_name}'.{self.property_name}: "
            f"{self.original_value!r} → {self.counterfactual_value!r}"
        )

    def is_actionable(self) -> bool:
        """
        True if a human operator could realistically make this change at runtime
        (without physical construction or renovation).

        Actionable properties: door state, access flags, staged goods, crowd
        density, illumination, obstacle density - all mutable without
        structural work.

        Non-actionable: corridor width, overhead clearance, slope angle -
        require construction.
        """
        ACTIONABLE = {
            "door_state",
            "is_accessible",
            "is_hazardous",
            "restricted",
            "blocked_width",
            "obstacle_density",
            "crowd_density",
            "illumination",
            "item_height",      # restock the item on a lower shelf or a cart
        }
        return self.property_name in ACTIONABLE

    def effort(self) -> str:
        """
        Qualitative effort estimate for the people on the floor.

        HIGH   - construction: widen an aisle, raise a clearance, regrade a
                 ramp (a project, months)
        MEDIUM - a work order or an authorisation: lift a zone restriction,
                 release an incident cordon, reopen a closed aisle, resurface
                 (hours to days)
        LOW    - one operator action now: open a door, move staged goods,
                 redirect people, switch on the lights (minutes)
        """
        HIGH   = {"width", "height", "slope_angle", "item_width", "item_mass"}
        MEDIUM = {"surface_type", "is_accessible", "restricted", "is_hazardous"}
        if self.property_name in HIGH:
            return "high"
        if self.property_name in MEDIUM:
            return "medium"
        return "low"


@dataclass
class Counterfactual:
    """
    The result of a "why not path X?" query.

    Encapsulates the actual path taken, the alternative being questioned,
    the minimal property changes that would have altered the decision,
    and derived metrics (cost delta, achievability).

    Attributes
    ----------
    query            : natural language question that prompted this analysis
    actual_path      : the path the robot actually chose
    alternative_path : the path being questioned
    changes          : minimal set of PropertyChange objects
    cf_cost          : cost of alternative in the counterfactual world
    cost_delta       : actual_cost − cf_cost (positive = alternative now cheaper)
    is_achievable    : True iff the operator-actionable changes ALONE make the
                       alternative feasible and no worse than the chosen path.
                       This is deliberately stricter than "some change happens
                       to be actionable": a saving that secretly depends on
                       construction work is never advertised as operator-doable.
    actionable_only_changes / _cost / _delta :
                       the world reachable using only actionable changes
    explanation      : auto-generated narrative sentence
    """
    query:            str
    actual_path:      NavPath
    alternative_path: NavPath
    changes:          List[PropertyChange]
    cf_cost:          float
    cost_delta:       float
    is_achievable:    bool
    explanation:      str = ""
    actionable_only_changes: List[PropertyChange] = field(default_factory=list)
    actionable_only_cost:    float = math.inf
    actionable_only_delta:   float = 0.0

    def num_changes(self) -> int:
        return len(self.changes)

    def actionable_changes(self) -> List[PropertyChange]:
        """Return only the changes an operator can make at runtime."""
        return [c for c in self.changes if c.is_actionable()]

    def non_actionable_changes(self) -> List[PropertyChange]:
        """Return only the structural/construction-level changes."""
        return [c for c in self.changes if not c.is_actionable()]

    def repair_changes(self) -> List[PropertyChange]:
        """The changes that clear impassable conditions on the questioned route."""
        return [c for c in self.changes if c.claim == "repair"]

    def saving_changes(self) -> List[PropertyChange]:
        """The changes that remove finite charges - claims conditional on feasibility."""
        return [c for c in self.changes if c.claim == "saving"]


# ---------------------------------------------------------------------------
# Counterfactual engine
# ---------------------------------------------------------------------------

class CounterfactualEngine:
    """
    Generates counterfactual explanations for navigation decisions.

    Constructor parameters
    ----------------------
    reasoner         : AffordanceReasoner - reused for re-evaluation
    planner          : AStarPlanner       - used to evaluate node sequences
    onto_individuals : dict               - live world name → individual mapping
                       (used as the source for cloning)
    """

    def __init__(
        self,
        reasoner:         AffordanceReasoner,
        planner:          AStarPlanner,
        onto_individuals: Dict[str, OntologyIndividual],
        search:           str = "greedy",
        config:           Optional[NavCostConfig] = None,
        attribute_costs:  bool = True,
        priority_order:   Optional[List[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        search   : "greedy"          - prescribe every applicable fix in
                                       priority order (the default; fast, but
                                       carries no minimality guarantee).
                   "min_cardinality" - exhaustive ascending-cardinality subset
                                       search over the same candidate pool.
                                       Returns a smallest satisfying set, which
                                       is therefore subset-minimal.  Used as an
                                       optimal reference point for measuring how
                                       much the greedy search over-prescribes.
        config   : numeric policy shared with the affordance reasoner. Using the
                   same config is what keeps the candidate generator consistent
                   with the rules that decide feasibility.
        attribute_costs :
                   also generate candidates from the *soft* terms of the cost
                   function, not only from hard blockers (see
                   :meth:`_cost_attribution_changes`). Setting it False recovers
                   the blocker-only generator, which is the ablation that shows
                   what attribution buys.
        priority_order :
                   property names in the order the greedy search prescribes
                   them. Passing an explicit order is what lets the evaluation
                   permute it and measure how much of the result the ordering is
                   responsible for; ``None`` uses :data:`DEFAULT_PRIORITY`.
        """
        if search not in ("greedy", "min_cardinality"):
            raise ValueError(f"Unknown search mode: {search!r}")
        self.reasoner  = reasoner
        self.planner   = planner
        self.onto_inds = onto_individuals
        self.search    = search
        self.config    = config or DEFAULT_CONFIG
        self.attribute_costs = attribute_costs
        self.priority    = list(priority_order or DEFAULT_PRIORITY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_why_not(
        self,
        actual_path: NavPath,
        alt_nodes:   List[str],
        agent:       OntologyIndividual,
    ) -> Counterfactual:
        """
        Answer: "Why didn't the robot take the route through *alt_nodes*?"

        Steps
        -----
        1. Evaluate the alternative's cost in the current world.
        2. If already preferred → trivial counterfactual (no changes needed).
        3. Identify problematic edges (inf cost or > 1.5× baseline).
        4. Generate minimal changes for each problematic edge.
        5. Apply changes to cloned individuals → re-evaluate → compute delta.
        6. Return Counterfactual with full explanation.

        Parameters
        ----------
        actual_path : the path the robot actually took
        alt_nodes   : the node sequence of the alternative to explain
        agent       : the robot agent individual
        """
        alt_path = self.planner.evaluate_sequence(alt_nodes, agent)
        query    = "Why not: " + " → ".join(alt_nodes) + "?"

        # Trivial case: alternative is already preferred
        # This can happen when the planner's k-shortest diverges from the user's
        # expectation, or when the world changed between planning and querying.
        if alt_path.is_feasible and alt_path.total_cost <= actual_path.total_cost:
            return Counterfactual(
                query            = query,
                actual_path      = actual_path,
                alternative_path = alt_path,
                changes          = [],
                cf_cost          = alt_path.total_cost,
                cost_delta       = actual_path.total_cost - alt_path.total_cost,
                is_achievable    = True,
                explanation      = (
                    "The alternative path is already feasible and equally or "
                    "more cost-effective - the planner should have chosen it. "
                    "(Possible tie-breaking or graph asymmetry.)"
                ),
            )

        # Identify problematic edges
        # "Problematic" = infinite cost (hard-blocked) OR cost > 1.5× the
        # sum of edge distance + a 2-unit door-opening buffer.  The 1.5× factor
        # catches edges that are technically passable but disproportionately
        # expensive (e.g. extremely crowded corridors).
        blockers:    List[PropertyChange]              = []
        attributions: List[Tuple[float, PropertyChange]] = []
        for edge in alt_path.edges:
            cost, af = self.reasoner.navigation_cost(
                edge.space_individual, agent, base_distance=edge.distance
            )
            baseline = edge.distance + self.config.cf_problematic_buffer
            if cost == math.inf or cost > self.config.cf_problematic_factor * baseline:
                blockers.extend(
                    self._minimal_changes(edge.space_individual, agent, af,
                                          hard_blocked=(cost == math.inf)))
            # Attribution applies to every segment the robot can already cross,
            # including the ones nothing is wrong with, which is exactly where a
            # blocker-driven generator has nothing to say.
            if self.attribute_costs and math.isfinite(cost):
                attributions.extend(
                    self._cost_attribution_changes(edge.space_individual, agent, af))

        # Blockers first, in the declared priority order; then the attributed
        # savings, largest first, an ordering the cost function computes rather
        # than one we assert.
        attributions.sort(key=lambda t: -t[0])
        pool = self._dedupe(self._order(blockers) + [c for _, c in attributions])

        # Select which candidate changes to prescribe
        # The candidate pool above is identical in both modes; only the search
        # over that pool differs, which is what makes the two comparable.
        if self.search == "min_cardinality":
            changes, cf_cost, _preferred = self._minimal_subset(
                alt_path, agent, pool, target_cost=actual_path.total_cost
            )
        else:
            # Greedy: prescribe the whole priority-ordered pool.  Clones isolate
            # the counterfactual world so the live world is never mutated.
            changes      = list(pool)
            cf_overrides = self._build_overrides(changes)
            cf_cost      = self._evaluate_with_overrides(alt_path, agent, cf_overrides)

        # Definition 2 is a claim about the questioned route, so a set that does
        # not move that route is not a repair set and is not reported as one.
        if not self._improves(alt_path, cf_cost):
            changes, cf_cost = [], alt_path.total_cost
        cost_delta = actual_path.total_cost - cf_cost

        # Honest attribution: ask separately what an operator could achieve with
        # no construction work, so "achievable" always names a world they can
        # actually reach rather than one that quietly needs building work.
        act_changes, act_cost = self._actionable_only_world(
            alt_path, agent, pool, actual_path.total_cost
        )
        is_achievable = math.isfinite(act_cost) and act_cost <= actual_path.total_cost
        act_delta = (actual_path.total_cost - act_cost
                     if math.isfinite(act_cost) and math.isfinite(actual_path.total_cost)
                     else 0.0)
        explanation   = self._narrative(actual_path, alt_path, changes, cf_cost, cost_delta)

        return Counterfactual(
            query            = query,
            actual_path      = actual_path,
            alternative_path = alt_path,
            changes          = changes,
            cf_cost          = cf_cost,
            cost_delta       = cost_delta,
            is_achievable    = is_achievable,
            explanation      = explanation,
            actionable_only_changes = act_changes,
            actionable_only_cost    = act_cost,
            actionable_only_delta   = act_delta,
        )

    def savings_on(
        self,
        path:  NavPath,
        agent: OntologyIndividual,
    ) -> List[Tuple[float, PropertyChange]]:
        """
        What would make the route the robot *takes* cheaper: cost attribution
        over its own legs, each candidate verified by re-evaluating the route.

        A deviation is not only "why not that way?" but also "why is this way
        so slow?": the crowd on the aisle being driven, the closed door the
        robot opens itself, the clutter it weaves through. Those terms sit on
        the chosen route, which no questioned alternative contains, so they are
        attributed here and reported as savings. Returns ``(saving, change)``
        pairs, largest saving first, savings below epsilon dropped.
        """
        if not path.is_feasible or not path.edges or not self.attribute_costs:
            return []
        out: List[Tuple[float, PropertyChange]] = []
        seen: set = set()
        for edge in path.edges:
            _cost, af = self.reasoner.navigation_cost(edge.space_individual, agent, base_distance=edge.distance)
            for _est, change in self._cost_attribution_changes(edge.space_individual, agent, af):
                key = (change.individual_name, change.property_name)
                if key in seen:
                    continue
                seen.add(key)
                cf_cost = self._evaluate_with_overrides(path, agent, self._build_overrides([change]))
                saving = path.total_cost - cf_cost
                if math.isfinite(cf_cost) and saving > self.config.cf_epsilon:
                    out.append((saving, change))
        out.sort(key=lambda t: -t[0])
        return out

    def batch_why_not(
        self,
        actual_path: NavPath,
        candidates:  List[NavPath],
        agent:       OntologyIndividual,
    ) -> List[Counterfactual]:
        """
        Explain all candidate alternative paths in one call.

        Deduplicates by node sequence so the same alternative is not
        explained twice (Yen's can occasionally produce near-duplicates
        due to floating-point cost ties).
        """
        result:    List[Counterfactual] = []
        seen_seqs: set                  = {tuple(actual_path.nodes)}

        for cand in candidates:
            seq = tuple(cand.nodes)
            if seq not in seen_seqs:
                seen_seqs.add(seq)
                cf = self.explain_why_not(actual_path, list(cand.nodes), agent)
                result.append(cf)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _minimal_changes(
        self,
        entity: OntologyIndividual,
        agent:  OntologyIndividual,
        af:     AffordanceResult,
        hard_blocked: bool = False,
    ) -> List[PropertyChange]:
        """
        Generate the smallest set of property mutations that would make
        *entity* traversable and reasonably cheap for *agent*.

        Strategy: check each blocking condition in priority order (cheapest
        actionable fix first).  We do NOT solve the full optimization problem -
        this is a greedy heuristic that works well for the typical cases.

        Priority order:
          1. Hazard removal (enable traversal)
          2. Access grant   (enable traversal)
          3. Door opening   (enable traversal)
          4. Geometry fix   (enable traversal - structural)
          5. Slope fix      (enable traversal - structural)
          6. Crowd reduction (reduce cost)
          7. Lighting fix   (reduce cost)

        *hard_blocked* says whether the segment's cost is infinite for this
        agent. Each change's ``claim`` is "repair" only when the condition it
        clears actually contributes to that infinity - a closed door the robot
        can open, on a segment blocked by something else, is a saving, not a
        repair. On a passable segment nothing is a repair.
        """
        changes: List[PropertyChange] = []
        cfg = self.config
        e = entity.properties
        a = agent.properties

        def blocking(contributes: bool) -> str:
            return "repair" if (hard_blocked and contributes) else "saving"

        # An item: one candidate per unsatisfied manipulation condition, each
        # naming the item property that would clear it.  A robot without an
        # arm gets none: nothing about the plant makes it able to pick, and
        # saying so is the explanation.
        if cfg.is_item(e):
            if not a.get("has_arm", False):
                return changes
            if not cfg.within_reach(e, a):
                changes.append(PropertyChange(
                    individual_name      = entity.name,
                    property_name        = "item_height",
                    original_value       = round(e.get("item_height", 0.0), 2),
                    counterfactual_value = round(max(0.0, a.get("reach_height", cfg.default_reach_height)
                                                     - cfg.cf_reach_margin), 2),
                    change_type          = "improve",
                    rationale            = f"Restocking {entity.name} lower brings it within the arm's reach.",
                    claim                = blocking(True),
                ))
            if not cfg.fits_gripper(e, a):
                changes.append(PropertyChange(
                    individual_name      = entity.name,
                    property_name        = "item_width",
                    original_value       = round(e.get("item_width", 0.0), 3),
                    counterfactual_value = round(a.get("gripper_opening", cfg.default_gripper_opening), 3),
                    change_type          = "improve",
                    rationale            = f"{entity.name} would need a grip no wider than the gripper opens.",
                    claim                = blocking(True),
                ))
            if not cfg.within_payload(e, a):
                changes.append(PropertyChange(
                    individual_name      = entity.name,
                    property_name        = "item_mass",
                    original_value       = round(e.get("item_mass", 0.0), 2),
                    counterfactual_value = round(a.get("payload", cfg.default_payload), 2),
                    change_type          = "improve",
                    rationale            = f"{entity.name} would have to weigh no more than the arm's payload.",
                    claim                = blocking(True),
                ))
            return changes

        # Priority 1: Hazard
        if e.get("is_hazardous", False):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "is_hazardous",
                original_value       = True,
                counterfactual_value = False,
                change_type          = "disable",
                rationale            = f"Clearing the hazard makes {entity.name} traversable.",
                claim                = blocking(True),
            ))

        # Priority 2a: Restricted area
        if e.get("restricted", False):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "restricted",
                original_value       = True,
                counterfactual_value = False,
                change_type          = "enable",
                rationale            = f"Granting robot access to the restricted area.",
                claim                = blocking(True),
            ))

        # Priority 2b: Inaccessible
        if not e.get("is_accessible", True):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "is_accessible",
                original_value       = False,
                counterfactual_value = True,
                change_type          = "enable",
                rationale            = f"Marking {entity.name} as accessible.",
                claim                = blocking(True),
            ))

        # Priority 3: Door state
        door_state = e.get("door_state")   # None if space has no door
        if door_state in (DoorState.CLOSED.value, DoorState.LOCKED.value):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "door_state",
                original_value       = door_state,
                counterfactual_value = DoorState.OPEN.value,
                change_type          = "enable",
                rationale            = f"Opening (or unlocking) the {entity.name} door.",
                # A locked door blocks everyone; a closed one blocks only a
                # robot that cannot open it. For a robot that can, the change
                # merely saves the stop-and-open penalty.
                claim                = blocking(
                    door_state == DoorState.LOCKED.value
                    or AffordanceType.OPENABLE not in af.affordances),
            ))

        # Priority 4: Width.  Passable needs free width, i.e. structural width
        # minus what is staged in the passage, to clear the robot plus its
        # clearances.  Two candidates, each for one term of that difference:
        # clearing the staged goods (actionable) whenever there are any, and
        # widening (structural) only when the passage is too narrow even
        # empty.  Both are emitted when both hold - the conjunctive case.
        required = cfg.required_width(a)
        actual_w = e.get("width", cfg.default_space_width)
        if not cfg.fits_width(e, a):
            staged = cfg.blocked_width(e)
            if staged > 0.0:
                changes.append(PropertyChange(
                    individual_name      = entity.name,
                    property_name        = "blocked_width",
                    original_value       = round(staged, 2),
                    counterfactual_value = 0.0,
                    change_type          = "improve",
                    rationale            = (
                        f"Moving the staged goods out of {entity.name} restores "
                        f"its full width."
                    ),
                    claim                = blocking(True),
                ))
            if not cfg.fits_width_if_cleared(e, a):
                changes.append(PropertyChange(
                    individual_name      = entity.name,
                    property_name        = "width",
                    original_value       = round(actual_w, 2),
                    # Add 0.1 m safety margin above the bare minimum
                    counterfactual_value = round(required + cfg.cf_width_safety_margin, 2),
                    change_type          = "improve",
                    rationale            = (
                        f"Widening {entity.name} to accommodate robot + clearance."
                    ),
                    claim                = blocking(True),
                ))

        # Priority 4b: Height / overhead clearance (structural)
        # A space can be wide enough yet too low: the blocking rule compares the
        # robot's height against the space's overhead clearance.  Without this
        # candidate the search misdiagnoses a height blocker as a width blocker
        # and proposes a widening that cannot restore feasibility.
        required_h = cfg.required_height(a)
        actual_h   = e.get("height")
        if actual_h is not None and not cfg.fits_height(e, a):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "height",
                original_value       = round(actual_h, 2),
                counterfactual_value = round(required_h + cfg.cf_height_safety_margin, 2),
                change_type          = "improve",
                rationale            = (
                    f"Raising overhead clearance in {entity.name} to admit the robot."
                ),
                claim                = blocking(True),
            ))

        # Priority 5: Slope (structural)
        slope     = e.get("slope_angle", cfg.default_slope_angle)
        max_slope = a.get("max_slope_angle", cfg.default_max_slope_angle)
        if not cfg.slope_within_limit(e, a):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "slope_angle",
                original_value       = round(slope, 1),
                # Reduce to 1° below robot's limit to ensure rule fires
                counterfactual_value = round(max_slope - cfg.cf_slope_margin, 1),
                change_type          = "improve",
                rationale            = (
                    f"Installing a ramp that reduces {entity.name} slope "
                    f"to within robot limits."
                ),
                # An over-limit slope blocks only a robot that cannot climb it;
                # for one that can, levelling merely trims the gradient cost.
                claim                = blocking(
                    AffordanceType.CLIMBABLE not in af.affordances),
            ))

        # Priority 6: Crowd density (actionable, soft cost)
        # Only flag crowd as a blocker if it is very high (> 50%).
        # Lower densities are already captured in the cost formula without
        # making the path infeasible.
        crowd = e.get("crowd_density", cfg.default_crowd_density)
        if crowd > cfg.crowd_block_threshold:
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "crowd_density",
                original_value       = round(crowd, 2),
                counterfactual_value = cfg.cf_crowd_target,
                change_type          = "improve",
                rationale            = (
                    f"Reducing crowd density in {entity.name} to enable safe transit."
                ),
            ))

        # Priority 7: Illumination (actionable, soft cost)
        illum = e.get("illumination", cfg.default_illumination)
        if not cfg.lit_enough(e):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "illumination",
                original_value       = round(illum, 2),
                counterfactual_value = cfg.cf_illumination_target,
                change_type          = "improve",
                rationale            = (
                    f"Increasing lighting in {entity.name} for reliable perception."
                ),
            ))

        return changes

    def _cost_attribution_changes(
        self,
        entity: OntologyIndividual,
        agent:  OntologyIndividual,
        af:     AffordanceResult,
    ) -> List[Tuple[float, PropertyChange]]:
        """
        Candidates read off the *soft* terms of the cost function.

        :meth:`_minimal_changes` generates a candidate only where something
        blocks the robot, which leaves a whole class of useful repairs
        unreachable: a segment the robot can already traverse, but pays to
        traverse, names no candidate at all even when removing that payment is
        the cheapest way to make the questioned route preferable.

        Equation~(2) already assigns every unit of that payment to exactly one
        property, so the generator is the same idea as the affordance rules
        applied to the cost rather than to feasibility: a term that is not zero
        names the property that would zero it. Blockers are the infinite terms;
        these are the finite ones.

        A term is only worth proposing if removing it saves at least
        ``cf_epsilon``, which is the threshold at which the evaluation credits a
        cost reduction at all. That keeps the pool from filling with changes
        that could never be scored as improvements, and it is a derived bound
        rather than another hand-set one.

        Returns ``(saving, change)`` pairs so the caller can order them by what
        they are worth, which is a computed ordering rather than a declared one.
        """
        cfg = self.config
        e   = entity.properties
        out: List[Tuple[float, PropertyChange]] = []
        if cfg.is_item(e):
            return out                     # handling has no soft terms to attribute

        # The emergency discount scales the whole expression, so it scales every
        # saving too: a term is worth what it is worth *after* the multiplier.
        beta = cfg.emergency_discount if e.get("emergency_route", False) else 1.0

        def add(saving: float, prop: str, new: Any, why: str, kind: str) -> None:
            if beta * saving >= cfg.cf_epsilon:
                out.append((beta * saving, PropertyChange(
                    individual_name      = entity.name,
                    property_name        = prop,
                    original_value       = e.get(prop),
                    counterfactual_value = new,
                    change_type          = kind,
                    rationale            = why,
                )))

        # An openable closed door is passable but costs a stop and a manipulation.
        if (e.get("door_state") == DoorState.CLOSED.value
                and AffordanceType.OPENABLE in af.affordances):
            add(cfg.door_opening_penalty, "door_state", DoorState.OPEN.value,
                f"Propping the {entity.name} door open saves the robot the stop.",
                "enable")

        crowd = e.get("crowd_density", cfg.default_crowd_density)
        if crowd > cfg.cf_crowd_target:
            add((crowd - cfg.cf_crowd_target) * cfg.crowd_cost_weight,
                "crowd_density", cfg.cf_crowd_target,
                f"Thinning the crowd in {entity.name} speeds the transit.", "improve")

        obstacle = e.get("obstacle_density", cfg.default_obstacle_density)
        if obstacle > 0.0:
            add(obstacle * cfg.obstacle_cost_weight, "obstacle_density", 0.0,
                f"Clearing the obstacles in {entity.name} opens the full width.",
                "improve")

        if AffordanceType.OBSERVABLE not in af.affordances:
            add(cfg.visibility_penalty, "illumination", cfg.cf_illumination_target,
                f"Lighting {entity.name} removes the perception-uncertainty penalty.",
                "improve")

        surface = e.get("surface_type", SurfaceType.SMOOTH.value)
        add(cfg.surface_cost.get(surface, 0.0), "surface_type",
            SurfaceType.SMOOTH.value,
            f"Resurfacing {entity.name} removes the traction penalty.", "improve")

        slope = e.get("slope_angle", cfg.default_slope_angle)
        if slope > 0.0:
            add(slope * cfg.slope_cost_coeff, "slope_angle", 0.0,
                f"Levelling {entity.name} removes the gradient penalty.", "improve")

        return out

    def _order(self, pool: List[PropertyChange]) -> List[PropertyChange]:
        """Sort a candidate pool into the declared priority order, stably."""
        rank = {p: i for i, p in enumerate(self.priority)}
        return sorted(pool, key=lambda c: rank.get(c.property_name, len(rank)))

    def _improves(self, alt_path: NavPath, cf_cost: float) -> bool:
        """
        Whether a repair set has demonstrably done something for the route asked about.

        Definition~2 asks a repair set to make the questioned route work. A set
        that leaves it infeasible, or shaves less off its cost than the epsilon
        at which a saving is credited, has not, and reporting it anyway would be
        advice that buys the user nothing.
        """
        if not math.isfinite(cf_cost):
            return False
        if not alt_path.is_feasible:
            return True                       # infeasible → planable is the whole point
        return (alt_path.total_cost - cf_cost) > self.config.cf_epsilon

    def _actionable_only_world(
        self,
        alt_path:    NavPath,
        agent:       OntologyIndividual,
        pool:        List[PropertyChange],
        target_cost: float,
    ) -> Tuple[List[PropertyChange], float]:
        """
        Evaluate the best world reachable using only operator-actionable changes.

        Runs the same search as :meth:`explain_why_not`, but over the actionable
        subset of the candidate pool. Returns the changes and the resulting cost
        of *alt_path*; the cost is infinite when no combination of actionable
        changes makes the route passable.
        """
        actionable = [c for c in pool if c.is_actionable()]
        if not actionable:
            return ([], math.inf)
        if self.search == "min_cardinality":
            changes, cost, _ = self._minimal_subset(
                alt_path, agent, actionable, target_cost
            )
            return (changes, cost)
        cost = self._evaluate_with_overrides(
            alt_path, agent, self._build_overrides(actionable)
        )
        # Only report these changes if they actually achieve something on their
        # own; otherwise the operator-reachable world is simply out of reach.
        if not (math.isfinite(cost) and cost <= target_cost):
            return ([], cost)
        return (actionable, cost)

    # ------------------------------------------------------------------
    # Minimum-cardinality subset search (optimal reference)
    # ------------------------------------------------------------------

    def _minimal_subset(
        self,
        alt_path:    NavPath,
        agent:       OntologyIndividual,
        pool:        List[PropertyChange],
        target_cost: float,
    ) -> Tuple[List[PropertyChange], float, bool]:
        """
        Find a smallest-cardinality subset of *pool* that makes *alt_path*
        feasible AND no more expensive than *target_cost*.

        Returns ``(changes, cf_cost, preferred)``:
          - the smallest satisfying subset and its cost, ``preferred=True``; or
          - if no subset is preferred, the cheapest feasible subset found,
            ``preferred=False`` (possibly the empty set); or
          - ``([], inf, False)`` if even the full pool cannot restore feasibility.

        Minimum cardinality implies subset-minimality: every smaller cardinality
        was enumerated and rejected first, so no proper subset of the result also
        satisfies the goal.  Ties are broken by (resulting cost, total operator
        effort), which is order-independent - unlike the greedy search, whose
        output depends on the fixed priority ordering.
        """
        pool = self._dedupe(pool)
        n    = len(pool)

        empty_cost = self._evaluate_subset(alt_path, agent, [])
        if n == 0:
            return ([], empty_cost, math.isfinite(empty_cost) and empty_cost <= target_cost)

        # Exhaustive search is 2^n; fall back to backward elimination if the
        # pool is large (still subset-minimal, see _reduce_from_full).
        if n > self.config.cf_max_pool:
            return self._reduce_from_full(alt_path, agent, pool, target_cost)

        # Seed with the do-nothing world when it is already feasible, so we never
        # prescribe changes on a task that needs none.
        best_feasible: Optional[Tuple[List[PropertyChange], float]] = (
            ([], empty_cost) if math.isfinite(empty_cost) else None
        )

        for k in range(1, n + 1):                        # ascending cardinality
            winners: List[Tuple[List[PropertyChange], float]] = []
            for combo in itertools.combinations(pool, k):
                subset = list(combo)
                cf     = self._evaluate_subset(alt_path, agent, subset)
                if math.isfinite(cf):
                    if best_feasible is None or cf < best_feasible[1]:
                        best_feasible = (subset, cf)
                    if cf <= target_cost:                # feasible AND preferred
                        winners.append((subset, cf))
            if winners:
                winners.sort(key=lambda w: (
                    w[1], sum(_EFFORT_RANK[c.effort()] for c in w[0])
                ))
                return (winners[0][0], winners[0][1], True)

        if best_feasible is not None:
            return (best_feasible[0], best_feasible[1], False)
        return ([], math.inf, False)

    def _reduce_from_full(
        self,
        alt_path:    NavPath,
        agent:       OntologyIndividual,
        pool:        List[PropertyChange],
        target_cost: float,
    ) -> Tuple[List[PropertyChange], float, bool]:
        """
        Subset-minimal fallback for pools too large for exhaustive enumeration.

        Every candidate change is *monotone*: it is feasibility-non-decreasing and
        cost-non-increasing (clearing a blocker cannot remove an affordance;
        widening, regrading, opening, de-crowding or lighting a space cannot raise
        its cost).  Applying the whole pool therefore yields maximum feasibility
        and minimum cost - if the full pool is not feasible-and-preferred, no
        subset is.  Given monotonicity, one backward-elimination pass (dropping
        highest-effort changes first, so the kept set stays as actionable as
        possible) yields a subset-minimal result.
        """
        full_cost = self._evaluate_subset(alt_path, agent, pool)
        if not math.isfinite(full_cost):
            return ([], math.inf, False)      # even everything cannot help

        preferred = full_cost <= target_cost
        selected  = list(pool)
        current   = full_cost
        for c in sorted(pool, key=lambda c: -_EFFORT_RANK[c.effort()]):
            trial = [x for x in selected if x is not c]
            cf    = self._evaluate_subset(alt_path, agent, trial)
            if math.isfinite(cf) and (cf <= target_cost if preferred else True):
                selected, current = trial, cf
        return (selected, current, preferred)

    # ------------------------------------------------------------------
    # World simulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe(pool: List[PropertyChange]) -> List[PropertyChange]:
        """Drop duplicate atoms (same individual + property + target value)."""
        seen: set = set()
        uniq: List[PropertyChange] = []
        for c in pool:
            key = (c.individual_name, c.property_name, repr(c.counterfactual_value))
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        return uniq

    def _build_overrides(
        self,
        changes: List[PropertyChange],
    ) -> Dict[str, OntologyIndividual]:
        """
        Group *changes* by individual, clone each affected live individual once,
        and apply the mutations to the clone.  The live world is never touched.
        """
        overrides: Dict[str, OntologyIndividual] = {}
        for change in changes:
            if change.individual_name not in overrides:
                orig = self.onto_inds.get(change.individual_name)
                if orig is None:
                    continue
                overrides[change.individual_name] = orig.clone()
            overrides[change.individual_name].set(
                change.property_name, change.counterfactual_value
            )
        return overrides

    def _evaluate_subset(
        self,
        alt_path: NavPath,
        agent:    OntologyIndividual,
        subset:   List[PropertyChange],
    ) -> float:
        """Total cost of *alt_path* with *subset* applied to cloned individuals."""
        return self._evaluate_with_overrides(
            alt_path, agent, self._build_overrides(subset)
        )

    def _evaluate_with_overrides(
        self,
        path:      NavPath,
        agent:     OntologyIndividual,
        overrides: Dict[str, OntologyIndividual],
    ) -> float:
        """
        Evaluate path cost using counterfactually modified individuals.

        For each edge, if the edge's space individual has a counterfactual
        clone in *overrides*, use the clone; otherwise use the original.
        This allows evaluating a mix of changed and unchanged spaces.
        """
        # A path with no edges is not a zero-cost path: it is one the graph could
        # not realise at all. Returning 0.0 here would read as a free route and
        # manufacture an enormous apparent saving.
        if not path.edges:
            return math.inf

        total = 0.0
        for edge in path.edges:
            # Use the counterfactual clone if available, else the live individual
            entity = overrides.get(edge.space_individual.name, edge.space_individual)
            cost, _ = self.reasoner.navigation_cost(entity, agent, edge.distance)
            if cost == math.inf:
                return math.inf          # one impassable edge blocks the path
            total += cost
        return total

    def _narrative(
        self,
        actual:     NavPath,
        alt:        NavPath,
        changes:    List[PropertyChange],
        cf_cost:    float,
        cost_delta: float,
    ) -> str:
        """
        Generate a one-paragraph English explanation for the counterfactual.

        Four cases:
          1. No changes, alternative feasible but costlier → "optimal decision"
          2. No changes, alternative infeasible → "no fixable property"
          3. Changes found, counterfactual improves cost → include savings
          4. Changes found, still infeasible after changes → note residual issue
        """
        # Format actual path label, handling the case where it is infeasible
        actual_label = (
            " → ".join(actual.nodes)
            if actual.nodes
            else "⚠ infeasible actual path"
        )
        actual_cost_label = (
            f"{actual.total_cost:.2f}"
            if actual.total_cost < math.inf
            else "∞ (infeasible)"
        )

        if not changes and not alt.is_feasible:
            # Alternative is hard-blocked and we found no fixable property
            return (
                f"Robot chose {actual_label} over {' → '.join(alt.nodes)} "
                f"because the alternative is infeasible with no identifiable "
                f"single-property fix."
            )

        if not changes:
            # Alternative is feasible but more expensive → correct decision
            return (
                f"Robot chose {actual_label} (cost {actual_cost_label}) over "
                f"{' → '.join(alt.nodes)} (cost {alt.total_cost:.2f}) because "
                f"the chosen route is already {abs(cost_delta):.2f} cost units "
                f"cheaper.  No world changes are needed - this is the optimal decision."
            )

        # Summarise the top-3 changes in the narrative
        reason_parts  = [c.description() for c in changes[:3]]
        reason        = ", ".join(reason_parts)

        if cost_delta > 0 and cf_cost < math.inf:
            # After changes, alternative becomes cheaper (or restores feasibility)
            if actual.total_cost == math.inf:
                # Actual path was infeasible - changes restore feasibility
                suffix = (
                    f"  If corrected, the alternative becomes feasible at cost {cf_cost:.2f}"
                    f" (restores navigation capability)."
                )
            else:
                suffix = (
                    f"  If corrected, the alternative would cost {cf_cost:.2f} vs "
                    f"actual {actual_cost_label} (saving {cost_delta:.2f} units)."
                )
        elif cf_cost == math.inf:
            # Even with changes the alternative remains blocked
            suffix = (
                "  Even after these minimal changes the alternative remains infeasible."
            )
        else:
            suffix = ""

        return (
            f"Robot chose {actual_label} over {' → '.join(alt.nodes)} because "
            f"the following properties block or penalise the route: {reason}.{suffix}"
        )
