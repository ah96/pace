"""
Numeric policy for the whole framework, in one place.

Every threshold, weight and geometric fallback used by the affordance rules,
the cost function and the counterfactual search lives here rather than being
written inline at the point of use. Two benefits follow:

  * A sensitivity analysis is a matter of constructing a different config,
    not of hunting for magic numbers across three modules.
  * The geometry formulas (does the robot fit? is the slope within limits?)
    have exactly one definition, so the affordance rules and the counterfactual
    candidate generator cannot drift apart.

Fields fall into two kinds, and the distinction matters:

  THRESHOLD / GEOMETRY  read by rule conditions; they decide *feasibility*.
  WEIGHT / PENALTY      read only by the soft-cost part of ``navigation_cost``;
                        they change how expensive an edge is, never whether it
                        is passable at all.

The dataclass is frozen so one shared instance cannot be mutated mid-run.
The defaults reproduce the framework's published numeric behaviour exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .domain import DoorState, MobilityType, SurfaceType


@dataclass(frozen=True)
class NavCostConfig:
    """Thresholds, weights, and the feasibility predicates derived from them."""

    # Geometric fallbacks, used when an individual omits a property
    default_space_width:    float = 10.0
    default_space_height:   float = 3.0
    default_robot_width:    float = 0.5
    default_robot_height:   float = 1.5
    default_min_clearance:  float = 0.15
    clearance_buffer_sides: int   = 2      # one clearance buffer on each side
    default_blocked_width:  float = 0.0    # nothing staged in the passage

    # Slope
    default_slope_angle:       float = 0.0
    default_max_slope_angle:   float = 8.0
    wheeled_stair_slope_limit: float = 15.0   # wheels fail on stair-like angles

    # Illumination
    default_illumination: float = 1.0
    observable_threshold: float = 0.25     # above this the space is OBSERVABLE

    # Manipulation (items)
    default_reach_height:    float = 0.0   # a robot without an arm reaches nothing
    default_gripper_opening: float = 0.0
    default_payload:         float = 0.0
    handling_cost:           float = 2.0   # stopping, grasping, stowing: in metres of detour
    cf_reach_margin:         float = 0.10  # restock this far below the reach limit

    # Occupancy
    default_crowd_density:    float = 0.0
    default_obstacle_density: float = 0.0
    crowd_cost_weight:        float = 4.0  # crowds are slow and risky
    obstacle_cost_weight:     float = 2.5  # scattered obstacles can be weaved around
    crowd_block_threshold:    float = 0.5  # above this a crowd is worth proposing a fix for

    # Soft-cost penalties
    door_opening_penalty: float = 2.5      # stopping and engaging a manipulator
    visibility_penalty:   float = 1.8      # added when OBSERVABLE is missing
    slope_cost_coeff:     float = 0.06     # per degree of gradient

    # Route preference
    emergency_discount: float = 0.75       # 25% off designated emergency routes
    cost_floor:         float = 0.01       # keeps edge costs strictly positive

    # Counterfactual search
    cf_crowd_target:        float = 0.2    # value a crowd fix aims for
    cf_illumination_target: float = 0.6    # value a lighting fix aims for
    cf_width_safety_margin: float = 0.1    # widen slightly past the bare minimum
    cf_height_safety_margin: float = 0.1   # likewise for overhead clearance
    cf_slope_margin:        float = 1.0    # regrade to just inside the robot's limit
    cf_problematic_factor:  float = 1.5    # edge is "expensive" past this multiple
    cf_problematic_buffer:  float = 2.0    # allowance added to the distance baseline
    cf_max_pool:            int   = 12     # above this, skip exhaustive enumeration
    # Smallest saving worth proposing a change for, and the same threshold at
    # which a repair set counts as an improvement at all. It is the epsilon the
    # evaluation uses to decide that a cost went down, so the search proposes
    # exactly the changes that could be credited.
    cf_epsilon: float = 1.0

    # Surface friction, added on top of the base edge distance
    surface_cost: Dict[str, float] = field(default_factory=lambda: {
        SurfaceType.SMOOTH.value:   0.0,
        SurfaceType.TILED.value:    0.0,
        SurfaceType.CARPETED.value: 0.3,
        SurfaceType.ROUGH.value:    0.6,
        SurfaceType.WET.value:      1.4,   # slip risk, so extra caution
        SurfaceType.GRAVEL.value:   0.9,
        SurfaceType.GRASS.value:    0.7,
    })

    # Derived predicates
    # These are the only definitions of the geometry and threshold formulas.
    # Both the affordance rules and the counterfactual candidate generator call
    # them, which is what keeps the two consistent.

    def required_width(self, a: Dict[str, Any]) -> float:
        """Width the robot needs: its body plus one clearance buffer per side."""
        return (a.get("robot_width", self.default_robot_width)
                + self.clearance_buffer_sides
                * a.get("min_clearance", self.default_min_clearance))

    def blocked_width(self, e: Dict[str, Any]) -> float:
        """Width taken up by temporary storage (staged goods, parked carts)."""
        return e.get("blocked_width", self.default_blocked_width)

    def free_width(self, e: Dict[str, Any]) -> float:
        """Width actually available to drive through: structural minus staged."""
        return e.get("width", self.default_space_width) - self.blocked_width(e)

    def fits_width(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """True if the free width admits the robot and its clearances."""
        return self.free_width(e) >= self.required_width(a)

    def fits_width_if_cleared(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """True if the *structural* width admits the robot, i.e. clearing the
        staged goods would (or does) make the passage wide enough.  False means
        only construction can: the passage is too narrow even when empty."""
        return e.get("width", self.default_space_width) >= self.required_width(a)

    def required_height(self, a: Dict[str, Any]) -> float:
        """Overhead clearance the robot needs, i.e. its own height."""
        return a.get("robot_height", self.default_robot_height)

    def fits_height(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """True if the space is tall enough for the robot to pass under."""
        return e.get("height", self.default_space_height) >= self.required_height(a)

    def slope_within_limit(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """True if the gradient is inside this robot's rated slope limit."""
        return (e.get("slope_angle", self.default_slope_angle)
                <= a.get("max_slope_angle", self.default_max_slope_angle))

    def wheeled_on_stairs(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """True for a wheeled robot facing a stair-like angle, which wheels cannot climb."""
        return (e.get("slope_angle", self.default_slope_angle) > self.wheeled_stair_slope_limit
                and a.get("mobility_type", MobilityType.WHEELED.value)
                    == MobilityType.WHEELED.value)

    def lit_enough(self, e: Dict[str, Any]) -> bool:
        """True if there is enough light for reliable perception."""
        return e.get("illumination", self.default_illumination) > self.observable_threshold

    def can_open_closed_door(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """True if the door is merely closed (not locked) and the robot can open it."""
        return (e.get("door_state") == DoorState.CLOSED.value
                and a.get("can_open_doors", False))

    # Items.  An individual is an item when it carries item_mass; the rules
    # and the candidate generator both use this one test.

    @staticmethod
    def is_item(e: Dict[str, Any]) -> bool:
        return "item_mass" in e

    def within_reach(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """True if an arm exists and gets to the surface the item rests on."""
        return (bool(a.get("has_arm", False))
                and e.get("item_height", 0.0) <= a.get("reach_height", self.default_reach_height))

    def fits_gripper(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """True if the item's grip dimension is within the gripper's opening."""
        return (bool(a.get("has_arm", False))
                and e.get("item_width", 0.0) <= a.get("gripper_opening", self.default_gripper_opening))

    def within_payload(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """True if the item's mass is within the arm's payload."""
        return (bool(a.get("has_arm", False))
                and e.get("item_mass", 0.0) <= a.get("payload", self.default_payload))


#: Shared default policy. Pass an explicit ``NavCostConfig`` to override it.
DEFAULT_CONFIG = NavCostConfig()
