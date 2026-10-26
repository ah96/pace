"""
scenarios/plant.py
==================
A remanufacturing plant: the industrial scenario the paper is built around.

The setting
-----------
A plant that takes back end-of-life products - traction batteries and power
electronics - and turns them into modules and parts for a second life.  Cores
arrive at the dock in unknown condition, are triaged in receiving, tested in
diagnostics, dismantled in two disassembly cells, sorted by hand, washed,
stored as modules and shipped.  Damaged packs wait in a quarantine zone.  A
heterogeneous fleet of autonomous mobile robots (AMRs) moves everything between
those zones, and the plant changes daily: pallets of returned goods are staged
in aisles, zones are cordoned during incidents, roller and fire doors are
closed, aisles are shut for floor repairs.  Most of what stops a robot is
therefore something a person on the floor can change in minutes; some of it is
the building.  Telling the two apart, per robot, is what PACE is for.

Layout (top-down, metres; zone centres 12 m apart, rows 16 m apart)
-------------------------------------------------------------------
  y=24  receiving --3.5--- diagnostics --3.0-- disassembly_A --3.0*-- disassembly_B --3.0c-- sorting
           |                    |                    |                     |                  |
        roller door        restricted            wet, dark             closed aisle        fire door
           |                    |                    |                     |                  |
  y=8    dock -----hazard--- quarantine ---2.5--- wash ---VNA 1.5--- module_store --1.6x1.9-- shipping
           |
        ramp 7 deg
           |
  y=-8   yard

  *  staged pallets take 1.4 m of the 3.0 m aisle     c  crowded (manual sorting)

Every number that decides feasibility here is also what the Gazebo twin builds
(:mod:`pace.twin.plant_world`): racking stands exactly ``width`` apart, the
mezzanine slab over the underpass sits at ``height``, the dock ramp is inclined
by ``slope_angle``, staged pallets occupy ``blocked_width`` of the aisle, doors
are closed, cordons are up.  The semantic world and the physical one are two
renderings of the one :class:`PlantSpec` below.

Robots
------
Four AMR classes, modelled on commercial platforms but parametrised only by what
the affordance rules read: body width and height, side clearance, rated slope,
locomotion, and whether the robot can operate a door.

  tote_amr            0.60 x 0.95 m, 3 deg, wheeled       - a MiR100-class tote carrier
  forklift_agv        1.20 x 2.20 m, 3 deg, wheeled       - an autonomous counterbalance
                                                            forklift, mast up, 3.2 m long
  mobile_manipulator  0.54 x 1.45 m, 10 deg, wheeled, arm - PAL's TIAGo: opens doors, picks
  yard_ugv            0.70 x 0.60 m, 20 deg, wheeled      - a Husky-class rugged UGV for
                                                            inspection rounds and the yard
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pace import (
    DoorState,
    MobilityType,
    NavEdge,
    NavigationGraph,
    NavNode,
    SurfaceType,
    build_navigation_ontology,
)

#: Half the side of a zone's open floor (zones are ZONE_HALF*2 m squares); the
#: passage between two zones runs from one zone's edge to the other's.
ZONE_HALF = 3.0

#: Robots of the fleet.  ``length`` and ``laser_height`` are read only by the
#: Gazebo twin (footprint and sensor placement); the affordance rules see the
#: rest.
ROBOTS: Dict[str, Dict[str, object]] = {
    "tote_amr": dict(
        robot_width=0.60, robot_height=0.95, min_clearance=0.10, max_slope_angle=3.0,
        mobility_type=MobilityType.WHEELED.value, can_open_doors=False, has_arm=False,
        battery_level=0.9, max_speed=1.5, length=0.90, laser_height=0.20),
    "forklift_agv": dict(
        robot_width=1.20, robot_height=2.20, min_clearance=0.25, max_slope_angle=3.0,
        mobility_type=MobilityType.WHEELED.value, can_open_doors=False, has_arm=False,
        battery_level=0.9, max_speed=1.2, length=3.20, laser_height=0.30),
    "mobile_manipulator": dict(
        robot_width=0.54, robot_height=1.45, min_clearance=0.15, max_slope_angle=10.0,
        mobility_type=MobilityType.WHEELED.value, can_open_doors=True, has_arm=True,
        reach_height=1.75, gripper_opening=0.08, payload=3.0,
        battery_level=0.9, max_speed=1.0, length=0.75, laser_height=0.12),
    "yard_ugv": dict(
        robot_width=0.70, robot_height=0.60, min_clearance=0.15, max_slope_angle=20.0,
        mobility_type=MobilityType.WHEELED.value, can_open_doors=False, has_arm=False,
        battery_level=0.9, max_speed=1.0, length=1.00, laser_height=0.35),
}

#: Properties the ontology schema declares; the rest of a ROBOTS entry is the twin's.
_ROBOT_SCHEMA_PROPS = ("robot_width", "robot_height", "min_clearance", "max_slope_angle",
                       "mobility_type", "can_open_doors", "has_arm", "reach_height",
                       "gripper_opening", "payload", "battery_level", "max_speed")


@dataclass
class Zone:
    """A named open area of the plant, one waypoint of the navigation graph."""
    id: str
    x: float
    y: float
    label: str
    kind: str = "hall"          # what the twin furnishes it as (dock, cell, store, ...)
    level: float = 0.0          # floor height (the yard is below the dock)


@dataclass
class Segment:
    """A passage between two zones: one space individual, one bidirectional edge."""
    id: str
    a: str
    b: str
    class_name: str             # ontology class: Corridor, Doorway, Ramp, ...
    kind: str                   # what the twin builds: aisle, vna, roller_door, fire_door,
                                # underpass, ramp, cordon, ...
    props: Dict[str, object] = field(default_factory=dict)


@dataclass
class Item:
    """A workpiece to pick up: the last leg of a mission that ends at it.

    ``height`` is the surface it rests on, ``width`` its grip dimension (the
    thickness at the grasp point, e.g. a module's handle), ``mass`` its mass.
    ``offset`` places it in the zone for the twin (metres from the zone centre).
    ``stand`` names what holds it in the twin ("bench", "rack" or "pack"); left
    empty, it follows from the kind and height.
    """
    id: str
    zone: str
    label: str
    height: float
    width: float
    mass: float
    kind: str = "module"
    offset: Tuple[float, float] = (0.0, 2.2)
    stand: str = ""

    @property
    def node(self) -> str:
        return f"pick_{self.id}"


@dataclass
class PlantSpec:
    name: str
    zones: List[Zone]
    segments: List[Segment]
    items: List[Item] = field(default_factory=list)

    def zone(self, zid: str) -> Zone:
        return next(z for z in self.zones if z.id == zid)

    def bounds(self, margin: float = ZONE_HALF + 2.0) -> Tuple[float, float, float, float]:
        xs = [z.x for z in self.zones]
        ys = [z.y for z in self.zones]
        return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


# ---------------------------------------------------------------------------
# Defaults for a passage nobody has done anything to
# ---------------------------------------------------------------------------

def _aisle(**over) -> Dict[str, object]:
    base = dict(
        width=3.0, height=4.5,
        surface_type=SurfaceType.SMOOTH.value, illumination=0.85,
        obstacle_density=0.0, crowd_density=0.0,
        is_accessible=True, is_hazardous=False, restricted=False,
    )
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# The plant of the paper
# ---------------------------------------------------------------------------

def plant_spec() -> PlantSpec:
    """The authored remanufacturing plant, as drawn in the module docstring."""
    Y_N, Y_S, Y_Y = 24.0, 8.0, -8.0
    X = [6.0, 18.0, 30.0, 42.0, 54.0]
    zones = [
        Zone("receiving",     X[0], Y_N, "Receiving / triage",      "triage"),
        Zone("diagnostics",   X[1], Y_N, "Diagnostics",             "test"),
        Zone("disassembly_A", X[2], Y_N, "Disassembly cell A",      "cell"),
        Zone("disassembly_B", X[3], Y_N, "Disassembly cell B",      "cell"),
        Zone("sorting",       X[4], Y_N, "Manual sorting",          "sorting"),
        Zone("dock",          X[0], Y_S, "Inbound dock",            "dock"),
        Zone("quarantine",    X[1], Y_S, "Quarantine (damaged packs)", "quarantine"),
        Zone("wash",          X[2], Y_S, "Component washing",       "wash"),
        Zone("module_store",  X[3], Y_S, "Module store",            "store"),
        Zone("shipping",      X[4], Y_S, "Shipping",                "shipping"),
        Zone("yard",          X[0], Y_Y, "Yard",                    "yard", level=-1.2),
    ]
    segments = [
        # ---- north hall: the process line -------------------------------
        Segment("seg_recv_diag", "receiving", "diagnostics", "Corridor", "aisle",
                _aisle(width=3.5, illumination=0.9, crowd_density=0.1, emergency_route=True)),
        Segment("seg_diag_disA", "diagnostics", "disassembly_A", "Corridor", "aisle",
                _aisle(width=3.0)),
        # Pallets of returned cores are staged along one side of the aisle:
        # 1.4 of its 3.0 m are gone until somebody moves them.
        Segment("seg_disA_disB", "disassembly_A", "disassembly_B", "Corridor", "aisle",
                _aisle(width=3.0, blocked_width=1.4, obstacle_density=0.2)),
        # Manual sorting stations line the aisle: people everywhere.
        Segment("seg_disB_sort", "disassembly_B", "sorting", "Corridor", "aisle",
                _aisle(width=3.0, crowd_density=0.6)),
        # ---- south hall: logistics -------------------------------------
        # A leaking pack: the aisle is cordoned off until the spill is cleared.
        Segment("seg_dock_quar", "dock", "quarantine", "Corridor", "cordon",
                _aisle(width=2.5, height=4.0, is_hazardous=True)),
        Segment("seg_quar_wash", "quarantine", "wash", "Corridor", "aisle",
                _aisle(width=2.5, height=4.0, illumination=0.8)),
        # A very-narrow-aisle rack line.
        Segment("seg_wash_store", "wash", "module_store", "Corridor", "vna",
                _aisle(width=1.5, height=4.0, illumination=0.7)),
        # The aisle runs under the mezzanine slab.
        Segment("seg_store_ship", "module_store", "shipping", "Corridor", "underpass",
                _aisle(width=1.6, height=1.9, illumination=0.6)),
        # ---- between the halls -----------------------------------------
        Segment("seg_dock_recv", "dock", "receiving", "Doorway", "roller_door",
                _aisle(width=3.0, height=3.5, door_state=DoorState.CLOSED.value)),
        # Thermal monitoring of damaged packs: authorised staff only.
        Segment("seg_quar_diag", "quarantine", "diagnostics", "Corridor", "restricted",
                _aisle(width=2.5, height=4.0, restricted=True)),
        # Wash-down water on the floor, half the lights out.
        Segment("seg_wash_disA", "wash", "disassembly_A", "Corridor", "aisle",
                _aisle(width=2.5, height=4.0, surface_type=SurfaceType.WET.value, illumination=0.2)),
        # Closed for floor repair.
        Segment("seg_store_disB", "module_store", "disassembly_B", "Corridor", "closed",
                _aisle(width=2.0, height=4.0, is_accessible=False)),
        # A fire door, normally closed.
        Segment("seg_ship_sort", "shipping", "sorting", "Doorway", "fire_door",
                _aisle(width=1.8, height=2.2, door_state=DoorState.CLOSED.value)),
        # ---- outside ---------------------------------------------------
        Segment("seg_dock_yard", "dock", "yard", "Ramp", "ramp",
                _aisle(width=3.0, height=5.0, slope_angle=7.0,
                       surface_type=SurfaceType.ROUGH.value, illumination=0.75)),
    ]
    items = [
        # a module on the disassembly bench: within reach, grip and payload
        Item("module_bench", "disassembly_A", "Module on bench A", height=0.85, width=0.03, mass=2.4,
             offset=(0.0, 2.3)),
        # the same kind of module on the upper level of the rack (the third beam
        # level, 3.0 m, plus its pallet): out of reach; restocking it on the
        # middle level (1.65 m) brings it within the arm's 1.75 m
        Item("module_rack", "module_store", "Module on upper rack", height=3.15, width=0.03, mass=2.4,
             offset=(0.0, -2.4)),
        # a whole damaged pack in quarantine: far beyond any arm's payload
        Item("pack_quarantine", "quarantine", "Damaged pack", height=0.45, width=0.03, mass=24.0,
             kind="pack", offset=(0.0, -2.3)),
        # a tote of sorted parts at the sorting stations
        Item("tote_sorted", "sorting", "Tote of sorted parts", height=0.9, width=0.03, mass=1.5,
             kind="tote", offset=(0.0, 2.3)),
    ]
    return PlantSpec("remanufacturing_plant", zones, segments, items)


# ---------------------------------------------------------------------------
# Spec -> semantic world
# ---------------------------------------------------------------------------

def segment_length(spec: PlantSpec, seg: Segment) -> float:
    """Length of the passage between two zone edges (centre distance minus the
    two half-zones), in the plane.  The graph's edge distance is centre to
    centre, as in every scenario; this is what the twin builds."""
    za, zb = spec.zone(seg.a), spec.zone(seg.b)
    return max(1.0, math.hypot(za.x - zb.x, za.y - zb.y) - 2 * ZONE_HALF)


def build_world_from_spec(spec: PlantSpec, robots: Optional[Dict[str, Dict[str, object]]] = None):
    """Construct (ontology, navigation graph, agents) from a :class:`PlantSpec`."""
    onto = build_navigation_ontology()
    graph = NavigationGraph()
    for z in spec.zones:
        graph.add_node(NavNode(node_id=z.id, position=(z.x, z.y), label=z.label))
    for seg in spec.segments:
        props = dict(seg.props)
        props.setdefault("length", round(segment_length(spec, seg), 2))
        individual = onto.create(seg.id, seg.class_name, **props)
        za, zb = spec.zone(seg.a), spec.zone(seg.b)
        graph.add_edge(
            NavEdge(from_id=seg.a, to_id=seg.b, space_individual=individual,
                    distance=math.hypot(za.x - zb.x, za.y - zb.y)),
            bidirectional=True,
        )
    for it in spec.items:
        z = spec.zone(it.zone)
        graph.add_node(NavNode(node_id=it.node, position=(z.x + it.offset[0], z.y + it.offset[1]),
                               label=it.label))
        individual = onto.create(it.id, "Item", item_height=it.height, item_width=it.width,
                                 item_mass=it.mass, located_at=it.zone)
        # one way: a mission ends at the item, nobody drives through it
        graph.add_edge(NavEdge(from_id=it.zone, to_id=it.node, space_individual=individual,
                               distance=max(0.5, math.hypot(*it.offset))), bidirectional=False)
    agents = {}
    for name, spec_r in (robots or ROBOTS).items():
        agents[name] = onto.create(name, "Robot",
                                   **{k: v for k, v in spec_r.items() if k in _ROBOT_SCHEMA_PROPS})
    return onto, graph, agents


def build_plant_world():
    """The paper's plant: (ontology, graph, agents)."""
    return build_world_from_spec(plant_spec())


# ---------------------------------------------------------------------------
# A family of plants nobody tuned: sampled layouts and operating states
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlantPriors:
    """Design ranges and operating-state frequencies for generated plants.

    Widths follow the aisle classes of pallet-racked warehouses: very narrow
    aisles for guided trucks, narrow aisles for reach trucks, wide aisles for
    counterbalance trucks and mixed traffic.  Everything else is an operating
    state a shift may or may not be in.  Fixed before any result was seen and
    referring to no robot, affordance or repair.
    """
    p_vna: float = 0.15;      vna: Tuple[float, float] = (1.4, 1.8)
    p_narrow: float = 0.25;   narrow: Tuple[float, float] = (1.9, 2.4)
    p_standard: float = 0.45; standard: Tuple[float, float] = (2.5, 3.5)
    wide: Tuple[float, float] = (3.6, 4.5)
    hall_height: Tuple[float, float] = (4.0, 6.0)
    p_underpass: float = 0.08; underpass: Tuple[float, float] = (1.8, 2.4)
    p_door: float = 0.20;     door_closed: float = 0.6; door_locked: float = 0.1
    door_width: Tuple[float, float] = (1.6, 3.2); door_height: Tuple[float, float] = (2.1, 3.6)
    p_ramp: float = 0.05;     ramp_slope: Tuple[float, float] = (3.0, 12.0)
    p_staged: float = 0.20;   staged_frac: Tuple[float, float] = (0.2, 0.6)
    p_restricted: float = 0.06
    p_hazard: float = 0.04
    p_closed: float = 0.05
    p_crowded: float = 0.25;  crowd_hi: Tuple[float, float] = (0.3, 0.8); crowd_lo: Tuple[float, float] = (0.0, 0.15)
    p_clutter: float = 0.30;  clutter: Tuple[float, float] = (0.1, 0.5)
    p_dark: float = 0.10;     dark: Tuple[float, float] = (0.1, 0.25); lit: Tuple[float, float] = (0.6, 1.0)
    surfaces: Tuple[Tuple[str, float], ...] = (
        (SurfaceType.SMOOTH.value, 0.70), (SurfaceType.ROUGH.value, 0.15),
        (SurfaceType.WET.value, 0.10), (SurfaceType.GRAVEL.value, 0.05))
    p_emergency: float = 0.15
    p_passage: float = 0.80          # a grid passage exists with this probability
    items_per_plant: Tuple[int, int] = (2, 4)
    item_height: Tuple[Tuple[float, float], ...] = ((0.3, 1.0), (1.0, 1.4), (1.4, 2.0))   # cart/bench, high bench, upper rack
    item_height_p: Tuple[float, ...] = (0.55, 0.25, 0.20)
    item_width: Tuple[Tuple[float, float], ...] = ((0.02, 0.06), (0.08, 0.30))            # handle, bare housing
    item_width_p: Tuple[float, ...] = (0.7, 0.3)
    item_mass: Tuple[Tuple[float, float], ...] = ((0.5, 2.8), (3.5, 12.0), (15.0, 40.0))  # part, module, pack
    item_mass_p: Tuple[float, ...] = (0.5, 0.3, 0.2)


PRIORS = PlantPriors()

_ZONE_KINDS = ["triage", "test", "cell", "cell", "sorting", "store", "store", "wash",
               "shipping", "quarantine", "hall", "hall"]


def _pick(rng: random.Random, table) -> object:
    r, acc = rng.random(), 0.0
    for value, p in table:
        acc += p
        if r < acc:
            return value
    return table[-1][0]


def _sample_segment(rng: random.Random, sid: str, a: str, b: str, pri: PlantPriors,
                    outdoor: bool = False) -> Segment:
    """One passage with a sampled class, geometry and operating state."""
    props = _aisle()
    r = rng.random()
    if r < pri.p_vna:
        props["width"] = rng.uniform(*pri.vna)
    elif r < pri.p_vna + pri.p_narrow:
        props["width"] = rng.uniform(*pri.narrow)
    elif r < pri.p_vna + pri.p_narrow + pri.p_standard:
        props["width"] = rng.uniform(*pri.standard)
    else:
        props["width"] = rng.uniform(*pri.wide)
    props["height"] = rng.uniform(*pri.hall_height)
    class_name, kind = "Corridor", "aisle"
    if outdoor or rng.random() < pri.p_ramp:
        class_name, kind = "Ramp", "ramp"
        props["slope_angle"] = rng.uniform(*pri.ramp_slope)
        props["surface_type"] = _pick(rng, ((SurfaceType.ROUGH.value, 0.5),
                                            (SurfaceType.GRAVEL.value, 0.3),
                                            (SurfaceType.SMOOTH.value, 0.2)))
    elif rng.random() < pri.p_door:
        class_name, kind = "Doorway", "door"
        props["width"] = rng.uniform(*pri.door_width)
        props["height"] = rng.uniform(*pri.door_height)
        props["door_state"] = _pick(rng, ((DoorState.CLOSED.value, pri.door_closed),
                                          (DoorState.LOCKED.value, pri.door_locked),
                                          (DoorState.OPEN.value, 1 - pri.door_closed - pri.door_locked)))
    elif rng.random() < pri.p_underpass:
        kind = "underpass"
        props["height"] = rng.uniform(*pri.underpass)
    if kind in ("aisle", "underpass") and rng.random() < pri.p_staged:
        props["blocked_width"] = round(props["width"] * rng.uniform(*pri.staged_frac), 2)
    if rng.random() < pri.p_restricted:
        props["restricted"] = True
    if rng.random() < pri.p_hazard:
        props["is_hazardous"] = True
    if rng.random() < pri.p_closed:
        props["is_accessible"] = False
    props["crowd_density"] = round(rng.uniform(*(pri.crowd_hi if rng.random() < pri.p_crowded
                                                 else pri.crowd_lo)), 2)
    if rng.random() < pri.p_clutter:
        props["obstacle_density"] = round(rng.uniform(*pri.clutter), 2)
    props["illumination"] = round(rng.uniform(*(pri.dark if rng.random() < pri.p_dark else pri.lit)), 2)
    if kind != "ramp":
        props["surface_type"] = _pick(rng, pri.surfaces)
    if rng.random() < pri.p_emergency:
        props["emergency_route"] = True
    for key in ("width", "height", "slope_angle"):
        if key in props:
            props[key] = round(props[key], 2)
    return Segment(sid, a, b, class_name, kind, props)


def random_plant_spec(seed: int, rows: int = 2, cols: int = 5,
                      priors: PlantPriors = PRIORS, yard: bool = True) -> PlantSpec:
    """A generated plant on a ``rows`` x ``cols`` grid of zones, 12 m apart in x
    and 16 m apart in y, with a yard below the first column.  Grid passages are
    kept with probability ``priors.p_passage`` and the result is made connected
    by re-adding passages along a spanning tree; every passage's class,
    geometry and state come from :func:`_sample_segment`."""
    rng = random.Random(seed)
    zones: List[Zone] = []
    for r in range(rows):
        for c in range(cols):
            zid = f"z{r}{c}"
            zones.append(Zone(zid, 6.0 + 12.0 * c, 8.0 + 16.0 * r, f"Zone {r}{c}",
                              kind=rng.choice(_ZONE_KINDS)))
    if yard:
        zones.append(Zone("yard", 6.0, 8.0 - 16.0, "Yard", "yard", level=-1.2))
    candidates: List[Tuple[str, str]] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                candidates.append((f"z{r}{c}", f"z{r}{c + 1}"))
            if r + 1 < rows:
                candidates.append((f"z{r}{c}", f"z{r + 1}{c}"))
    kept = [e for e in candidates if rng.random() < priors.p_passage]
    # connectivity: union-find over kept edges, then add missing candidates in
    # a shuffled order until one component remains
    parent = {z.id: z.id for z in zones if z.id != "yard"}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in kept:
        parent[find(a)] = find(b)
    rest = [e for e in candidates if e not in kept]
    rng.shuffle(rest)
    for a, b in rest:
        if find(a) != find(b):
            parent[find(a)] = find(b)
            kept.append((a, b))
    kept.sort(key=lambda e: candidates.index(e))
    segments = [_sample_segment(rng, f"seg_{a}_{b}", a, b, priors) for a, b in kept]
    if yard:
        segments.append(_sample_segment(rng, "seg_z00_yard", "z00", "yard", priors, outdoor=True))
    items: List[Item] = []
    n_items = rng.randint(*priors.items_per_plant)
    indoor = [z.id for z in zones if z.kind != "yard"]
    for k, zid in enumerate(rng.sample(indoor, min(n_items, len(indoor)))):
        h = rng.uniform(*_pick(rng, tuple(zip(priors.item_height, priors.item_height_p))))
        w = rng.uniform(*_pick(rng, tuple(zip(priors.item_width, priors.item_width_p))))
        m = rng.uniform(*_pick(rng, tuple(zip(priors.item_mass, priors.item_mass_p))))
        items.append(Item(f"item{k}", zid, f"Item {k}", round(h, 2), round(w, 3), round(m, 1)))
    return PlantSpec(f"plant_{seed}", zones, segments, items)


def build_random_plant(seed: int, **kw):
    """A generated plant as (ontology, graph, agents)."""
    return build_world_from_spec(random_plant_spec(seed, **kw))


# ---------------------------------------------------------------------------
# Command line: the three answers to one question
# ---------------------------------------------------------------------------

def explain_task(onto, graph, agent, start: str, goal: str, k: int = 3):
    """PACE on one task, as the fleet manager sees it: the chosen route if there
    is one, and for every questioned route what would have to change.

    Returns ``(path, [(route_nodes, Counterfactual)])``.  Infeasible tasks are
    the ones the paper is about, and :meth:`PaceNavigator.navigate` has no
    chosen route to explain there, so the questioned routes are enumerated on
    the topological graph exactly as the evaluation does.
    """
    from pace import PaceNavigator
    from pace.evaluation import questioned_routes
    nav = PaceNavigator(onto, graph)
    path, _ = nav.navigate(start, goal, agent, k_alternatives=k, explain=False)
    out = []
    for seq in questioned_routes(graph, start, goal, k):
        if tuple(seq) == tuple(path.nodes):
            continue
        out.append((seq, nav.cf_engine.explain_why_not(path, seq, agent)))
    return path, out


def run() -> None:
    """Ask, for every robot, how it gets from the dock to shipping and, where
    it cannot, what would open each route.  ``python -m scenarios.plant``."""
    onto, graph, agents = build_plant_world()
    for name, agent in agents.items():
        print("\n" + "=" * 72 + f"\n  {name}: dock -> shipping\n" + "=" * 72)
        path, cfs = explain_task(onto, graph, agent, "dock", "shipping")
        if path.is_feasible:
            print(f"  route: {' > '.join(path.nodes)}  (cost {path.total_cost:.1f})")
        else:
            print("  no route")
        for seq, cf in cfs:
            print(f"\n  why not {' > '.join(seq)}?")
            if not cf.changes:
                print("    nothing to change" if cf.alternative_path.is_feasible
                      else "    no verified repair set within the questioned route")
            for c in cf.changes:
                tag = "actionable" if c.is_actionable() else "STRUCTURAL"
                print(f"    [{c.claim:6s} {tag:10s} {c.effort():6s}] {c.description()}")


if __name__ == "__main__":
    run()
