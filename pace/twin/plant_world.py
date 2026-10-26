"""The plant as a Gazebo (classic 11) world: the digital twin of a :class:`PlantSpec`.

One spec, two renderings.  :mod:`scenarios.plant` turns the spec into the
ontology and navigation graph PACE reasons over; this module turns the same
spec into an SDF world in which every number the affordance rules read is
built:

* a passage of ``width`` is flanked by racking (or walls) exactly ``width``
  apart, and nothing else connects two zones, so the world's topology is the
  graph's;
* ``blocked_width`` metres of an aisle are taken up by staged pallets along one
  side, for the length of the aisle;
* an ``underpass`` runs under a mezzanine slab whose underside is at ``height``;
* a ``Doorway`` has a roller or fire door across it, closed or open as
  ``door_state`` says;
* a hazardous, restricted or inaccessible passage is cordoned off at both ends
  with barriers (red-white for a hazard, yellow-black for a restriction, a
  closed-aisle barrier otherwise);
* a ``Ramp`` is inclined by ``slope_angle`` and leads down to the yard;
* crowd and obstacle density place people and clutter along the sides, out of
  the driving lane, since in the semantic model they cost but do not block;
* illumination places a light over the passage or leaves it dark; a wet
  surface is drawn as such, an emergency route gets its green markings.

The *repaired* world is the same generator run on a spec with PACE's property
changes applied (:func:`apply_changes`): the door leaf moves, the pallets go,
the cordon comes down, the racking moves apart, the slab rises.  Beyond the
world file the module writes the robot's prior map for Nav2 (the static
geometry rasterised at 5 cm) and the zone poses to navigate to.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scenarios.plant import ZONE_HALF, Item, PlantSpec, Segment, Zone, segment_length
from . import sdf as S

# ------------------------------------------------------------------ colours
CONCRETE = "#a9a7a1"
CONCRETE_DARK = "#8f8d88"
YARD = "#7a7268"
WALL = "#dcd8cf"
UPRIGHT = "#2b4f8f"          # RAL 5010-ish blue racking uprights
BEAM = "#e26d1a"             # orange beams
PALLET = "#a8754a"
BOX = "#c9a06c"
SLAB = "#9aa0a6"
ROLLER = "#6d7f8f"
FIRE_DOOR = "#8b2e2e"
HAZARD_A, HAZARD_B = "#d62728", "#f4f4f2"
RESTRICT_A, RESTRICT_B = "#f2c319", "#202022"
CLOSED_A, CLOSED_B = "#f2c319", "#f4f4f2"
EMERGENCY = "#2ca02c"
WET = "#6f8fb3"
PEOPLE = ("person", "nurse", "scrubs")

RACK_DEPTH = 1.1
RACK_BAY = 2.7
RACK_H = 4.0
WALL_T = 0.3
HALL_WALL_H = 5.0


@dataclass
class Look:
    """One box of scenery for the vendored Fuel/procedural builders in :mod:`.sdf`."""
    gazebo_label: str
    label: str
    cx: float
    cy: float
    dx: float
    dy: float
    yaw: float = 0.0
    height: float = 1.0
    openability: bool = False
    is_open: bool = False
    hinge: tuple | None = None
    is_human: bool = False
    model: str | None = None

    def footprint(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        local = np.array([(-self.dx / 2, -self.dy / 2), (self.dx / 2, -self.dy / 2),
                          (self.dx / 2, self.dy / 2), (-self.dx / 2, self.dy / 2)])
        return local @ np.array([[c, s], [-s, c]]) + np.array([self.cx, self.cy])

    def open_pose(self):
        return self.cx, self.cy, self.yaw


# ------------------------------------------------------------- geometry

@dataclass
class Rect:
    """An axis-aligned rectangle on the floor, [x0, x1] x [y0, y1]."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2
    @property
    def dx(self): return self.x1 - self.x0
    @property
    def dy(self): return self.y1 - self.y0

    def ok(self, eps: float = 0.05) -> bool:
        return self.dx > eps and self.dy > eps


@dataclass
class Corridor:
    """Where a segment is built: its rectangle and its axis ('x' or 'y')."""
    seg: Segment
    rect: Rect
    axis: str
    za: Zone
    zb: Zone

    @property
    def length(self) -> float:
        return self.rect.dx if self.axis == "x" else self.rect.dy

    @property
    def width(self) -> float:
        return self.rect.dy if self.axis == "x" else self.rect.dx

    def along(self, t: float, off: float = 0.0) -> Tuple[float, float]:
        """A point at fraction ``t`` of the length, ``off`` metres to the side."""
        if self.axis == "x":
            return self.rect.x0 + t * self.rect.dx, self.rect.cy + off
        return self.rect.cx + off, self.rect.y0 + t * self.rect.dy

    @property
    def yaw(self) -> float:
        return 0.0 if self.axis == "x" else math.pi / 2


def corridors(spec: PlantSpec) -> Dict[str, Corridor]:
    """The rectangle of every passage: between the two zones' edges, ``width`` wide."""
    out: Dict[str, Corridor] = {}
    for seg in spec.segments:
        za, zb = spec.zone(seg.a), spec.zone(seg.b)
        w = float(seg.props.get("width", 3.0))
        if abs(za.y - zb.y) < 1e-6:                       # along x
            lo, hi = sorted((za.x, zb.x))
            rect = Rect(lo + ZONE_HALF, za.y - w / 2, hi - ZONE_HALF, za.y + w / 2)
            axis = "x"
        else:
            lo, hi = sorted((za.y, zb.y))
            rect = Rect(za.x - w / 2, lo + ZONE_HALF, za.x + w / 2, hi - ZONE_HALF)
            axis = "y"
        a, b = (za, zb) if (za.x, za.y) <= (zb.x, zb.y) else (zb, za)
        out[seg.id] = Corridor(seg, rect, axis, a, b)
    return out


def _grid(spec: PlantSpec):
    indoor = [z for z in spec.zones if z.kind != "yard"]
    xs = sorted({z.x for z in indoor})
    ys = sorted({z.y for z in indoor})
    return indoor, xs, ys


def solid_bands(spec: PlantSpec, cors: Dict[str, Corridor]) -> List[Tuple[Rect, Dict[str, Optional[Corridor]]]]:
    """Everything inside the hall that is neither a zone nor a passage, as
    rectangles, each with the passage (if any) that touches each of its faces
    ('S', 'N', 'W', 'E').  The faces decide the finish: racking towards an
    aisle, a wall towards a doorway, nothing towards the outside."""
    indoor, xs, ys = _grid(spec)
    H = ZONE_HALF

    def hcor(i: int, j: int) -> Optional[Corridor]:          # between column i and i+1 at row j
        for c in cors.values():
            if c.axis == "x" and abs(c.rect.cy - ys[j]) < 1e-6 and abs(c.rect.x0 - (xs[i] + H)) < 1e-6:
                return c
        return None

    def vcor(i: int, j: int) -> Optional[Corridor]:          # between row j and j+1 at column i
        for c in cors.values():
            if c.axis == "y" and abs(c.rect.cx - xs[i]) < 1e-6 and abs(c.rect.y0 - (ys[j] + H)) < 1e-6:
                return c
        return None

    def hw(c: Optional[Corridor]) -> float:
        return c.width / 2 if c else 0.0

    bands: List[Tuple[Rect, Dict[str, Optional[Corridor]]]] = []
    C, R = len(xs), len(ys)
    # interior blocks between rows j, j+1 and columns i, i+1
    for j in range(R - 1):
        for i in range(C - 1):
            l, r = vcor(i, j), vcor(i + 1, j)
            b, t = hcor(i, j), hcor(i, j + 1)
            mid = Rect(xs[i] + hw(l), ys[j] + H, xs[i + 1] - hw(r), ys[j + 1] - H)
            bot = Rect(xs[i] + H, ys[j] + hw(b), xs[i + 1] - H, ys[j] + H)
            top = Rect(xs[i] + H, ys[j + 1] - H, xs[i + 1] - H, ys[j + 1] - hw(t))
            bands.append((mid, {"W": l, "E": r, "S": None, "N": None}))
            bands.append((bot, {"S": b, "N": None, "W": None, "E": None}))
            bands.append((top, {"N": t, "S": None, "W": None, "E": None}))
    # outer strips beside the outermost rows and columns
    for i in range(C - 1):
        b = hcor(i, 0)
        bands.append((Rect(xs[i] + H, ys[0] - H, xs[i + 1] - H, ys[0] - hw(b)), {"N": b, "S": None, "W": None, "E": None}))
        t = hcor(i, R - 1)
        bands.append((Rect(xs[i] + H, ys[-1] + hw(t), xs[i + 1] - H, ys[-1] + H), {"S": t, "N": None, "W": None, "E": None}))
    for j in range(R - 1):
        l = vcor(0, j)
        bands.append((Rect(xs[0] - H, ys[j] + H, xs[0] - hw(l), ys[j + 1] - H), {"E": l, "W": None, "S": None, "N": None}))
        r = vcor(C - 1, j)
        bands.append((Rect(xs[-1] + hw(r), ys[j] + H, xs[-1] + H, ys[j + 1] - H), {"W": r, "E": None, "S": None, "N": None}))
    return [(b, f) for b, f in bands if b.ok()]


def hall_rect(spec: PlantSpec) -> Rect:
    _, xs, ys = _grid(spec)
    return Rect(xs[0] - ZONE_HALF, ys[0] - ZONE_HALF, xs[-1] + ZONE_HALF, ys[-1] + ZONE_HALF)


# ----------------------------------------------------------- SDF pieces

def _box(name: str, cx: float, cy: float, cz: float, dx: float, dy: float, dz: float,
         colour: str, yaw: float = 0.0, pitch: float = 0.0, emissive: float = 0.0,
         transparency: float = 0.0) -> str:
    return S.box_visual(name, f"{cx:.4f} {cy:.4f} {cz:.4f} 0 {pitch:.5f} {yaw:.5f}", (dx, dy, dz),
                        S.material(colour, emissive=emissive, transparency=transparency))


def _collision(name: str, cx: float, cy: float, cz: float, dx: float, dy: float, dz: float,
               yaw: float = 0.0, pitch: float = 0.0) -> str:
    return (f'<collision name="{name}"><pose>{cx:.4f} {cy:.4f} {cz:.4f} 0 {pitch:.5f} {yaw:.5f}</pose>'
            f"<geometry><box><size>{dx:.4f} {dy:.4f} {dz:.4f}</size></box></geometry></collision>")


def _model(name: str, body: str, static: bool = True, pose: str = "0 0 0 0 0 0") -> str:
    return (f'<model name="{name}"><static>{"true" if static else "false"}</static><pose>{pose}</pose>'
            f'<link name="link">{body}</link></model>')


def rack_row(name: str, x0: float, y0: float, x1: float, y1: float, rng: random.Random,
             height: float = RACK_H, fill: float = 0.8) -> str:
    """Pallet racking filling the rectangle: uprights per bay along the long
    side, beams at three levels, pallets with boxes where ``fill`` says."""
    dx, dy = x1 - x0, y1 - y0
    along_x = dx >= dy
    L, D = (dx, dy) if along_x else (dy, dx)
    n_bays = max(1, int(round(L / RACK_BAY)))
    bay = L / n_bays
    parts: List[str] = []
    levels = [0.0, 1.5, 3.0] if height >= 3.6 else [0.0, 1.5]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def at(u: float, v: float) -> Tuple[float, float]:
        """u along the row from its start, v across from its centre."""
        return (x0 + u, cy + v) if along_x else (cx + v, y0 + u)

    for k in range(n_bays + 1):
        u = min(L - 0.05, max(0.05, k * bay))
        for v in (-D / 2 + 0.06, D / 2 - 0.06):
            px, py = at(u, v)
            parts.append(_box(f"up{k}{'a' if v < 0 else 'b'}", px, py, height / 2, 0.1, 0.1, height, UPRIGHT))
    for li, z in enumerate(levels[1:] + [height]):
        for v in (-D / 2 + 0.06, D / 2 - 0.06):
            px, py = at(L / 2, v)
            size = (L, 0.08, 0.12) if along_x else (0.08, L, 0.12)
            parts.append(_box(f"beam{li}{'a' if v < 0 else 'b'}", px, py, z - 0.06, *size, BEAM))
    for k in range(n_bays):
        for li, z in enumerate(levels):
            if rng.random() > fill:
                continue
            u = (k + 0.5) * bay
            px, py = at(u, 0.0)
            pl = (min(bay - 0.4, 2.4), D - 0.2) if along_x else (D - 0.2, min(bay - 0.4, 2.4))
            parts.append(_box(f"pal{k}{li}", px, py, z + 0.075, pl[0], pl[1], 0.15, PALLET))
            bh = rng.uniform(0.6, 1.1)
            shade = rng.choice((BOX, "#b58c5a", "#d4b483", "#8d8d8d"))
            parts.append(_box(f"box{k}{li}", px, py, z + 0.15 + bh / 2, pl[0] * 0.92, pl[1] * 0.85, bh, shade))
    return f'<model name="{name}"><static>true</static><link name="look">{"".join(parts)}</link></model>'


def block_model(name: str, rect: Rect, faces: Dict[str, Optional[Corridor]], rng: random.Random) -> str:
    """One solid band: a collision box for all of it, racking along the faces
    that meet an aisle, a wall along the faces that meet a doorway, and a low
    grey block for whatever is left."""
    parts = [_collision("c", rect.cx, rect.cy, RACK_H / 2, rect.dx, rect.dy, RACK_H)]
    inner = Rect(rect.x0, rect.y0, rect.x1, rect.y1)
    racks: List[str] = []
    for face, cor in faces.items():
        if cor is None:
            continue
        kind = cor.seg.kind
        wallish = kind in ("roller_door", "fire_door", "door", "ramp", "cordon", "restricted", "closed")
        depth = min(RACK_DEPTH if not wallish else WALL_T, (rect.dy if face in "SN" else rect.dx) - 0.05)
        if depth <= 0.05:
            continue
        if face == "S":
            r = Rect(rect.x0, rect.y0, rect.x1, rect.y0 + depth); inner.y0 = max(inner.y0, r.y1)
        elif face == "N":
            r = Rect(rect.x0, rect.y1 - depth, rect.x1, rect.y1); inner.y1 = min(inner.y1, r.y0)
        elif face == "W":
            r = Rect(rect.x0, rect.y0, rect.x0 + depth, rect.y1); inner.x0 = max(inner.x0, r.x1)
        else:
            r = Rect(rect.x1 - depth, rect.y0, rect.x1, rect.y1); inner.x1 = min(inner.x1, r.x0)
        if wallish:
            parts.append(_box(f"wall{face}", r.cx, r.cy, HALL_WALL_H / 2, r.dx, r.dy, HALL_WALL_H, WALL))
        else:
            racks.append(rack_row(f"{name}_rack{face}", r.x0, r.y0, r.x1, r.y1, rng))
    if inner.ok(0.3):
        parts.append(_box("core", inner.cx, inner.cy, 1.25, inner.dx, inner.dy, 2.5, CONCRETE_DARK))
        parts.append(_box("cap", inner.cx, inner.cy, 2.52, inner.dx + 0.02, inner.dy + 0.02, 0.04, "#5c5f66"))
    return _model(name, "".join(parts)) + "".join(racks)


def outer_walls(spec: PlantSpec, cors: Dict[str, Corridor]) -> str:
    """The hall's perimeter, with an opening where a passage leaves it (the ramp)."""
    h = hall_rect(spec)
    openings: List[Tuple[str, float, float]] = []       # (side, lo, hi) along that side
    for c in cors.values():
        if c.seg.kind == "ramp" or spec.zone(c.seg.a).kind == "yard" or spec.zone(c.seg.b).kind == "yard":
            z = c.za if c.za.kind == "yard" else c.zb
            other = c.zb if z is c.za else c.za
            if z.y < h.y0:
                openings.append(("S", other.x - c.width / 2, other.x + c.width / 2))
    parts: List[str] = []
    t, H = WALL_T, HALL_WALL_H

    def wall(name, x0, y0, x1, y1):
        r = Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if not r.ok(0.02):
            return
        parts.append(_collision(f"c{name}", r.cx, r.cy, H / 2, r.dx, r.dy, H))
        parts.append(_box(f"w{name}", r.cx, r.cy, H / 2, r.dx, r.dy, H, WALL))

    # south wall, split by openings
    cuts = sorted(o[1:] for o in openings if o[0] == "S")
    x = h.x0 - t
    for k, (lo, hi) in enumerate(cuts):
        wall(f"s{k}", x, h.y0 - t, lo, h.y0)
        x = hi
    wall("s_end", x, h.y0 - t, h.x1 + t, h.y0)
    wall("n", h.x0 - t, h.y1, h.x1 + t, h.y1 + t)
    wall("w", h.x0 - t, h.y0, h.x0, h.y1)
    wall("e", h.x1, h.y0, h.x1 + t, h.y1)
    return _model("hall_walls", "".join(parts))


def floor_models(spec: PlantSpec, cors: Dict[str, Corridor]) -> List[str]:
    """The hall slab, the yard slab (lower), and the zone floor markings."""
    h = hall_rect(spec)
    out = [('<model name="ground"><static>true</static><link name="link"><collision name="c">'
            '<geometry><plane><normal>0 0 1</normal><size>400 400</size></plane></geometry>'
            '<surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface></collision>'
            + _box("slab", h.cx, h.cy, -0.01, h.dx + 2, h.dy + 2, 0.02, CONCRETE) + "</link></model>")]
    for z in spec.zones:
        if z.kind == "yard":
            yard = Rect(z.x - 8, z.y - 8, z.x + 8, z.y + ZONE_HALF)
            body = (_collision("c", yard.cx, yard.cy, z.level - 0.1, yard.dx, yard.dy, 0.2)
                    + _box("yard", yard.cx, yard.cy, z.level - 0.1, yard.dx, yard.dy, 0.2, YARD))
            out.append(_model("yard_slab", body))
        else:
            tint = {"dock": "#a9b7c6", "quarantine": "#e8c3c3", "wash": "#b9cfe0", "store": "#c9c4b6",
                    "shipping": "#a9b7c6", "cell": "#c6d3c0", "sorting": "#d8d1b8", "test": "#c8c9d8",
                    "triage": "#cfd6c4"}.get(z.kind, CONCRETE)
            out.append(_model(f"zone_{z.id}", _box("mark", z.x, z.y, 0.003, 2 * ZONE_HALF - 0.3,
                                                   2 * ZONE_HALF - 0.3, 0.006, tint)))
    return out


def _dims(c: Corridor, along: float, across: float) -> Tuple[float, float]:
    """(dx, dy) of a box ``along`` metres long in the corridor's direction and
    ``across`` metres wide across it."""
    return (along, across) if c.axis == "x" else (across, along)


def _across_rect(c: Corridor, t: float, thickness: float, span: Optional[float] = None) -> Rect:
    """The floor rectangle of something spanning the corridor at fraction ``t``."""
    w = span if span is not None else c.width
    mx, my = c.along(t)
    dx, dy = _dims(c, thickness, w)
    return Rect(mx - dx / 2, my - dy / 2, mx + dx / 2, my + dy / 2)


def door_models(c: Corridor, door_state: str, swing: Optional[Tuple[float, float]] = None) -> List[str]:
    """A roller door (one panel that rolls up) or a fire door (two leaves that
    swing back against the frame), across the middle of the passage.  For a
    still of a closed fire door being opened, ``swing`` gives the angle of the
    leaf at negative and at positive across offset (radians from the closed
    plane, positive towards the passage's end at ``t = 1``)."""
    w, h = c.width, float(c.seg.props.get("height", 3.0))
    mx, my = c.along(0.5)
    frame_t = 0.15
    parts = []
    for s in (-1, 1):                                   # posts, flush with the flanking walls
        px, py = c.along(0.5, s * (w / 2 - frame_t / 2))
        parts.append(_box(f"post{s + 1}", px, py, h / 2 + 0.15, *_dims(c, 0.25, frame_t), h + 0.3, "#4a4b4d"))
    parts.append(_box("lintel", mx, my, h + 0.15, *_dims(c, 0.25, w), 0.3, "#4a4b4d"))
    frame = _model(f"frame_{c.seg.id}", "".join(parts))
    leaf_t = 0.08
    opening = w - 2 * frame_t
    roller = c.seg.kind == "roller_door" or (c.seg.kind == "door" and w > 2.4)
    if roller:
        if door_state == "open":                        # rolled up above the opening
            body = _box("roll", mx, my, h + 0.25, *_dims(c, 0.4, opening), 0.45, ROLLER)
            body += _box("roll2", mx, my, h + 0.25, *_dims(c, 0.44, opening + 0.02), 0.47, "#3d4a55", transparency=0.6)
            return [frame, _model(f"door_{c.seg.id}", body)]
        body = _collision("c", mx, my, h / 2, *_dims(c, leaf_t, opening), h)
        body += _box("panel", mx, my, h / 2, *_dims(c, leaf_t, opening), h, ROLLER)
        for k in range(1, int(h / 0.4)):
            body += _box(f"slat{k}", mx, my, k * 0.4, *_dims(c, leaf_t + 0.02, opening + 0.02), 0.03, "#4d5c69")
        return [frame, _model(f"door_{c.seg.id}", body)]
    leaf_w = opening / 2
    L = c.length
    (ax0, ay0), (ax1, ay1) = c.along(0.0), c.along(1.0)
    fwd = ((ax1 - ax0) / L, (ay1 - ay0) / L)            # along the corridor, towards the far zone
    leaves = []
    for s in (-1, 1):
        hinge_off = s * (w / 2 - frame_t)               # across offset of the hinge
        phi = float((swing or (0.0, 0.0))[0 if s < 0 else 1])
        if door_state == "open":                        # swung 90 deg, flat against the frame post
            lx, ly = c.along(0.5 + leaf_w / 2 / L, s * (w / 2 - frame_t - leaf_t / 2))
            leaves.append(_box(f"leaf{s + 1}", lx, ly, h / 2, *_dims(c, leaf_w, leaf_t), h, FIRE_DOOR))
        elif phi:                                       # part-way through its swing
            hx, hy = c.along(0.5, hinge_off)
            ix, iy = c.along(0.5, hinge_off - s)        # one metre in from the hinge
            inw = (ix - hx, iy - hy)
            d = (inw[0] * math.cos(phi) + fwd[0] * math.sin(phi), inw[1] * math.cos(phi) + fwd[1] * math.sin(phi))
            yaw = math.atan2(d[1], d[0])
            lx, ly = hx + d[0] * leaf_w / 2, hy + d[1] * leaf_w / 2
            leaves.append(_collision(f"c{s + 1}", lx, ly, h / 2, leaf_w, leaf_t, h, yaw=yaw))
            leaves.append(_box(f"leaf{s + 1}", lx, ly, h / 2, leaf_w, leaf_t, h, FIRE_DOOR, yaw=yaw))
            leaves.append(_box(f"bar{s + 1}", lx, ly, 1.0, leaf_w * 0.6, leaf_t + 0.06, 0.05, "#c0c0c0", yaw=yaw))
        else:
            lx, ly = c.along(0.5, hinge_off - s * leaf_w / 2)
            leaves.append(_collision(f"c{s + 1}", lx, ly, h / 2, *_dims(c, leaf_t, leaf_w), h))
            leaves.append(_box(f"leaf{s + 1}", lx, ly, h / 2, *_dims(c, leaf_t, leaf_w), h, FIRE_DOOR))
            leaves.append(_box(f"bar{s + 1}", lx, ly, 1.0, *_dims(c, leaf_t + 0.06, leaf_w * 0.6), 0.05, "#c0c0c0"))
    return [frame, _model(f"door_{c.seg.id}", "".join(leaves))]


def underpass_models(c: Corridor) -> List[str]:
    """The mezzanine slab over the passage, its underside at ``height``."""
    h = float(c.seg.props.get("height", 2.2))
    mx, my = c.along(0.5)
    L, w = c.length, c.width
    ext = 1.2                                          # the slab overhangs the racking either side
    size = (L, w + 2 * ext, 0.3) if c.axis == "x" else (w + 2 * ext, L, 0.3)
    body = _collision("c", mx, my, h + 0.15, *size) + _box("slab", mx, my, h + 0.15, *size, SLAB)
    # a railing on top and columns at the corners
    rail_h = 1.1
    for s in (-1, 1):
        rx, ry = c.along(0.5, s * (w / 2 + ext - 0.05))
        rs = (L, 0.05, rail_h) if c.axis == "x" else (0.05, L, rail_h)
        body += _box(f"rail{s + 1}", rx, ry, h + 0.3 + rail_h / 2, *rs, "#f2c319", transparency=0.3)
    for t in (0.02, 0.98):
        for s in (-1, 1):
            px, py = c.along(t, s * (w / 2 - 0.12))
            body += _box(f"col{t}{s}", px, py, h / 2, 0.2, 0.2, h, "#5c5f66")
            body += _collision(f"cc{t}{s}", px, py, h / 2, 0.2, 0.2, h)
    # something stored on the mezzanine
    body += _box("stock", mx, my, h + 0.3 + 0.5, size[0] * 0.6, size[1] * 0.5, 1.0, BOX)
    return [_model(f"mezzanine_{c.seg.id}", body)]


def cordon_models(c: Corridor, kind: str) -> List[str]:
    """Barriers across both ends of a cordoned passage: 1 m high plastic
    barrier panels every lidar sees, coloured by why the passage is closed."""
    a, b = {"hazard": (HAZARD_A, HAZARD_B), "restricted": (RESTRICT_A, RESTRICT_B)}.get(kind, (CLOSED_A, CLOSED_B))
    w = c.width
    out = []
    for end, t in (("a", 0.03), ("b", 0.97)):
        mx, my = c.along(t)
        body = _collision("c", mx, my, 0.5, *_dims(c, 0.12, w), 1.0)
        n = max(2, int(w / 0.6))
        for k in range(n):
            off = -w / 2 + (k + 0.5) * w / n
            px, py = c.along(t, off)
            body += _box(f"p{k}", px, py, 0.5, *_dims(c, 0.12, w / n - 0.04), 0.9, a if k % 2 == 0 else b)
            body += _box(f"f{k}", px, py, 0.05, *_dims(c, 0.37, w / n - 0.04), 0.1, "#3a3a3a")
        if kind == "hazard":                            # cones behind the barrier
            for s in (-1, 1):
                px, py = c.along(t + (0.06 if end == "a" else -0.06), s * w / 4)
                body += _box(f"cone{s + 1}", px, py, 0.3, 0.3, 0.3, 0.6, "#ff6f1a")
        out.append(_model(f"cordon_{c.seg.id}_{end}", body))
    return out


def staged_pallets(c: Corridor, blocked: float, rng: random.Random) -> List[str]:
    """Pallets stacked with boxes along one side of the aisle, ``blocked`` deep,
    for the length of the aisle: the lane that is left is width - blocked."""
    L, w = c.length, c.width
    depth = min(blocked, w - 0.3)
    side = -1
    parts = _collision("c", *c.along(0.5, side * (w / 2 - depth / 2)), 0.7,
                       *((L - 0.6, depth, 1.4) if c.axis == "x" else (depth, L - 0.6, 1.4)))
    n = max(1, int((L - 0.6) / 1.3))
    for k in range(n):
        t = (k + 0.5) / n
        px, py = c.along(t, side * (w / 2 - depth / 2))
        pl = (1.2, depth - 0.05) if c.axis == "x" else (depth - 0.05, 1.2)
        parts += _box(f"pal{k}", px, py, 0.075, pl[0], pl[1], 0.15, PALLET)
        bh = rng.uniform(0.7, 1.25)
        parts += _box(f"box{k}", px, py, 0.15 + bh / 2, pl[0] * 0.9, pl[1] * 0.9, bh,
                      rng.choice((BOX, "#b58c5a", "#8d8d8d", "#5a6b7a")))
        if rng.random() < 0.5:
            parts += _box(f"top{k}", px, py, 0.15 + bh + 0.2, pl[0] * 0.5, pl[1] * 0.5, 0.4, "#d4b483")
    return [_model(f"staged_{c.seg.id}", parts)]


def _free_interval(c: Corridor, blocked: float) -> Tuple[float, float]:
    """The across-corridor interval (offsets from the centreline) not taken by
    staged goods: the pallets stand along the -side, so the lane is beside them."""
    return (-c.width / 2 + blocked, c.width / 2)


def clutter_models(c: Corridor, density: float, rng: random.Random, lane: float,
                   blocked: float = 0.0) -> List[str]:
    """Boxes and carts against the sides of an aisle, leaving a lane of ``lane``
    metres free in the middle of what the staged goods leave."""
    L = c.length
    lo, hi = _free_interval(c, blocked)
    room = (hi - lo - lane) / 2                       # per side, beyond the lane
    n = int(round(density * L / 2.0))
    out = []
    for k in range(n):
        kind = rng.choice(("box", "box", "cart", "bin"))
        dx = rng.uniform(0.4, 0.8) if kind == "box" else 0.6
        dy = rng.uniform(0.4, 0.7) if kind == "box" else 0.9
        if room < dy / 2 + 0.05:
            break                                     # nothing fits beside the lane
        side = rng.choice((-1, 1))
        off = (lo + dy / 2 + 0.05) if side < 0 else (hi - dy / 2 - 0.05)
        px, py = c.along(rng.uniform(0.12, 0.88), off)
        look = Look(f"clutter_{c.seg.id}_{k}", kind, px, py, dx, dy, c.yaw + rng.uniform(-0.3, 0.3),
                    height=rng.uniform(0.5, 1.1) if kind == "box" else 0.9, model=kind)
        out.append(S.object_model(look))
    return out


def people_models(c: Corridor, density: float, rng: random.Random, lane: float,
                  blocked: float = 0.0) -> List[str]:
    """People along the sides of a crowded aisle, out of the driving lane."""
    L = c.length
    lo, hi = _free_interval(c, blocked)
    room = (hi - lo - lane) / 2
    n = int(round(density * L / 1.5))
    out = []
    if room < 0.35:
        return out
    for k in range(n):
        side = rng.choice((-1, 1))
        off = (lo + 0.3) if side < 0 else (hi - 0.3)
        px, py = c.along(rng.uniform(0.1, 0.9), off + rng.uniform(-0.04, 0.04))
        look = Look(f"person_{c.seg.id}_{k}", "person", px, py, 0.55, 0.55, rng.uniform(0, 2 * math.pi),
                    height=rng.uniform(1.65, 1.85), is_human=True, model=rng.choice(PEOPLE))
        out.append(S.object_model(look))
    return out


def marking_models(c: Corridor) -> List[str]:
    """Floor markings: green edge lines on an emergency route, a wet-floor
    decal and sign on a wet surface, gravel on a rough outdoor passage."""
    out = []
    L, w = c.length, c.width
    mx, my = c.along(0.5)
    if c.seg.props.get("emergency_route"):
        for s in (-1, 1):
            px, py = c.along(0.5, s * (w / 2 - 0.15))
            size = (L, 0.12, 0.004) if c.axis == "x" else (0.12, L, 0.004)
            out.append(_model(f"emerg_{c.seg.id}_{s + 1}", _box("line", px, py, 0.004, *size, EMERGENCY, emissive=0.2)))
    if c.seg.props.get("surface_type") == "wet":
        size = (L - 0.4, w - 0.3, 0.006) if c.axis == "x" else (w - 0.3, L - 0.4, 0.006)
        body = _box("wet", mx, my, 0.005, *size, WET, transparency=0.35)
        sx, sy = c.along(0.12, -w / 2 + 0.45)
        body += _box("sign", sx, sy, 0.45, 0.35, 0.05, 0.9, "#f2c319", pitch=0.25)
        body += _box("sign2", sx, sy, 0.45, 0.35, 0.05, 0.9, "#f2c319", pitch=-0.25)
        out.append(_model(f"wet_{c.seg.id}", body))
    return out


def ramp_models(c: Corridor, spec: PlantSpec) -> List[str]:
    """The inclined slab between the dock level and the yard, kerbs either side."""
    z_lo = min(spec.zone(c.seg.a).level, spec.zone(c.seg.b).level)
    slope = math.radians(float(c.seg.props.get("slope_angle", 5.0)))
    L, w = c.length, c.width
    drop = -z_lo
    run = L
    length = math.hypot(run, drop)
    mx, my = c.along(0.5)
    cz = z_lo / 2 - 0.1
    # the ramp descends away from the hall: along -y for a southern yard
    pitch = -slope if c.axis == "y" else 0.0
    yaw = 0.0 if c.axis == "x" else math.pi / 2
    size = (length, w, 0.2)
    body = (f'<collision name="c"><pose>{mx:.4f} {my:.4f} {cz:.4f} 0 {pitch:.5f} {yaw:.5f}</pose>'
            f"<geometry><box><size>{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}</size></box></geometry>"
            f"<surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>")
    body += S.box_visual("ramp", f"{mx:.4f} {my:.4f} {cz:.4f} 0 {pitch:.5f} {yaw:.5f}", size, S.material("#8a8580"))
    for s in (-1, 1):
        kx, ky = c.along(0.5, s * (w / 2 + 0.1))
        body += S.box_visual(f"kerb{s + 1}", f"{kx:.4f} {ky:.4f} {cz + 0.2:.4f} 0 {pitch:.5f} {yaw:.5f}",
                             (length, 0.2, 0.25), S.material("#f2c319"))
        body += (f'<collision name="k{s + 1}"><pose>{kx:.4f} {ky:.4f} {cz + 0.2:.4f} 0 {pitch:.5f} {yaw:.5f}</pose>'
                 f"<geometry><box><size>{length:.4f} 0.2 0.25</size></box></geometry></collision>")
    return [_model(f"ramp_{c.seg.id}", body)]


def light_models(spec: PlantSpec, cors: Dict[str, Corridor]) -> str:
    """A light over every zone and every lit passage; none over a dark one."""
    out = []

    def lamp(name, x, y, z=6.0, strength=0.9, rng_=None):
        out.append(f'<light type="point" name="{name}"><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>'
                   f"<diffuse>{0.55 * strength:.2f} {0.55 * strength:.2f} {0.52 * strength:.2f} 1</diffuse><specular>0.05 0.05 0.05 1</specular>"
                   "<attenuation><range>18</range><constant>0.5</constant><linear>0.05</linear>"
                   "<quadratic>0.004</quadratic></attenuation><cast_shadows>false</cast_shadows></light>")

    for z in spec.zones:
        lamp(f"l_{z.id}", z.x, z.y, 6.0 + z.level)
    for c in cors.values():
        illum = float(c.seg.props.get("illumination", 0.85))
        if illum <= 0.25:
            continue
        n = max(1, int(c.length / 8))
        for k in range(n):
            x, y = c.along((k + 0.5) / n)
            lamp(f"l_{c.seg.id}_{k}", x, y, 6.0, 0.5 + 0.5 * illum)
    return "".join(out)


# ------------------------------------------------------------ zones' decor

def zone_decor(z: Zone, spec: PlantSpec, cors: Dict[str, Corridor], rng: random.Random) -> List[str]:
    """Light furnishing of a zone along the sides no passage enters, so the
    centre lane and every passage mouth stay free."""
    used = set()
    for c in cors.values():
        for zz, other in ((c.za, c.zb), (c.zb, c.za)):
            if zz.id == z.id:
                used.add("E" if other.x > zz.x else "W" if other.x < zz.x else "N" if other.y > zz.y else "S")
    out = []
    H = ZONE_HALF
    k = 0
    for side in ("N", "S", "W", "E"):
        if side in used:
            continue
        for u in (-1.6, 0.0, 1.6):
            if z.kind == "yard":
                break
            if side == "N":
                cx, cy, yaw = z.x + u, z.y + H - 0.55, 0.0
            elif side == "S":
                cx, cy, yaw = z.x + u, z.y - H + 0.55, 0.0
            elif side == "W":
                cx, cy, yaw = z.x - H + 0.55, z.y + u, math.pi / 2
            else:
                cx, cy, yaw = z.x + H - 0.55, z.y + u, math.pi / 2
            kind = {"dock": "box", "shipping": "box", "store": "box", "cell": "table", "test": "desk",
                    "triage": "table", "sorting": "table", "wash": "bin", "quarantine": "box"}.get(z.kind, "cabinet")
            look = Look(f"decor_{z.id}_{k}", kind, cx, cy, 1.2 if kind != "bin" else 0.5, 0.8 if kind != "bin" else 0.5,
                        yaw, height=0.9 if kind in ("table", "desk") else rng.uniform(0.6, 1.3), model=kind)
            # furnishing is scenery: no collision hull, so neither the lidar
            # nor a long robot turning in the zone is ever stopped by it
            out.append(re.sub(r"<collision.*?</collision>", "", S.object_model(look), flags=re.S))
            k += 1
    if z.kind == "yard":
        for k2 in range(4):
            look = Look(f"yard_box_{k2}", "box", z.x - 5 + 3 * k2, z.y - 5, 1.1, 1.0, rng.uniform(0, 1), height=1.0, model="box")
            xml = S.object_model(look).replace("<pose>", f"<pose>", 1)
            out.append(xml.replace(f"{look.cx:.4f} {look.cy:.4f} 0 0 0", f"{look.cx:.4f} {look.cy:.4f} {z.level:.4f} 0 0"))
    return out


# ------------------------------------------------------------------ items

MODULE = (0.30, 0.20, 0.10)          # a battery module: footprint and height (m)
TOTE = (0.40, 0.30, 0.25)
HANDLE = (0.12, 0.03, 0.08)          # a carry handle on top of a module or a tote, gripped across its 3 cm
ITEM_INSET = 0.20                    # a workpiece waits this far in from the front edge of what holds it
FRONT_GRASP_HEIGHT = 0.05            # a front grasp holds the handle this far above the body (the palm clears it)
RACK_LEVELS = (0.0, 1.5, 3.0)        # beam levels of a rack bay; a pallet on each


def _stand(it: Item) -> str:
    if it.stand:
        return it.stand
    return "pack" if it.kind == "pack" else ("bench" if it.height <= 1.2 else "rack")


def _stand_frame(spec: PlantSpec, it: Item) -> Tuple[float, float, float]:
    z = spec.zone(it.zone)
    return z.x + it.offset[0], z.y + it.offset[1], math.atan2(it.offset[1], it.offset[0])   # it faces the zone centre


def stand_boxes(spec: PlantSpec, it: Item) -> List[Tuple[str, float, float, float, float, float, float, float]]:
    """The collision boxes of what holds ``it``, as (name, x, y, z, dx, dy, dz,
    yaw) with centres in the world: for the physics and for MoveIt's scene.  A
    rack bay is its uprights, beam levels and pallets, so that what stands on
    a level rests on it."""
    cx, cy, yaw = _stand_frame(spec, it)
    h = it.height
    kind = _stand(it)
    if kind == "pack":
        return [("cp", cx, cy, 0.075, 1.2, 0.8, 0.15, yaw),
                ("ck", cx, cy, 0.15 + (h - 0.15) / 2, 1.0, 0.7, h - 0.15, yaw)]
    if kind == "bench":
        return [("ct", cx, cy, h / 2, 1.2, 0.7, h, yaw)]
    boxes = []
    for sx in (-0.65, 0.65):
        for sy in (-0.5, 0.5):
            lx = cx + sx * math.cos(yaw) - sy * math.sin(yaw)
            ly = cy + sx * math.sin(yaw) + sy * math.cos(yaw)
            boxes.append((f"cu{sx}{sy}", lx, ly, RACK_H / 2, 0.1, 0.1, RACK_H, 0.0))
    for k, lz in enumerate(RACK_LEVELS[1:] + (RACK_H,)):
        boxes.append((f"cb{k}", cx, cy, lz - 0.06, 1.4, 1.1, 0.12, yaw))
    for k, lz in enumerate(RACK_LEVELS):
        boxes.append((f"cq{k}", cx, cy, lz + 0.075, 1.2, 0.8, 0.15, yaw))
    return boxes


def item_pose(spec: PlantSpec, it: Item) -> Tuple[float, float, float, float, float]:
    """Where a workpiece rests: x, y, the height of its underside, yaw, and how
    far the front of its stand (the side towards the zone centre, where a
    robot comes from) is from its centre."""
    cx, cy, yaw = _stand_frame(spec, it)
    support, outer = {"pack": (0.5, 0.6), "bench": (0.6, 0.6), "rack": (0.6, 0.7)}[_stand(it)]
    back = support - ITEM_INSET
    return cx - back * math.cos(yaw), cy - back * math.sin(yaw), it.height, yaw, outer - back


def front_grasp(height: float) -> np.ndarray:
    """Where TIAGo's grasping frame is, in an item's frame, when it holds the handle from the front:
    at the handle's near end, level, ``FRONT_GRASP_HEIGHT`` above a body ``height`` tall."""
    T = np.eye(4)
    T[:3, 3] = (-HANDLE[0] / 2, 0.0, height / 2 + FRONT_GRASP_HEIGHT)
    return T


def item_anchor(height: float) -> np.ndarray:
    """Where TIAGo's wrist (``arm_7_link``, into which Gazebo lumps the gripper) is, in an item's
    frame, when it holds the handle from the front.  An item's link has its frame there: the
    model attachment plugin makes a part fast by moving it until its link's frame meets the
    wrist's, which then moves it by no more than the grasp's error."""
    from .tiago_fk import fk
    return front_grasp(height) @ np.linalg.inv(fk("gripper_grasping_frame", {})) @ fk("arm_7_link", {})


def _pose_text(T: np.ndarray) -> str:
    r, p_, y = S.matrix_rpy(T[:3, :3])
    return f"{T[0, 3]:.5f} {T[1, 3]:.5f} {T[2, 3]:.5f} {r:.6f} {p_:.6f} {y:.6f}"


def item_models(spec: PlantSpec) -> Tuple[List[str], List[Tuple[Rect, float]]]:
    """Every item on something that holds it at ``height``: a bench for
    bench height, a rack bay for rack heights, the floor pallet for a pack.
    Returns the models and the static footprints (for the map)."""
    models, static = [], []
    for it in spec.items:
        cx, cy, yaw = _stand_frame(spec, it)
        h = it.height
        kind = _stand(it)
        if kind == "pack":
            body = _box("pallet", cx, cy, 0.075, 1.2, 0.8, 0.15, PALLET, yaw=yaw)
            body += _box("pack", cx, cy, 0.15 + (h - 0.15) / 2, 1.0, 0.7, h - 0.15, "#3a3f46", yaw=yaw)
            static.append((Rect(cx - 0.6, cy - 0.6, cx + 0.6, cy + 0.6), h))
        elif kind == "bench":
            body = _box("top", cx, cy, h - 0.02, 1.2, 0.7, 0.04, "#c9b697", yaw=yaw)
            for sx in (-0.55, 0.55):
                for sy in (-0.3, 0.3):
                    lx = cx + sx * math.cos(yaw) - sy * math.sin(yaw)
                    ly = cy + sx * math.sin(yaw) + sy * math.cos(yaw)
                    body += _box(f"leg{sx}{sy}", lx, ly, (h - 0.04) / 2, 0.05, 0.05, h - 0.04, "#4a4b4d")
            static.append((Rect(cx - 0.6, cy - 0.6, cx + 0.6, cy + 0.6), h))
        else:                                                   # a rack bay with beam levels
            body = ""
            for sx in (-0.65, 0.65):
                for sy in (-0.5, 0.5):
                    lx = cx + sx * math.cos(yaw) - sy * math.sin(yaw)
                    ly = cy + sx * math.sin(yaw) + sy * math.cos(yaw)
                    body += _box(f"up{sx}{sy}", lx, ly, RACK_H / 2, 0.1, 0.1, RACK_H, UPRIGHT)
            for k, lz in enumerate(RACK_LEVELS[1:] + (RACK_H,)):
                body += _box(f"beam{k}", cx, cy, lz - 0.06, 1.4, 1.1, 0.12, BEAM, yaw=yaw)
            for k, lz in enumerate(RACK_LEVELS):
                body += _box(f"pal{k}", cx, cy, lz + 0.075, 1.2, 0.8, 0.15, PALLET, yaw=yaw)
            static.append((Rect(cx - 0.75, cy - 0.75, cx + 0.75, cy + 0.75), RACK_H))
        body += "".join(_collision(n, x, y, z, dx, dy, dz, yaw=yw) for n, x, y, z, dx, dy, dz, yw in stand_boxes(spec, it))
        stand = _model(f"stand_{it.id}", body)
        # the workpiece itself, a free body the arm can pick by its handle; its link's frame is
        # where the wrist is when it holds the handle from the front (item_anchor), the body and
        # handle placed back from there
        ix, iy, ih, iyaw, _ = item_pose(spec, it)
        w, d, mh = TOTE if it.kind == "tote" else MODULE
        hx, hy, hz = HANDLE
        A = item_anchor(mh)
        back = np.linalg.inv(A)
        body, handle = _pose_text(back), _pose_text(back @ S.pose_matrix(0, 0, mh / 2 + hz / 2))
        item = (f'<model name="item_{it.id}"><pose>{ix:.4f} {iy:.4f} {ih + mh / 2 + 0.002:.4f} 0 0 {iyaw:.4f}</pose>'
                f'<link name="link"><pose>{_pose_text(A)}</pose>'
                f'{_inertial_xml(min(it.mass, 5.0), w, d, mh, body)}'
                f'<collision name="c"><pose>{body}</pose><geometry><box><size>{w} {d} {mh}</size></box></geometry>'
                f'<surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision>'
                f'<collision name="handle"><pose>{handle}</pose>'
                f'<geometry><box><size>{hx} {hy} {hz}</size></box></geometry>'
                f'<surface><friction><ode><mu>1.5</mu><mu2>1.5</mu2></ode></friction>'
                f'<contact><ode><kp>100000</kp><kd>20</kd><max_vel>0.05</max_vel><min_depth>0.001</min_depth></ode></contact>'
                f'</surface></collision>'
                f'{S.box_visual("v", body, (w, d, mh), S.material("#2b4f8f"))}'
                f'{S.box_visual("handle", handle, HANDLE, S.material("#f2c319"))}'
                f"</link></model>")
        models += [stand, item]
    return models, static


def _inertial_xml(mass: float, dx: float, dy: float, dz: float, pose: str = "0 0 0 0 0 0") -> str:
    ixx = mass / 12 * (dy ** 2 + dz ** 2); iyy = mass / 12 * (dx ** 2 + dz ** 2); izz = mass / 12 * (dx ** 2 + dy ** 2)
    return (f"<inertial><pose>{pose}</pose><mass>{mass:.3f}</mass><inertia><ixx>{ixx:.5f}</ixx><iyy>{iyy:.5f}</iyy>"
            f"<izz>{izz:.5f}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>")


# --------------------------------------------------------------- the world

def apply_changes(spec: PlantSpec, changes: Iterable[Tuple[str, str, object]]) -> PlantSpec:
    """A copy of the spec with ``(individual, property, value)`` changes applied:
    the repaired plant.  Passage properties change on the segment; an item's
    ``item_height`` restocks it (the stand is rebuilt at the new height)."""
    over: Dict[str, Dict[str, object]] = {}
    for sid, prop, val in changes:
        over.setdefault(sid, {})[prop] = val
    segs = []
    for s_ in spec.segments:
        props = dict(s_.props)
        props.update(over.get(s_.id, {}))
        segs.append(replace(s_, props=props))
    items = []
    for it in spec.items:
        ch = over.get(it.id, {})
        items.append(replace(it, height=float(ch.get("item_height", it.height)),
                             width=float(ch.get("item_width", it.width)),
                             mass=float(ch.get("item_mass", it.mass))))
    return PlantSpec(spec.name, list(spec.zones), segs, items)


@dataclass
class Built:
    """What one build wrote."""
    world: str                           # the SDF
    static: List[Tuple[Rect, float]]     # collision rectangles in the lidar band (rect, height)
    bounds: Tuple[float, float, float, float]


def build_world(spec: PlantSpec, seed: int = 0, people: bool = True, clutter: bool = True,
                extra_models: Sequence[str] = (), camera_xml: str = "", rate: float = 250.0,
                lane: float = 2.2, door_swing: Optional[Dict[str, Tuple[float, float]]] = None,
                step: float = 0.004, solver_iters: Optional[int] = None, world_plugins: str = "") -> Built:
    """The SDF of the plant in the state the spec describes.

    ``lane`` is the width people and clutter keep free in a crowded or
    cluttered aisle (they cost in the semantic model, they do not block): wide
    enough for the widest robot of the fleet with its clearances.
    ``door_swing`` (segment id -> the two leaves' angles) catches a closed fire
    door part-way open, for stills.  ``step`` is the physics step and ``rate``
    the steps per wall-clock second (real time at ``rate * step = 1``).
    """
    rng = random.Random(seed)
    cors = corridors(spec)
    models: List[str] = []
    static: List[Tuple[Rect, float]] = []
    models += floor_models(spec, cors)
    models.append(outer_walls(spec, cors))
    h = hall_rect(spec)
    static += [(Rect(h.x0 - WALL_T, h.y0 - WALL_T, h.x1 + WALL_T, h.y0), HALL_WALL_H),
               (Rect(h.x0 - WALL_T, h.y1, h.x1 + WALL_T, h.y1 + WALL_T), HALL_WALL_H),
               (Rect(h.x0 - WALL_T, h.y0, h.x0, h.y1), HALL_WALL_H),
               (Rect(h.x1, h.y0, h.x1 + WALL_T, h.y1), HALL_WALL_H)]
    for k, (band, faces) in enumerate(solid_bands(spec, cors)):
        models.append(block_model(f"block{k}", band, faces, rng))
        static.append((band, RACK_H))
    for c in cors.values():
        p = c.seg.props
        kind = c.seg.kind
        if kind in ("roller_door", "fire_door", "door"):
            state = str(p.get("door_state", "open"))
            models += door_models(c, state, (door_swing or {}).get(c.seg.id))
            if state != "open":
                static.append((_across_rect(c, 0.5, 0.12), float(p.get("height", 3.0))))
        if kind == "underpass" or (kind == "aisle" and float(p.get("height", 4.5)) < 2.6):
            models += underpass_models(c)
        if kind == "ramp":
            models += ramp_models(c, spec)
        cordon = ("hazard" if p.get("is_hazardous") else "restricted" if p.get("restricted")
                  else "closed" if not p.get("is_accessible", True) else None)
        if cordon:
            models += cordon_models(c, cordon)
            for t in (0.03, 0.97):
                static.append((_across_rect(c, t, 0.12), 1.0))
        blocked = float(p.get("blocked_width", 0.0))
        if blocked > 0.05:
            models += staged_pallets(c, blocked, rng)
            depth = min(blocked, c.width - 0.3)
            L = c.length
            sx, sy = c.along(0.5, -(c.width / 2 - depth / 2))
            static.append((Rect(sx - (L - 0.6) / 2 if c.axis == "x" else sx - depth / 2,
                                sy - depth / 2 if c.axis == "x" else sy - (L - 0.6) / 2,
                                sx + (L - 0.6) / 2 if c.axis == "x" else sx + depth / 2,
                                sy + depth / 2 if c.axis == "x" else sy + (L - 0.6) / 2), 1.4))
        if clutter and float(p.get("obstacle_density", 0.0)) > 0.05:
            models += clutter_models(c, float(p["obstacle_density"]), rng, lane=lane, blocked=blocked)
        if people and float(p.get("crowd_density", 0.0)) > 0.05:
            models += people_models(c, float(p["crowd_density"]), rng, lane=lane, blocked=blocked)
        models += marking_models(c)
    for z in spec.zones:
        models += zone_decor(z, spec, cors, rng)
    item_xml, item_static = item_models(spec)
    models += item_xml
    static += item_static
    models += list(extra_models)
    if camera_xml:
        models.append(camera_xml)
    lights = light_models(spec, cors)
    world = _world_xml(spec.name, models, lights, rate, step, solver_iters, world_plugins)
    return Built(world, static, spec.bounds())


def _world_xml(name: str, models: List[str], lights: str, rate: float, step: float = 0.004,
               solver_iters: Optional[int] = None, world_plugins: str = "") -> str:
    solver = (f"<ode><solver><type>quick</type><iters>{solver_iters}</iters><sor>1.3</sor></solver></ode>"
              if solver_iters else "")
    return f"""<?xml version="1.0"?>
<!-- generated by pace.twin.plant_world -->
<sdf version="1.6">
  <world name="{name}">
    <scene><ambient>0.32 0.32 0.34 1</ambient><background>0.82 0.85 0.88 1</background>
      <shadows>false</shadows><grid>false</grid><origin_visual>false</origin_visual></scene>
    <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 30 0 0 0</pose>
      <diffuse>0.35 0.35 0.35 1</diffuse><specular>0.05 0.05 0.05 1</specular><direction>-0.3 0.2 -0.9</direction></light>
    {lights}
    <physics type="ode">
      <max_step_size>{step}</max_step_size>
      <real_time_factor>{rate * step:.3f}</real_time_factor>
      <real_time_update_rate>{rate:.0f}</real_time_update_rate>
      {solver}
    </physics>
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so"><ros><namespace>/gazebo</namespace></ros>
      <update_rate>1.0</update_rate></plugin>
    {world_plugins}
{''.join(models)}
  </world>
</sdf>
"""


# ------------------------------------------------------------- Nav2 map

def rasterize(static: Sequence[Tuple[Rect, float]], bounds: Tuple[float, float, float, float],
              resolution: float = 0.05, min_height: float = 0.15) -> np.ndarray:
    """Occupancy (True = occupied) of the static geometry taller than the lidar band."""
    x0, y0, x1, y1 = bounds
    W, H = int(math.ceil((x1 - x0) / resolution)), int(math.ceil((y1 - y0) / resolution))
    occ = np.zeros((H, W), dtype=bool)
    for r, h in static:
        if h < min_height:
            continue
        i0, i1 = int((r.y0 - y0) / resolution), int(math.ceil((r.y1 - y0) / resolution))
        j0, j1 = int((r.x0 - x0) / resolution), int(math.ceil((r.x1 - x0) / resolution))
        occ[max(0, i0):min(H, i1), max(0, j0):min(W, j1)] = True
    return occ


def write_map(built: Built, out: Path, resolution: float = 0.05) -> Tuple[Path, Path]:
    """map.pgm / map.yaml of the robot's prior map (the static world)."""
    import yaml
    occ = rasterize(built.static, built.bounds, resolution)
    img = np.where(occ, 0, 254).astype(np.uint8)[::-1]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "map.pgm", "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())
    x0, y0, _, _ = built.bounds
    (out / "map.yaml").write_text(yaml.safe_dump({
        "image": "map.pgm", "resolution": resolution, "origin": [float(x0), float(y0), 0.0],
        "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196, "mode": "trinary"}))
    return out / "map.pgm", out / "map.yaml"


def write_world(spec: PlantSpec, out: Path, **kw) -> Built:
    """world.sdf, map.pgm/.yaml and places.yaml for a spec, into ``out``."""
    import yaml
    out.mkdir(parents=True, exist_ok=True)
    built = build_world(spec, **kw)
    (out / "world.sdf").write_text(built.world)
    write_map(built, out)
    (out / "places.yaml").write_text(yaml.safe_dump(
        {z.id: {"x": z.x, "y": z.y, "z": z.level, "label": z.label} for z in spec.zones}))
    return built


def plant_camera(spec: PlantSpec, width: int = 1920, height: int = 1080, yaw: float = math.radians(58),
                 pitch: float = math.radians(38)) -> S.Camera:
    """A camera looking over the whole plant from the south-west."""
    return S.fit_camera(spec.bounds(margin=ZONE_HALF), width, height, yaw, pitch=pitch, top_m=RACK_H, margin=0.02)


if __name__ == "__main__":
    import argparse
    from scenarios.plant import plant_spec, random_plant_spec
    ap = argparse.ArgumentParser(description="Write the plant's Gazebo world, Nav2 map and places.")
    ap.add_argument("--out", default="build/plant")
    ap.add_argument("--seed", type=int, default=None, help="a generated plant instead of the authored one")
    ap.add_argument("--no-people", action="store_true")
    a = ap.parse_args()
    spec = plant_spec() if a.seed is None else random_plant_spec(a.seed)
    b = write_world(spec, Path(a.out), people=not a.no_people)
    print(f"wrote {a.out}/world.sdf ({len(b.world) / 1e3:.0f} kB), map and places")
