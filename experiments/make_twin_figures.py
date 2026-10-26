"""The paper's qualitative figures: PACE's answer drawn into the Gazebo twin.

An explanation is only as clear as its picture, so the twin renders the plant
with the robots in it, the questioned routes dashed on the floor, and every
passage or item PACE names marked by what it says about it -- green for a
low-effort action, amber for one that needs authorisation, red for
construction -- and matplotlib annotates the still with the recommendation
text projected onto the right spot.

    PYTHONPATH=. DISPLAY=:0 python experiments/make_twin_figures.py --out build/figures

Rendering needs an OpenGL display (the laptop's or the lab PC's :0).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pace import AStarPlanner  # noqa: E402
from pace.affordance import AffordanceReasoner  # noqa: E402
from pace.counterfactual import CounterfactualEngine  # noqa: E402
from pace.evaluation import questioned_routes  # noqa: E402
from pace.twin import plant_world as PW  # noqa: E402
from pace.twin import render as R  # noqa: E402
from pace.twin import robots as RB  # noqa: E402
from pace.twin import sdf as S  # noqa: E402
from scenarios.plant import ROBOTS, ZONE_HALF, build_world_from_spec, plant_spec  # noqa: E402

LOW, MEDIUM, HIGH = "#2ca02c", "#e6a100", "#d62728"
ROUTE_COLOURS = ["#1f5fa8", "#e26d1a", "#7b3fa0", "#2a9d8f"]
EFFORT_COLOUR = {"low": LOW, "medium": MEDIUM, "high": HIGH}
#: the stills as Gazebo rendered them, before annotation
RAW_DIR = Path("build/figs")
#: the stills the paper shows (the pick mission is described in its text)
PAPER_FIGURES = {"twin_overview"}
PRETTY = {"door_state": "open door", "is_hazardous": "clear cordon",
          "restricted": "authorise access", "is_accessible": "reopen aisle",
          "blocked_width": "move staged pallets", "width": "widen",
          "height": "raise the slab", "item_height": "restock lower",
          "item_mass": "too heavy for the arm", "item_width": "too wide for the gripper",
          "slope_angle": "regrade the ramp", "crowd_density": "redirect people",
          "illumination": "lights on", "surface_type": "dry the floor",
          "obstacle_density": "clear the clutter"}
SEG_NAMES = {"seg_dock_recv": "roller door", "seg_ship_sort": "fire door", "seg_disA_disB": "aisle A-B",
             "seg_dock_quar": "dock aisle", "seg_quar_diag": "monitoring zone", "seg_store_disB": "closed aisle",
             "seg_wash_store": "VNA aisle", "seg_store_ship": "underpass", "seg_wash_disA": "wash aisle",
             "seg_disB_sort": "sorting aisle", "seg_dock_yard": "dock ramp", "seg_recv_diag": "main aisle",
             "seg_diag_disA": "aisle", "seg_quar_wash": "aisle", "module_rack": "module (upper rack)",
             "module_bench": "module (bench)", "pack_quarantine": "damaged pack", "tote_sorted": "tote"}


def explain(spec, robot: str, start: str, goal: str, k: int = 3):
    onto, graph, agents = build_world_from_spec(spec)
    reasoner = AffordanceReasoner()
    planner = AStarPlanner(graph, reasoner)
    engine = CounterfactualEngine(reasoner, planner, onto.individuals)
    agent = agents[robot]
    actual = planner.find_path(start, goal, agent)
    out = []
    for seq in questioned_routes(graph, start, goal, k):
        if tuple(seq) == tuple(actual.nodes):
            continue
        out.append((seq, engine.explain_why_not(actual, seq, agent)))
    return actual, out, graph


def route_xy(spec, seq: List[str], graph) -> np.ndarray:
    pts = []
    for n in seq:
        node = graph.nodes[n]
        pts.append(node.position)
    return np.array(pts, dtype=float)


def marker_looks(spec, cors, name: str) -> PW.Look:
    if name in cors:
        c = cors[name]
        return PW.Look(f"m_{name}", "marker", c.rect.cx, c.rect.cy, c.rect.dx, c.rect.dy, 0.0, 0.5)
    for it in spec.items:
        if it.id == name:
            z = spec.zone(it.zone)
            return PW.Look(f"m_{name}", "marker", z.x + it.offset[0], z.y + it.offset[1], 1.2, 1.2, 0.0, 0.5)
    raise KeyError(name)


def layout_callouts(items: List[Tuple[float, float, float]], width: float, top: float, row_h: float,
                    gap: float | None = None, max_rows: int = 6) -> List[Tuple[float, float]]:
    """Places callout labels in rows along the top of a picture, each with a
    straight leader down to its marker.  ``items`` are (u, v, half width) in
    pixels, markers in any order; returns each label's centre.

    A label goes where it lands closest to its marker, a deeper row counting as
    much as sliding 2.5 rows' height sideways, since it covers more of the
    picture: it slides sideways past the labels of its row and past the
    leaders that cross the row, and its own leader may not run through a label
    in a deeper row, so no two labels overlap and no leader crosses a label.
    """
    gap = 0.008 * width if gap is None else gap
    margin = 0.01 * width
    band = 0.3 * row_h                                    # half the height of a label box, with room
    row_cost = 2.5 * row_h
    boxes: List[Tuple[float, float, int]] = []            # (x0, x1, row)
    leaders: List[Tuple[float, float, float, float]] = [] # (x, y, u, v)

    def at(ld, y):                                        # a leader's x at height y, if it gets there
        x, y0, u, v = ld
        if not min(y0, v) <= y <= max(y0, v) or v == y0:
            return None
        return x + (u - x) * (y - y0) / (v - y0)

    spots: List[Tuple[float, float]] = [(0.0, 0.0)] * len(items)
    for k in sorted(range(len(items)), key=lambda n: items[n][0]):
        u, v, half = items[k]
        lo, hi = margin + half, width - margin - half
        best = None
        for i in range(max_rows):
            y = top + i * row_h
            obst = [(a, b) for a, b, r in boxes if r == i]
            for ld in leaders:
                xs = [q for q in (at(ld, y - band), at(ld, y + band)) if q is not None]
                if xs:
                    obst.append((min(xs), max(xs)))
            for step in (1, -1):
                x, moved = min(max(u, lo), hi), True
                while moved and lo <= x <= hi:
                    moved = False
                    for a, b in obst:
                        if x - half < b + gap and x + half > a - gap:
                            x, moved = (b + gap + half if step > 0 else a - gap - half), True
                if not lo <= x <= hi:
                    continue
                crosses = any(y < top + r * row_h < v and a - gap < at((x, y, u, v), top + r * row_h) < b + gap
                              for a, b, r in boxes)
                if crosses:
                    continue
                score = abs(x - u) + row_cost * i
                if best is None or score < best[0]:
                    best = (score, i, x)
            if best is not None and best[0] <= row_cost * (i + 1):
                break                                     # no deeper row can do better
        if best is None:                                  # nowhere clean: the deepest row, as asked
            best = (0.0, max_rows - 1, min(max(u, lo), hi))
        _, i, x = best
        y = top + i * row_h
        boxes.append((x - half, x + half, i))
        leaders.append((x, y, u, v))
        spots[k] = (x, y)
    return spots


def overview(spec, robot: str, start: str, goal: str, out: Path, width: int = 2400, height: int = 1240,
             others: Dict[str, Tuple[str, float]] | None = None, k: int = 3, max_routes: int = 2,
             yaw_deg: float = 58.0, pitch_deg: float = 42.0, title: str | None = None,
             reuse_raw: bool = False) -> dict:
    """Render and annotate one question; returns what was drawn (for the caption)."""
    actual, cfs, graph = explain(spec, robot, start, goal, k)
    cors = PW.corridors(spec)
    routes = cfs[:max_routes]
    extra: List[str] = []
    for r, (seq, cf) in enumerate(routes):
        xy = route_xy(spec, seq, graph)
        extra.append(S.route_model(xy, f"route{r}", ROUTE_COLOURS[r], period=1.8, dash=1.1, width=0.55))
    marks: Dict[str, Tuple[str, str, int]] = {}
    for r, (seq, cf) in enumerate(routes):
        for c in cf.changes:
            if c.claim != "repair":
                continue
            key = c.individual_name
            eff = c.effort()
            if key not in marks or ["low", "medium", "high"].index(eff) > ["low", "medium", "high"].index(marks[key][0]):
                marks[key] = (eff, c.property_name, r)
    for name, (eff, prop, r) in marks.items():
        look = marker_looks(spec, cors, name)
        extra.append(S.halo_model(f"halo_{name}", look, EFFORT_COLOUR[eff], pad=0.25, width=0.22))
        extra.append(S.beacon_model(f"beacon_{name}", look, EFFORT_COLOUR[eff]))
    # robots
    sz = spec.zone(start)
    nxt = graph.nodes[routes[0][0][1]].position if routes else (sz.x + 1, sz.y)
    yaw = math.atan2(nxt[1] - sz.y, nxt[0] - sz.x)
    extra.append(RB.robot_static(robot, ROBOTS[robot], sz.x, sz.y, yaw, sz.level, label=f"r_{robot}"))
    for rname, (zid, ryaw) in (others or {}).items():
        z = spec.zone(zid)
        extra.append(RB.robot_static(rname, ROBOTS[rname], z.x + 0.8, z.y - 0.8, ryaw, z.level, label=f"r_{rname}"))
    h = PW.hall_rect(spec)
    cam = S.fit_camera((h.x0 - 1.0, h.y0 - 4.0, h.x1 + 1.0, h.y1 + 1.0), width, height, math.radians(yaw_deg),
                       pitch=math.radians(pitch_deg), top_m=PW.RACK_H, margin=0.01)
    raw = RAW_DIR / f"{out.stem}.raw.jpg"
    if reuse_raw and raw.exists():
        import imageio.v3 as iio
        img = iio.imread(raw)
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        img = R.render(lambda fdir: PW.build_world(spec, extra_models=extra, camera_xml=cam.sdf(fdir, rate=4.0)).world,
                       out=raw, frames=5, timeout=300)
    # ---- annotate
    fig = plt.figure(figsize=(width / 300, height / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    fs = 6.5
    for zid in set(sum([seq for seq, _ in routes], [])):
        z = spec.zone(zid) if not zid.startswith("pick_") else None
        if z is None:
            continue
        u, v = cam.project([(z.x, z.y, 0.05)])[0]
        ax.text(u, v, z.label.split(" (")[0], fontsize=fs, ha="center", va="center", color="#202022",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    placed = []
    for name, (eff, prop, r) in marks.items():
        look = marker_looks(spec, cors, name)
        u, v = cam.project([(look.cx, look.cy, 1.6)])[0]
        placed.append((u, v, name, eff, prop))
    placed.sort()
    # callouts in rows along the top of the picture, each with a leader down to
    # its marker (see layout_callouts)
    renderer = fig.canvas.get_renderer()
    pad_px = 0.25 * fs * fig.dpi / 72.0                   # the label box's padding
    items = []
    for u, v, name, eff, prop in placed:
        label = f"{SEG_NAMES.get(name, name)}: {PRETTY.get(prop, prop)} ({eff})"
        probe = ax.text(0, 0, label, fontsize=fs, weight="bold")
        half = probe.get_window_extent(renderer).width / 2 + pad_px + 1
        probe.remove()
        items.append((u, v, half, label, eff))
    spots = layout_callouts([(u, v, half) for u, v, half, _l, _e in items], width,
                            top=0.045 * height, row_h=0.062 * height)
    for (u, v, _h, label, eff), (x, y) in zip(items, spots):
        ax.annotate(label, (u, v), xytext=(x, y), textcoords="data", fontsize=fs,
                    ha="center", va="center", color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=EFFORT_COLOUR[eff], ec="none", alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color=EFFORT_COLOUR[eff], lw=1.2, shrinkA=0, shrinkB=0))
    # legend / question
    q = title or f"{robot.replace('_', ' ')}: why can't you reach {goal.replace('_', ' ')} from {start}?"
    ax.text(0.988, 0.03 + 0.055 * len(routes), q, transform=ax.transAxes, fontsize=fs + 1.5, va="bottom", ha="right",
            weight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9))
    for r, (seq, cf) in enumerate(routes):
        reps = [c for c in cf.changes if c.claim == "repair"]
        acts = sum(c.is_actionable() for c in reps)
        txt = (f"route {r + 1}: {' > '.join(n.replace('_', ' ') for n in seq)}\n"
               f"   {len(reps)} repair(s), {acts} actionable"
               + (f"; {len(reps) - acts} structural" if len(reps) > acts else ""))
        ax.text(0.988, 0.03 + 0.055 * (len(routes) - 1 - r), txt, transform=ax.transAxes, fontsize=fs, va="bottom",
                ha="right", color=ROUTE_COLOURS[r],
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return {"robot": robot, "start": start, "goal": goal,
            "routes": [{"seq": seq, "changes": [(c.individual_name, c.property_name, c.claim, c.effort(),
                                                 c.is_actionable()) for c in cf.changes]} for seq, cf in routes]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/figures")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reuse-raw", action="store_true", help="annotate the last render again instead of rendering")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = plant_spec()
    shots = {
        "twin_overview": dict(robot="forklift_agv", start="dock", goal="shipping",
                              others={"tote_amr": ("receiving", 0.0), "mobile_manipulator": ("diagnostics", 3.0),
                                      "yard_ugv": ("yard", 1.2)}),
        "twin_mission": dict(robot="mobile_manipulator", start="dock", goal="pick_module_rack",
                             others={"forklift_agv": ("shipping", 3.1)}),
    }
    log = {}
    for name, kw in shots.items():
        if a.only and name not in a.only:
            continue
        dest = out if name in PAPER_FIGURES else RAW_DIR  # the rest stay out of the paper's folder
        dest.mkdir(parents=True, exist_ok=True)
        log[name] = overview(spec, out=dest / f"{name}.jpg", reuse_raw=a.reuse_raw, **kw)
        print("wrote", dest / f"{name}.jpg")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "twin_figures.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
