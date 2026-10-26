"""Forward kinematics over PAL's TIAGo URDF, and a solver for postures that
put the gripper somewhere: how :data:`pace.twin.robots.PUSHING` was found.

    PYTHONPATH=. python -m pace.twin.tiago_fk 0.72 -0.05 1.0
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from .assets import A
from .sdf import pose_matrix

ARM = [f"arm_{i}_joint" for i in range(1, 8)]


@lru_cache(maxsize=1)
def joints() -> Dict[str, dict]:
    """Every joint of the vendored URDF: type, parent, child, origin, axis, limits."""
    root = ET.parse(A.PAL_DIR / "tiago.urdf").getroot()
    out = {}
    for j in root.findall("joint"):
        o, ax, lim = j.find("origin"), j.find("axis"), j.find("limit")
        out[j.get("name")] = dict(
            type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            T=pose_matrix(*[float(t) for t in (o.get("xyz") if o is not None else "0 0 0").split()],
                          *[float(t) for t in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]),
            axis=np.array([float(t) for t in (ax.get("xyz") if ax is not None else "1 0 0").split()]),
            lim=(float(lim.get("lower")), float(lim.get("upper"))) if lim is not None and lim.get("lower") else None)
    return out


def _rot(ax: np.ndarray, q: float) -> np.ndarray:
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K
    return T


def fk(link: str, q: Dict[str, float]) -> np.ndarray:
    """The pose of ``link`` in ``base_footprint`` at joint values ``q``
    (unnamed joints at zero), as a 4x4 matrix."""
    J = joints()
    child2joint = {d["child"]: n for n, d in J.items()}
    chain = []
    while link in child2joint:
        n = child2joint[link]
        chain.append(n)
        link = J[n]["parent"]
    T = np.eye(4)
    for n in reversed(chain):
        d = J[n]
        T = T @ d["T"]
        v = float(q.get(n, 0.0))
        if d["type"] in ("revolute", "continuous") and v:
            T = T @ _rot(d["axis"], v)
        elif d["type"] == "prismatic" and v:
            Tq = np.eye(4)
            Tq[:3, 3] = d["axis"] * v
            T = T @ Tq
    return T


def solve_reach(target: Tuple[float, float, float], base: Dict[str, float], elbow_y: float = -0.30,
                restarts: int = 80, seed: int = 1) -> Dict[str, float]:
    """A torso and arm posture whose grasping frame sits at ``target`` (in
    ``base_footprint``), approaching straight ahead with the fingers side by
    side, the elbow out at ``elbow_y``, every joint 0.12 rad inside its
    limits: Nelder-Mead from random starts (needs scipy)."""
    from scipy.optimize import minimize

    J = joints()
    names = ["torso_lift_joint"] + ARM
    lims = [J[n]["lim"] for n in names]

    def cost(x):
        q = dict(base)
        q.update(zip(names, x))
        Tg = fk("gripper_grasping_frame", q)
        Te = fk("arm_4_link", q)
        c = 40 * float(np.sum((Tg[:3, 3] - target) ** 2))
        c += 3.0 * (1 - float(Tg[0, 0]))                       # approach along +x
        c += 0.5 * abs(float(Tg[2, 1]))                        # fingers side by side, horizontal
        c += 4.0 * (float(Te[1, 3]) - elbow_y) ** 2            # elbow out to the side
        c += 10.0 * max(0.0, float(Te[2, 3]) - 1.15) ** 2      # elbow not above the shoulder
        for v, (lo, hi) in zip(x, lims):
            c += 50 * (max(0.0, lo + 0.12 - v) ** 2 + max(0.0, v - (hi - 0.12)) ** 2)
        return c

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        r = minimize(cost, x0, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-5, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    q = dict(base)
    q.update({n: round(float(v), 3) for n, v in zip(names, best.x)})
    return q


if __name__ == "__main__":
    from .robots import TUCKED
    tgt = tuple(float(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (0.72, -0.05, 1.0)
    q = solve_reach(tgt, TUCKED)
    Tg = fk("gripper_grasping_frame", q)
    print({n: q[n] for n in ["torso_lift_joint"] + ARM})
    print("grasping frame at", np.round(Tg[:3, 3], 3), "approach", np.round(Tg[:3, 0], 2))
