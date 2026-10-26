"""Drive PACE's questioned routes in the Gazebo twin, before and after the repair.

For every task in the twin protocol this script asks PACE what would open each
questioned route, then runs the route in the twin (:mod:`pace.twin.episode`)
on the plant as it is and on the plant with PACE's repair set applied.  A
repair is *verified* when the route is blocked before and driven to the end
after; a route the semantic model calls feasible is *sound* when the robot
drives it.  Rows are written to a CSV as episodes finish, so a stopped run
resumes, and episodes run in parallel, each in its own ROS domain and Gazebo
port.

    PYTHONPATH=. python experiments/run_twin.py --procs 4 --output results/twin_episodes.csv
    PYTHONPATH=. python experiments/run_twin.py --smoke          # a handful of episodes

Which routes
------------
Every connected (robot, start, goal) task of the plant has up to k+2 questioned
routes.  Driving all of them is hours of simulation for little gain, since most
share their blocking segments, so the protocol takes, per robot:

* every questioned route whose repair set (PACE, k=3) is non-empty and whose
  blockers are *within the twin's scope* - doors, cordons, staged goods, width,
  overhead clearance - once per distinct set of repaired segments; routes over
  the ramp are excluded (a rated slope limit is torque and tipping, which a
  kinematic drive does not model), as are routes whose only repair is a
  permission the twin has no barrier for;
* every route the semantic model calls feasible, once per distinct segment set,
  for the soundness check.

Each selected route is driven before and, where there is a repair, after.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace import AStarPlanner  # noqa: E402
from pace.affordance import AffordanceReasoner  # noqa: E402
from pace.counterfactual import CounterfactualEngine  # noqa: E402
from pace.evaluation import questioned_routes  # noqa: E402
from scenarios.plant import ROBOTS, plant_spec, build_world_from_spec  # noqa: E402

#: repairs the twin can adjudicate (a barrier, a door leaf, pallets, racking, a slab)
IN_SCOPE = {"door_state", "is_hazardous", "restricted", "is_accessible", "blocked_width", "width", "height"}
OUT_OF_SCOPE = {"slope_angle", "surface_type", "illumination", "crowd_density", "obstacle_density"}


def plan_episodes(k: int = 3, robots: Optional[List[str]] = None, max_legs: int = 6,
                  max_changes: int = 3) -> List[dict]:
    """The list of (robot, route, changes) episodes the protocol selects."""
    spec = plant_spec()
    onto, graph, agents = build_world_from_spec(spec)
    reasoner = AffordanceReasoner()
    planner = AStarPlanner(graph, reasoner)
    engine = CounterfactualEngine(reasoner, planner, onto.individuals)
    nodes = sorted(graph.nodes)
    episodes: List[dict] = []
    for rname in (robots or sorted(agents)):
        agent = agents[rname]
        seen_repairs: set = set()
        feasible: Dict[frozenset, dict] = {}
        for start in nodes:
            for goal in nodes:
                if start == goal:
                    continue
                actual = planner.find_path(start, goal, agent)
                for seq in questioned_routes(graph, start, goal, k):
                    if "yard" in seq or len(seq) - 1 > max_legs or seq[-1].startswith("pick_"):
                        continue        # the ramp is out of scope; no marathons; missions run with MoveIt
                    alt = planner.evaluate_sequence(seq, agent)
                    segs = tuple(e.space_individual.name for e in alt.edges)
                    if alt.is_feasible:
                        key = frozenset(segs)
                        if key not in feasible or len(seq) < len(feasible[key]["route"]):
                            feasible[key] = {"robot": rname, "route": seq, "kind": "feasible", "changes": [],
                                             "segments": list(segs), "task": f"{start}->{goal}"}
                        continue
                    cf = engine.explain_why_not(actual, seq, agent)
                    repairs = [c for c in cf.changes if c.claim == "repair"]
                    if not repairs:
                        continue
                    props = {c.property_name for c in repairs}
                    if props & OUT_OF_SCOPE or not props <= IN_SCOPE or len(repairs) > max_changes:
                        continue
                    key = frozenset((c.individual_name, c.property_name) for c in repairs)
                    if key in seen_repairs:
                        continue
                    seen_repairs.add(key)
                    episodes.append({"robot": rname, "route": seq, "kind": "repair",
                                     "changes": [(c.individual_name, c.property_name, c.counterfactual_value)
                                                 for c in repairs],
                                     "segments": list(segs), "task": f"{start}->{goal}",
                                     "named": sorted({c.individual_name for c in repairs})})
        # soundness: the shortest feasible routes that together cover every
        # segment this robot can drive (a greedy set cover)
        uncovered = set().union(*(set(v["segments"]) for v in feasible.values())) if feasible else set()
        pool = sorted(feasible.values(), key=lambda e: (len(e["route"]), e["task"]))
        while uncovered and pool:
            best = max(pool, key=lambda e: (len(set(e["segments"]) & uncovered), -len(e["route"])))
            if not set(best["segments"]) & uncovered:
                break
            episodes.append(best)
            uncovered -= set(best["segments"])
            pool.remove(best)
    return episodes


def episode_id(ep: dict, phase: str) -> str:
    return f"{ep['robot']}__{'_'.join(ep['route'])}__{phase}"


def _run_one(args: Tuple[dict, str, str, int, float, float, bool]) -> dict:
    ep, phase, out_root, index, timeout, rate, people = args
    os.environ["ROS_DOMAIN_ID"] = str(1 + index % 29)
    from pace.twin import plant_world as PW
    from pace.twin.episode import Episode
    spec = plant_spec()
    if phase == "after":
        spec = PW.apply_changes(spec, [tuple(c) for c in ep["changes"]])
    eid = episode_id(ep, phase)
    out = Path(out_root) / eid
    e = Episode(spec, ep["robot"], list(ep["route"]), out, index=index, timeout=timeout, rate=rate,
                people=people, tag=eid)
    res = e.run()
    if res["outcome"] == "error":                       # the stack, not the route: one more try
        e = Episode(spec, ep["robot"], list(ep["route"]), out, index=index + 50, timeout=timeout, rate=rate,
                    people=people, tag=eid)
        res = e.run()
    res.update({"episode": eid, "phase": phase, "kind": ep["kind"], "task": ep["task"],
                "changes": json.dumps(ep["changes"]), "named": json.dumps(ep.get("named", [])),
                "route_segments": json.dumps(ep["segments"])})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/twin_episodes.csv")
    ap.add_argument("--out-root", default="results/twin")
    ap.add_argument("--procs", type=int, default=max(1, min(4, (os.cpu_count() or 4) // 4)))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--robots", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="list the episodes and exit")
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--rate", type=float, default=250.0, help="physics updates per wall second (250 = real time)")
    ap.add_argument("--no-people", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--redo-list", default=None, help="file with episode ids to run again")
    a = ap.parse_args()

    episodes = plan_episodes(a.k, a.robots)
    jobs = []
    for ep in episodes:
        jobs.append((ep, "before"))
        if ep["kind"] == "repair":
            jobs.append((ep, "after"))
    if a.smoke:
        picked, seen = [], set()
        for ep, phase in jobs:
            if (ep["robot"], ep["kind"], phase) not in seen and len(picked) < 6:
                seen.add((ep["robot"], ep["kind"], phase))
                picked.append((ep, phase))
        jobs = picked
    print(f"{len(episodes)} routes, {len(jobs)} episodes")
    if a.plan_only:
        for ep, phase in jobs:
            print(f"  {phase:6s} {ep['robot']:20s} {ep['kind']:8s} {'>'.join(ep['route'])}  {ep['changes']}")
        return

    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done: Dict[str, dict] = {}
    redo_ids = set(Path(a.redo_list).read_text().split()) if a.redo_list else set()
    if output.exists() and not a.redo:
        for _, row in pd.read_csv(output).iterrows():
            if row["outcome"] not in ("error", "interrupted") and row["episode"] not in redo_ids:
                done[f"{row['episode']}"] = dict(row)
    todo = [(ep, phase) for ep, phase in jobs if episode_id(ep, phase) not in done]
    print(f"{len(done)} done, {len(todo)} to run, {a.procs} at a time")
    t0 = time.time()
    rows = list(done.values())
    work = [(ep, phase, a.out_root, i, a.timeout, a.rate, not a.no_people) for i, (ep, phase) in enumerate(todo)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procs, maxtasksperchild=1) as pool:
        for k, res in enumerate(pool.imap_unordered(_run_one, work)):
            rows.append(res)
            pd.DataFrame(rows).to_csv(output, index=False)
            print(f"[{k + 1}/{len(work)}] {res['episode']}: {res['outcome']} "
                  f"(legs {res['legs_done']}/{res['legs']}, stopped at {res['stopped_at']}, {res['seconds']}s)"
                  f"  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    print("all done")


if __name__ == "__main__":
    main()
