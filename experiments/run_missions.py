"""The pick missions in the twin: TIAGo drives to an item and picks it, before and after a restock.

    PYTHONPATH=. python experiments/run_missions.py --procs 2 --output results/twin_missions.csv

For every item of the plant, the mobile manipulator's mission from a start
zone is driven with Nav2 + MoveIt (:mod:`pace.twin.mission`).  Where PACE's
repair set for the mission touches the item (a restock), the mission is
driven again with the restock applied; where it touches passages, those
repairs are applied in both phases so that the manipulation step is what is
tested.  Payload and grip are ratings a position-controlled arm in Gazebo
cannot adjudicate, so items whose only repair is mass or width are driven
once, for the record.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace import AStarPlanner  # noqa: E402
from pace.affordance import AffordanceReasoner  # noqa: E402
from pace.counterfactual import CounterfactualEngine  # noqa: E402
from pace.evaluation import questioned_routes  # noqa: E402
from scenarios.plant import plant_spec, build_world_from_spec  # noqa: E402

START = "dock"


def plan_missions(k: int = 3) -> list:
    spec = plant_spec()
    onto, graph, agents = build_world_from_spec(spec)
    reasoner = AffordanceReasoner()
    planner = AStarPlanner(graph, reasoner)
    engine = CounterfactualEngine(reasoner, planner, onto.individuals)
    agent = agents["mobile_manipulator"]
    jobs = []
    for it in spec.items:
        goal = it.node
        actual = planner.find_path(START, goal, agent)
        routes = questioned_routes(graph, START, goal, k)
        if actual.is_feasible:
            seq = actual.nodes
            jobs.append({"item": it.id, "route": seq[:-1], "phase": "before", "changes": [], "kind": "feasible"})
            continue
        # the questioned route with the fewest repairs
        best = None
        for seq in routes:
            cf = engine.explain_why_not(actual, seq, agent)
            reps = [c for c in cf.changes if c.claim == "repair"]
            if reps and (best is None or len(reps) < len(best[1])):
                best = (seq, reps)
        if best is None:
            jobs.append({"item": it.id, "route": routes[0][:-1] if routes else [START, it.zone], "phase": "before",
                         "changes": [], "kind": "unrepairable"})
            continue
        seq, reps = best
        nav_changes = [(c.individual_name, c.property_name, c.counterfactual_value) for c in reps
                       if c.individual_name != it.id]
        item_changes = [(c.individual_name, c.property_name, c.counterfactual_value) for c in reps
                        if c.individual_name == it.id]
        kind = "restock" if any(p == "item_height" for _, p, _ in item_changes) else "rated"
        jobs.append({"item": it.id, "route": seq[:-1], "phase": "before", "changes": nav_changes, "kind": kind})
        if kind == "restock":
            jobs.append({"item": it.id, "route": seq[:-1], "phase": "after", "changes": nav_changes + item_changes,
                         "kind": kind})
    return jobs


def _run_one(args):
    job, out_root, index, timeout, film = args
    from pace.twin import plant_world as PW
    from pace.twin.mission import MissionEpisode
    spec = PW.apply_changes(plant_spec(), [tuple(c) for c in job["changes"]])
    eid = f"{job['item']}__{job['phase']}"
    ep = MissionEpisode(spec, list(job["route"]), job["item"], Path(out_root) / eid, index=index,
                        timeout=timeout, tag=eid, film=film)
    res = ep.run()
    if res["outcome"] == "error":                       # the stack, not the mission: one more try
        ep = MissionEpisode(spec, list(job["route"]), job["item"], Path(out_root) / eid, index=index + 50,
                            timeout=timeout, tag=eid, film=film)
        res = ep.run()
    res.update({"episode": eid, "phase": job["phase"], "kind": job["kind"], "item": job["item"],
                "changes": json.dumps(job["changes"])})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/twin_missions.csv")
    ap.add_argument("--out-root", default="results/missions")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--only", nargs="*", default=None, help="episode ids (item__phase) to run, e.g. module_bench__before")
    ap.add_argument("--film", action="store_true", help="save frames of each pick (needs DISPLAY)")
    a = ap.parse_args()
    jobs = plan_missions()
    if a.only:
        jobs = [j for j in jobs if f"{j['item']}__{j['phase']}" in a.only]
    print(f"{len(jobs)} mission episodes")
    for j in jobs:
        print(f"  {j['phase']:6s} {j['item']:16s} {j['kind']:12s} {'>'.join(j['route'])}  {j['changes']}")
    if a.plan_only:
        return
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.only and output.exists():                     # keep the other episodes' rows
        ids = {f"{j['item']}__{j['phase']}" for j in jobs}
        rows = [r for r in pd.read_csv(output).to_dict("records") if r["episode"] not in ids]
    work = [(j, a.out_root, i, a.timeout, a.film) for i, j in enumerate(jobs)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procs, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(_run_one, work):
            rows.append(res)
            pd.DataFrame(rows).to_csv(output, index=False)
            print(f"{res['episode']}: {res['outcome']} ({res['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
