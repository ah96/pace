"""The reach sweep in the twin: TIAGo picks one module or tote placed at a range of heights.

    PYTHONPATH=. python experiments/run_reach_sweep.py --procs 2 --output results/reach_sweep.csv

PACE's Reachable rule compares the height of the surface an item rests on
with the arm's rated reach (TIAGo: 1.75 m).  The twin grasps the item by its
handle, which stands above that surface by the body and the handle (0.15 m
for a module, 0.30 m for a tote), so the sweep measures where the rule and
the arm agree.  Each item stands on a bench of its own height in disassembly
bay A, where the bench mission picks, and TIAGo comes from diagnostics; one
more series puts the module on the rack's middle level (1.65 m), the height
PACE restocks the upper-rack module to.  Every episode records PACE's verdict
next to the twin's outcome; a run resumes from the rows already written.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace import AStarPlanner  # noqa: E402
from pace.affordance import AffordanceReasoner  # noqa: E402
from scenarios.plant import Item, plant_spec, build_world_from_spec  # noqa: E402

ZONE, START = "disassembly_A", "diagnostics"
HEIGHTS = (0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.35, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.90, 2.00)
SERIES = [("module", "bench", HEIGHTS), ("tote", "bench", HEIGHTS), ("module", "rack", (1.65,))]


def sweep_spec(kind: str, stand: str, height: float):
    """The authored plant with one item, of ``kind`` on a ``stand`` at ``height``, as its only item."""
    base = plant_spec()
    bench = next(it for it in base.items if it.id == "module_bench")
    item = Item("sweep", ZONE, f"{kind} at {height:.2f} m", height=height, width=0.03,
                mass=2.4 if kind == "module" else 1.5, kind=kind, offset=bench.offset, stand=stand)
    return replace(base, items=[item])


def pace_reachable(spec) -> bool:
    """PACE's verdict: is the mission from START to the item feasible for TIAGo?"""
    onto, graph, agents = build_world_from_spec(spec)
    planner = AStarPlanner(graph, AffordanceReasoner())
    return planner.find_path(START, spec.items[0].node, agents["mobile_manipulator"]).is_feasible


def plan_jobs(repeats: int) -> list:
    from pace.twin import plant_world as PW
    jobs = []
    for kind, stand, heights in SERIES:
        body = (PW.TOTE if kind == "tote" else PW.MODULE)[2]
        for h in heights:
            ok = pace_reachable(sweep_spec(kind, stand, h))
            for r in range(repeats):
                jobs.append({"episode": f"{kind}_{stand}_{h:.2f}_r{r}", "kind": kind, "stand": stand, "height": h,
                             "grasp_height": round(h + body + PW.FRONT_GRASP_HEIGHT, 3), "repeat": r,
                             "pace_reachable": ok})
    return jobs


def _run_one(args):
    job, out_root, index, timeout, film = args
    from pace.twin.mission import MissionEpisode
    spec = sweep_spec(job["kind"], job["stand"], job["height"])
    eid = job["episode"]
    res = {}
    for attempt in (0, 1):                     # an error is the stack, not the mission: one more try
        ep = MissionEpisode(spec, [START, ZONE], "sweep", Path(out_root) / eid, index=index + 50 * attempt,
                            timeout=timeout, tag=eid, film=film, seed=job["repeat"])
        res = ep.run()
        if res["outcome"] != "error":
            break
    res.update({k: job[k] for k in ("episode", "kind", "stand", "height", "grasp_height", "repeat",
                                     "pace_reachable")})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/reach_sweep.csv")
    ap.add_argument("--out-root", default="results/reach_sweep")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="episode ids to run, e.g. module_bench_1.65_r0")
    ap.add_argument("--film", action="store_true", help="save frames of each pick (needs DISPLAY)")
    a = ap.parse_args()
    jobs = plan_jobs(a.repeats)
    if a.only:
        jobs = [j for j in jobs if j["episode"] in a.only]
    output = Path(a.output)
    rows = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {r["episode"] for r in rows if r.get("outcome") != "error"}
    rows = [r for r in rows if r["episode"] in done]
    todo = [j for j in jobs if j["episode"] not in done]
    print(f"{len(jobs)} sweep episodes, {len(jobs) - len(todo)} already done")
    for j in todo:
        print(f"  {j['episode']:24s} grasp at {j['grasp_height']:.2f} m  PACE reachable: {j['pace_reachable']}")
    if a.plan_only or not todo:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    work = [(j, a.out_root, i, a.timeout, a.film) for i, j in enumerate(todo)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procs, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(_run_one, work):
            rows.append(res)
            pd.DataFrame(rows).to_csv(output, index=False)
            print(f"[done] {res['episode']}: {res['outcome']} ({res['seconds']}s)", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
