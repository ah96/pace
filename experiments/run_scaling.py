"""How the explanation cost grows with the size of the facility.

The two hand-authored scenarios have nine waypoints each, so every runtime in
the results table is measured on a graph small enough that any method looks
instant. A facility is not nine waypoints, and the honest question is what the
method costs when it is not.

This sweeps procedurally generated lattices (scenarios/synthetic.py) from 9 to
several hundred waypoints and times one explanation per task, for PACE and for
the two baselines that also put the planner in the loop. The generated worlds
are not claimed to be realistic; they exist to vary one quantity.

    PYTHONPATH=. python experiments/run_scaling.py --output results/scaling.csv

What the shape should be, and why
---------------------------------
PACE questions k+2 routes and generates candidates per blocking condition on
each, so its work is proportional to the length of those routes and not to the
size of the graph: the growth it does show comes from the planner underneath
it, which has to search a bigger graph. Unconstrained re-plans once per member
of the full change space, and that space grows linearly in the number of
segments, so it should pick up an extra factor. Recourse re-plans once per
candidate per greedy step per diverse counterfactual, so it should be the
steepest of the three.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import networkx as nx
import pandas as pd

from pace.affordance import AffordanceReasoner
from pace.evaluation import predict_pace, run_baseline
from pace.navigation import AStarPlanner
from scenarios.synthetic import build_synthetic_world

#: Waypoint counts to sweep. The top end is chosen to keep the slowest method
#: inside a few minutes rather than to mark any limit of the approach.
DEFAULT_SIZES = (9, 16, 25, 49, 100, 196, 400)

#: Methods timed. The ranking-only baselines are omitted: they never call the
#: planner, so their cost is a property of the sort and says nothing about
#: whether explanation scales.
METHODS = ("pace", "unconstrained", "recourse")


def _tasks(graph, agents, n_tasks: int, seed: int) -> List[tuple]:
    """A spread of (robot, start, goal) triples with a path between them."""
    import random

    rng = random.Random(seed)
    ug = nx.DiGraph()
    for nid in graph.nodes:
        ug.add_node(nid)
    for fid, elist in graph.edges.items():
        for edge in elist:
            if edge.to_id in graph.nodes:
                ug.add_edge(fid, edge.to_id)

    names = sorted(graph.nodes)
    robots = sorted(agents)
    out: List[tuple] = []
    attempts = 0
    while len(out) < n_tasks and attempts < n_tasks * 50:
        attempts += 1
        s, g = rng.sample(names, 2)
        if nx.has_path(ug, s, g):
            out.append((robots[len(out) % len(robots)], s, g))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    p.add_argument("--tasks", type=int, default=12, help="Tasks timed per size.")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/scaling.csv")
    args = p.parse_args()

    reasoner = AffordanceReasoner()
    rows = []

    for n in args.sizes:
        onto, graph, agents = build_synthetic_world(n, seed=args.seed)
        n_segments = len(list(graph.all_space_individuals()))
        tasks = _tasks(graph, agents, args.tasks, args.seed)
        print(f"n={n:4d} waypoints, {n_segments:4d} segments, {len(tasks)} tasks")

        for robot, start, goal in tasks:
            agent = agents[robot]
            feasible = AStarPlanner(graph, reasoner).find_path(
                start, goal, agent).is_feasible
            for method in METHODS:
                t0 = time.perf_counter()
                if method == "pace":
                    predicted = predict_pace(graph, agent, start, goal,
                                             reasoner, k=args.k)
                else:
                    predicted = run_baseline(
                        method=method, graph=graph, agent=agent,
                        reasoner=reasoner, start=start, goal=goal,
                        k=args.k, seed=args.seed)
                rows.append({
                    "n_nodes": n, "n_segments": n_segments,
                    "robot": robot, "start": start, "goal": goal,
                    "feasible": feasible, "method": method,
                    "runtime_ms": (time.perf_counter() - t0) * 1000.0,
                    "num_returned": len(predicted),
                })

        summary = pd.DataFrame(rows)
        summary = summary[summary.n_nodes == n].groupby("method").runtime_ms.mean()
        print("   " + "  ".join(f"{m}={v:.1f}ms" for m, v in summary.items()))

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\nMean runtime (ms) per explanation:")
    pivot = df.pivot_table(index="n_nodes", columns="method",
                           values="runtime_ms", aggfunc="mean")
    print(pivot.round(2).to_string())

    # Empirical growth exponent: slope of log(time) against log(size). Reported
    # rather than a claimed complexity, because what a reader wants to know is
    # how the implementation behaves, not what the loop bounds permit.
    import numpy as np
    print("\nEmpirical exponent (log-log slope of runtime vs waypoints):")
    for method in METHODS:
        sub = pivot[method].dropna()
        slope = np.polyfit(np.log(sub.index.values.astype(float)),
                           np.log(sub.values), 1)[0]
        print(f"  {method:15s} {slope:.2f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
