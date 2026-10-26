"""Run PACE benchmark experiments.

PACE: Pathwise Affordance Counterfactual Explanations

Each experiment is a (scenario, robot, start, goal) tuple.  For every
combination we run the PACE counterfactual engine + all baselines, compute
metrics against operational ground truth, and save results to CSV.

Example:
    python experiments/run_pace.py --k 3 --output results/pace_results.csv

For a quick smoke test:
    python experiments/run_pace.py --smoke --output results/pace_smoke.csv
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd

from pace.affordance import AffordanceReasoner
from pace.counterfactual import DEFAULT_PRIORITY
from pace.navigation import AStarPlanner
from pace.evaluation import (
    compute_metrics,
    score_prediction_set,
    compute_ground_truth,
    compute_ground_truth_routewise,
    precision_at_k,
    recall_at_k,
    predict_pace as predict_pace,
    repairable,
    run_baseline,
)
from scenarios.plant import build_plant_world, build_random_plant

#: Baselines, plus their actionability-filtered variants. The ``-act`` forms
#: rank the same way but only over properties a user can change, which is the
#: ablation separating PACE's search from its knowledge of that split.
_BASELINES = ("random", "distance", "semantic", "occupancy", "unconstrained",
              "recourse")

#: Ablations of PACE itself, which isolate the parts of the method that are
#: declared rather than computed. ``pace-noattr`` generates candidates only from
#: hard blockers, as the method did before cost attribution; ``pace-rndprio``
#: keeps the full generator but shuffles the priority order, so what survives
#: the shuffle is what verification rather than the ordering is responsible for;
#: ``pace-pooled`` pools repair and saving claims into one ranked list, the
#: design before the two were separated, which is the ablation that reproduces
#: the labelling failure the separation exists to prevent.
_ABLATIONS = ("pace-noattr", "pace-rndprio", "pace-pooled")

SUPPORTED_METHODS = ({"pace", "pace-opt"} | set(_ABLATIONS) | set(_BASELINES)
                     | {b + "-act" for b in _BASELINES}
                     # The set-search form of the excuse baseline: exhaustive
                     # ascending-cardinality enumeration, run separately because
                     # its cost is measured in seconds per task, not
                     # milliseconds.
                     | {"unconstrained-sets"})


def _all_pairs(graph, agents):
    """Every (robot, start, goal) the topology connects, in a stable order."""
    import networkx as nx

    ug = nx.DiGraph()
    for nid in graph.nodes:
        ug.add_node(nid)
    for fid, elist in graph.edges.items():
        for edge in elist:
            if edge.to_id in graph.nodes:
                ug.add_edge(fid, edge.to_id)

    out = []
    for robot in sorted(agents):
        for start in sorted(graph.nodes):
            for goal in sorted(graph.nodes):
                if start != goal and nx.has_path(ug, start, goal):
                    out.append((robot, start, goal))
    return out


def build_tasks(smoke: bool = False, all_pairs: bool = False, plants: int = 0,
                plant_seed0: int = 0, rows: int = 2, cols: int = 5):
    """Return list of (scenario_name, onto, graph, agent_name, agent, start, goal).

    The authored plant first; with ``all_pairs``, every (robot, start, goal)
    with start != goal and a topological path between them, instead of the
    hand-picked showcase tasks.  ``plants`` adds that many generated plants
    (seeds ``plant_seed0`` onwards), always as all pairs: the point of them is
    cases nobody chose.
    """
    tasks = []
    onto_p, graph_p, agents_p = build_plant_world()
    plant_queries = [
        ("tote_amr",           "dock", "shipping"),
        ("forklift_agv",         "dock", "shipping"),
        ("mobile_manipulator", "dock", "shipping"),
        ("yard_ugv",  "dock", "shipping"),
        ("forklift_agv",         "module_store", "shipping"),   # the conjunctive underpass
        ("forklift_agv",         "receiving", "disassembly_B"), # staged pallets
        ("tote_amr",           "dock", "yard"),               # the ramp
        ("mobile_manipulator", "quarantine", "sorting"),
    ]
    if all_pairs:
        plant_queries = _all_pairs(graph_p, agents_p)
    elif smoke:
        plant_queries = plant_queries[:3]
    for agent_name, start, goal in plant_queries:
        tasks.append(("plant", onto_p, graph_p, agent_name, agents_p[agent_name], start, goal))
    for seed in range(plant_seed0, plant_seed0 + plants):
        onto_r, graph_r, agents_r = build_random_plant(seed, rows=rows, cols=cols)
        queries = _all_pairs(graph_r, agents_r)
        if smoke:
            queries = queries[:4]
        for agent_name, start, goal in queries:
            tasks.append((f"plant_{seed}", onto_r, graph_r, agent_name, agents_r[agent_name], start, goal))
    return tasks


def predict(method, graph, agent, reasoner, start, goal, k, seed, stats=None):
    """One method's ranked recommendations for one task.

    Both runners go through here, so the external benchmark cannot quietly
    differ from the authored one in how a method is invoked.
    """
    if method in ("pace", "pace-opt") or method in _ABLATIONS:
        order = None
        if method == "pace-rndprio":
            # Seeded per task, so the permutation is a different one for every
            # task rather than one unlucky order applied to all.
            order = list(DEFAULT_PRIORITY)
            random.Random(hash((seed, agent.name, start, goal))
                          & 0xFFFFFFFF).shuffle(order)
        return predict_pace(
            graph, agent, start, goal, reasoner, k=k,
            search="min_cardinality" if method == "pace-opt" else "greedy",
            stats=stats,
            attribute_costs=method != "pace-noattr",
            priority_order=order,
            separate_claims=method != "pace-pooled",
        )
    return run_baseline(method=method, graph=graph, agent=agent,
                        reasoner=reasoner, start=start, goal=goal,
                        k=k, seed=seed)


def categorise(feasible: bool, num_truth: int, can_repair: bool = True) -> str:
    """
    The five task types the results table groups by.

    A task is *feasible* when the robot already reaches its goal. It has an
    *actionable* fix when at least one operator-changeable property, applied
    alone, restores feasibility or lowers the optimal cost - which is exactly
    the ground-truth set being non-empty. An infeasible task with an empty
    actionable ground truth is *structural* if some change to the plant
    (construction included, conjunctions included) restores it, and
    *unrepairable* if none does: the cause is on the robot's side, as when a
    robot without an arm is sent to pick something up, and the honest answer
    recommends nothing.

    Derived here rather than by hand so the table and the CSV cannot disagree.
    """
    if feasible:
        return "feasible, deviation" if num_truth else "feasible, no change"
    if num_truth:
        return "infeasible, actionable"
    return "infeasible, structural" if can_repair else "infeasible, unrepairable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PACE semantic experiments.")
    parser.add_argument("--k", type=int, default=3, help="Number of explanation factors.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pace", "pace-opt", "pace-noattr", "pace-rndprio", "pace-pooled",
                 "semantic", "distance", "occupancy",
                 "random", "unconstrained", "recourse",
                 "semantic-act", "unconstrained-act", "random-act",
                 "distance-act", "occupancy-act", "recourse-act"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/pace_results.csv")
    parser.add_argument("--smoke", action="store_true", help="Run first two tasks per scenario.")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Sweep every connected (robot, start, goal) instead "
                             "of the hand-picked tasks.")
    parser.add_argument("--plants", type=int, default=0,
                        help="Number of generated plants to add (all pairs each).")
    parser.add_argument("--plant-seed0", type=int, default=0)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--only-plants", action="store_true",
                        help="Skip the authored plant (for sharded generated runs).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = set(args.methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"Unsupported methods: {sorted(unknown)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    reasoner = AffordanceReasoner()
    tasks    = build_tasks(args.smoke, args.all_pairs, args.plants, args.plant_seed0,
                           args.rows, args.cols)
    if args.only_plants:
        tasks = [t for t in tasks if t[0] != "plant"]
    rows     = []

    for scenario_name, onto, graph, agent_name, agent, start, goal in tasks:
        print(f"\n[{scenario_name}] {agent_name}: {start} → {goal}")

        truth = compute_ground_truth(graph, agent, start, goal, reasoner)
        feasible = AStarPlanner(graph, reasoner).find_path(start, goal, agent).is_feasible
        can_repair = feasible or repairable(graph, agent, start, goal, reasoner)
        cat = categorise(feasible, len(truth), can_repair)
        # The route-wise target scores a method on the question an explanation
        # is asked, rather than on whether it improved the optimal path. Both
        # are computed here so the two can be reported side by side.
        truth_route = compute_ground_truth_routewise(
            graph, agent, start, goal, reasoner, k=args.k)
        print(f"  ground truth: {len(truth)} optimal-path, "
              f"{len(truth_route)} route-wise factor(s)")

        for method in args.methods:
            stats: dict = {}
            t0 = time.perf_counter()
            predicted = predict(method, graph, agent, reasoner, start, goal,
                                args.k, args.seed, stats)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Score every method on what its advice actually achieves, so the
            # comparison does not depend on how each engine reports itself.
            feas_rec, cost_improve = score_prediction_set(
                graph, agent, start, goal, reasoner, predicted, args.k
            )
            metrics = compute_metrics(predicted, truth, args.k,
                                      feas_rec=feas_rec, cost_improve=cost_improve)
            rows.append({
                "scenario":      scenario_name,
                "agent":         agent_name,
                "start":         start,
                "goal":          goal,
                "method":        method,
                "feasible":      feasible,
                "cat":           cat,
                "repairable":    can_repair,
                "k":             args.k,
                "num_truth":     len(truth),
                "num_truth_route": len(truth_route),
                "num_predicted": len(predicted),
                "mean_changeset_size": stats.get("mean_changeset_size", float("nan")),
                "runtime_ms":    elapsed_ms,
                "precision_at_k_route": precision_at_k(predicted, truth_route, args.k),
                "recall_at_k_route":    recall_at_k(predicted, truth_route, args.k),
                **metrics,
            })
            print(f"  [{method:>14}] P@k={metrics['precision_at_k']:.2f}  "
                  f"R@k={metrics['recall_at_k']:.2f}  "
                  f"Adm={metrics['actionable_fraction']:.2f}  "
                  f"FeasRec={metrics['feasibility_recovery']:.0f}")

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"\nSaved {len(df)} rows to {output}")
    print(df.groupby("method")[
        ["precision_at_k", "recall_at_k", "precision_at_k_route",
         "recall_at_k_route", "feasibility_recovery",
         "actionable_fraction", "num_returned", "mean_changeset_size",
         "runtime_ms"]
    ].mean().to_string())


if __name__ == "__main__":
    main()
