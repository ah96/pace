"""The twin's verdicts, counted: what the paper reports from results/twin_episodes.csv.

    PYTHONPATH=. python experiments/summarise_twin.py [--input results/twin_episodes.csv]

A repair is *verified* when its route was not driven to the end before the
change and was after it; the twin *agrees* with PACE on where a route blocks
when the robot came to rest in (or at the mouth of) a passage PACE named; a
feasible route is *sound* when it was driven to the end.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/twin_episodes.csv")
    ap.add_argument("--tex", default=None, help="write \\newcommand lines for the paper")
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    df["arrived"] = df.outcome == "arrived"
    feas = df[df.kind == "feasible"]
    rep = df[df.kind == "repair"]
    before = rep[rep.phase == "before"].set_index(["robot", "route"])
    after = rep[rep.phase == "after"].set_index(["robot", "route"])
    both = before.join(after, lsuffix="_b", rsuffix="_a", how="inner")
    verified = both[(~both.arrived_b) & (both.arrived_a)]
    refuted = both[(~both.arrived_b) & (~both.arrived_a)]
    passable = both[both.arrived_b]
    import ast
    named_hits = 0
    for (robot, route_s), r in both[~both.arrived_b].iterrows():
        named = set(json.loads(r["named_b"])) if isinstance(r["named_b"], str) else set()
        segs = json.loads(r["route_segments_b"])
        stopped = r["stopped_at_b"]
        route = ast.literal_eval(route_s) if isinstance(route_s, str) else list(route_s)
        # the robot stops in the named passage, or in the zone just before it
        before_named = False
        if stopped in route:
            k = route.index(stopped)
            before_named = k < len(segs) and segs[k] in named
        named_hits += int(stopped in named or before_named)
    n_blocked = int((~both.arrived_b).sum())
    out = {
        "episodes": len(df), "routes_repair": len(both), "routes_feasible": len(feas),
        "feasible_sound": int(feas.arrived.sum()),
        "blocked_before": n_blocked, "verified": len(verified), "refuted": len(refuted),
        "already_passable": len(passable), "stopped_in_named": named_hits,
        "by_robot": {r: {"verified": int(((~g.arrived_b) & g.arrived_a).sum()), "routes": len(g)}
                     for r, g in both.groupby(level=0)},
        "outcomes_before": before.outcome.value_counts().to_dict(),
        "outcomes_after": after.outcome.value_counts().to_dict(),
    }
    print(json.dumps(out, indent=1))
    if a.tex:
        with open(a.tex, "w") as f:
            f.write(f"\\newcommand{{\\twinroutes}}{{{len(both)}}}\n")
            f.write(f"\\newcommand{{\\twinverified}}{{{len(verified)}}}\n")
            f.write(f"\\newcommand{{\\twinrefuted}}{{{len(refuted)}}}\n")
            f.write(f"\\newcommand{{\\twinpassable}}{{{len(passable)}}}\n")
            f.write(f"\\newcommand{{\\twinfeasible}}{{{len(feas)}}}\n")
            f.write(f"\\newcommand{{\\twinsound}}{{{int(feas.arrived.sum())}}}\n")
            f.write(f"\\newcommand{{\\twinnamed}}{{{named_hits}}}\n")
            f.write(f"\\newcommand{{\\twinblocked}}{{{n_blocked}}}\n")


if __name__ == "__main__":
    main()
