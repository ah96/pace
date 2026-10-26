"""Summarise the reach sweep: PACE's reach verdict against the twin's pick, height by height.

    PYTHONPATH=. python experiments/summarise_reach_sweep.py --input results/reach_sweep.csv

The twin *reaches* an item when MoveIt plans the arm to a grasp (every outcome
but ``unreachable``) and *picks* it when the item also rises with the gripper.
PACE's Reachable rule is judged against reaching; picking is reported beside it.
"""

from __future__ import annotations

import argparse

import pandas as pd

ARM = ("picked", "grasped", "missed", "unreachable")      # outcomes in which the arm was tried


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/reach_sweep.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    tried = df[df.outcome.isin(ARM)].copy()
    tried["reached"] = tried.outcome != "unreachable"
    tried["picked"] = tried.outcome == "picked"
    other = df[~df.outcome.isin(ARM)]
    if len(other):
        print(f"{len(other)} episodes never reached the arm stage: "
              + ", ".join(f"{e} ({o})" for e, o in zip(other.episode, other.outcome)))
    g = (tried.groupby(["kind", "stand", "height"])
         .agg(grasp_height=("grasp_height", "first"), pace=("pace_reachable", "first"),
              n=("reached", "size"), reached=("reached", "sum"), picked=("picked", "sum"),
              outcomes=("outcome", lambda s: "/".join(s)))
         .reset_index())
    print(g.to_string(index=False))
    agree = (tried.reached == tried.pace_reachable)
    print(f"\nPACE's reach verdict agrees with the twin on {agree.sum()} of {len(tried)} episodes")
    for name, sub in tried.groupby("kind"):
        wrong = sub[sub.reached != sub.pace_reachable]
        print(f"  {name}: {len(sub) - len(wrong)}/{len(sub)} agree"
              + (f"; disagreements at {sorted(set(wrong.height))} m (grasp at {sorted(set(wrong.grasp_height))} m)"
                 if len(wrong) else ""))
    for name, sub in tried[tried.reached].groupby("kind"):
        print(f"  {name}: reached at grasp heights {sub.grasp_height.min():.2f}-{sub.grasp_height.max():.2f} m")


if __name__ == "__main__":
    main()
