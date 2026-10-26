"""Emit the LaTeX body of the results table straight from the results CSV.

The table in the paper used to be typed by hand, which is how it came to
aggregate one row block by taking a maximum over three baselines and another by
taking a mean. Generating it removes the opportunity.

    PYTHONPATH=. python experiments/make_results_table.py \\
        --input results/pace_results_allpairs.csv \\
        --output build/results_table.tex

Layout: one row per method, column groups per task type, and within each group
only the metrics that are defined and informative for it. A block of four
separate sub-tables would be more uniform and would not fit the page, and most
of what it would add is columns of n/a.

Actionability-filtered variants are folded together: the generator groups them
by their rounded metrics and emits one row per distinct behaviour, so a filter
row is printed once for however many baselines behave identically under it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LABEL = {
    "pace": r"\pace", "pace-opt": r"\pace-Opt",
    "pace-noattr": r"\pace, blockers only", "pace-rndprio": r"\pace, order shuffled",
    "pace-pooled": r"\pace, claims pooled",
    "recourse": "Recourse",
    "unconstrained": "Unconstrained",
    "unconstrained-sets": r"Unconstr.\ sets $\leq$3", "semantic": "Semantic",
    "distance": "Distance", "occupancy": "Occupancy", "random": "Random",
}

#: Rows that are ablations of PACE rather than competing methods. They are set
#: below a rule so a reader does not read them as baselines.
ABLATIONS = ("pace-noattr", "pace-rndprio", "pace-pooled")
SHORT = {
    "recourse": "Recourse", "unconstrained": "Unconstr.", "semantic": "Semantic",
    "distance": "Dist.", "occupancy": "Occ.", "random": "Rand.",
}

#: (task type, CSV column) pairs in the order they appear across the table.
COLUMNS = [
    ("infeasible, actionable", "recall_at_k"),
    ("infeasible, actionable", "actionable_fraction"),
    ("infeasible, actionable", "feasibility_recovery"),
    ("infeasible, structural", "feasibility_recovery"),
    ("feasible, deviation",    "recall_at_k"),
    ("feasible, no change",    "num_returned"),
    ("infeasible, unrepairable", "num_returned"),
]


def cells(agg: dict, method: str) -> list[str]:
    """The row of numbers for one method, in COLUMNS order, plus its runtime."""
    out = []
    for cat, col in COLUMNS:
        if cat not in agg or method not in agg[cat].index:
            out.append("--")
            continue
        val = agg[cat].loc[method, col]
        out.append(f"{val:.1f}" if col == "num_returned" else f"{val:.2f}")
    runtime = sum(agg[c].loc[method, "runtime_ms"] * n
                  for c, n in agg["_tasks"].items() if c in agg and method in agg[c].index)
    out.append(f"{runtime / max(1, sum(agg['_tasks'].values())):.0f}")
    return out


def aggregate(df: pd.DataFrame) -> dict:
    """Per-method means per task type, plus how many tasks each type holds."""
    metrics = sorted({c for _, c in COLUMNS} | {"runtime_ms"})

    # What identifies one task. Floor-plan worlds all carry scenario
    # "floorplan" and reuse node ids r0, r1, ..., so without the plan the same
    # (robot, start, goal) triple in forty different buildings would collapse to
    # one task and every count in the header would be forty times too small.
    key = ["scenario", "agent", "start", "goal"]
    if "plan_id" in df.columns:
        key.insert(1, "plan_id")
    # Pooled seeds re-run the same task under different sampled properties;
    # each draw is its own task instance, and the counts should say so.
    if "property_seed" in df.columns:
        key.insert(1, "property_seed")

    agg: dict = {"_tasks": {}}
    for cat in {c for c, _ in COLUMNS}:
        sub = df[df.cat == cat]
        agg["_tasks"][cat] = sub.groupby(key).ngroups
        if len(sub):
            agg[cat] = sub.groupby("method")[metrics].mean()
    return agg


def block(agg: dict, wanted, no_filter_rows: bool,
          filter_rows_for=None) -> list:
    """The body rows for one benchmark, ablations and filter rows included."""
    present = [m for m in wanted if m in agg[COLUMNS[0][0]].index]

    lines = [f"{LABEL[m]} & " + " & ".join(cells(agg, m)) + r" \\"
             for m in present if m not in ABLATIONS]

    for m in [m for m in present if m in ABLATIONS]:
        lines.append(f"{LABEL[m]} & " + " & ".join(cells(agg, m)) + r" \\")

    groups: dict = {}
    if not no_filter_rows:
        for m in (filter_rows_for if filter_rows_for is not None else present):
            act = f"{m}-act"
            if act not in agg[COLUMNS[0][0]].index:
                continue
            row = tuple(cells(agg, act)[:-1])
            if row == tuple(cells(agg, m)[:-1]):
                continue
            groups.setdefault(row, []).append(m)
    for row, members in groups.items():
        names = "/".join(SHORT.get(m, LABEL[m]) for m in members)
        runtime = max(float(cells(agg, f"{m}-act")[-1]) for m in members)
        lines.append(f"{names} $+$ filter & " + " & ".join(row)
                     + f" & {runtime:.0f}" + r" \\")
    return lines


def counts(agg: dict) -> str:
    """The five task-type counts, for the block's own heading row."""
    n = agg["_tasks"]
    return (f"{n['infeasible, actionable']}/{n['infeasible, structural']}/"
            f"{n['feasible, deviation']}/{n['feasible, no change']}/{n['infeasible, unrepairable']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/plant_allpairs.csv")
    p.add_argument("--output", default=None)
    # A second benchmark is stacked into the same tabular rather than given a
    # table of its own: the columns are identical, and a reader comparing the
    # two blocks wants them under one set of headings, not a page apart.
    p.add_argument("--second", default=None,
                   help="A second results CSV, stacked as a labelled block.")
    p.add_argument("--labels", nargs=2, default=None,
                   help="Block labels for --input and --second.")
    p.add_argument("--include", nargs="+", default=None,
                   help="Restrict the table to these methods, in this order. "
                        "Used for the compact external-benchmark table, where "
                        "the sanity-floor rankers add nothing a reader needs.")
    p.add_argument("--include-second", nargs="+", default=None,
                   help="Method list for the --second block, when it should "
                        "differ from --include. The sanity floors earn their "
                        "row once; repeating them in the second block costs "
                        "lines the page does not have.")
    p.add_argument("--no-filter-rows", action="store_true",
                   help="Omit the actionability-filtered block.")
    p.add_argument("--filter-rows-for", nargs="+", default=None,
                   help="Show filtered variants only for these methods. The "
                        "filtered sanity floors repeat the verifying methods' "
                        "signature; one row per behaviour still holds, this "
                        "just narrows whose behaviour is worth a row.")
    args = p.parse_args()

    agg = aggregate(pd.read_csv(args.input))
    wanted = args.include or list(LABEL)

    # The whole tabular is emitted, not just its rows. A file of rows alone has
    # to be \input inside the tabular, and a file ending in "\\" leaves TeX
    # unable to accept the \bottomrule that follows the \input. Emitting the
    # environment keeps the \input at document level, where nothing is fragile.
    # The task counts move out of the column headings and into each block's own
    # heading row, since two benchmarks under one set of headings cannot share
    # them -- and a heading wider than the columns it spans widens the table
    # past the text block, which a reader sees as a table in the margin.
    def stack(top: str, bottom: str) -> str:
        return r"\shortstack{" + top + r"\\" + bottom + "}"

    header = "\n".join([
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{" + stack("Infeasible,", "actionable") + "}",
        r" & " + stack("Infeasible,", "structural"),
        r" & \shortstack{Devi-\\ation} & \shortstack{No\\change} & \shortstack{Unre-\\pairable} & All \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}"
        r"\cmidrule(lr){7-7}\cmidrule(lr){8-8}\cmidrule(lr){9-9}",
        r"\textbf{Method} & R@3 & Act & FeasRec & FeasRec",
        r"  & R@3 & \#rec & \#rec & ms \\",
        r" & \scriptsize$\uparrow1$ & \scriptsize$\uparrow1$ & \scriptsize$\uparrow1$",
        # Structural Act is left out: its ideal is "low but above zero"
        # (verified conjunctions genuinely mix door and wall), which needs a
        # caption note the page cannot spare; the prose quotes it instead.
        r" & \scriptsize$\uparrow1$ & \scriptsize$\uparrow1$",
        r" & \scriptsize$\downarrow0$ & \scriptsize$\downarrow0$ & \scriptsize$\downarrow$ \\",
    ])

    def heading(label: str, agg_: dict) -> str:
        return (r"\midrule" + "\n" + r"\multicolumn{9}{@{}l}{\itshape " + label
                + r", " + counts(agg_) + r" tasks by type} \\")

    labels = args.labels or ["The authored plant", "Generated plants"]
    lines = [heading(labels[0], agg)] + block(agg, wanted, args.no_filter_rows,
                                              filter_rows_for=args.filter_rows_for)
    if args.second:
        agg2 = aggregate(pd.read_csv(args.second))
        lines += [heading(labels[1], agg2)]
        # The filtered variants are omitted from the second block: what they
        # demonstrate -- that giving a ranker the actionable/structural split
        # trades false advice for none -- is established once in the first, and
        # repeating it costs four rows a page cannot spare.
        lines += block(agg2, args.include_second or wanted, no_filter_rows=True)

    body = "\n".join([header] + lines + [r"\bottomrule", r"\end{tabular}"])

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body + "\n")
        print(f"wrote {out}\n")
    print(body)
    print("\n% task counts: " + ", ".join(
        f"{c} = {n}" for c, n in sorted(agg["_tasks"].items())))


if __name__ == "__main__":
    main()
