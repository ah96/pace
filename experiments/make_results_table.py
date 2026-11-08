"""Emit the LaTeX body of the results table straight from the results CSVs.

The table in the paper used to be typed by hand, which is how it came to
aggregate one row block by taking a maximum over three baselines and another by
taking a mean. Generating it removes the opportunity.

    PYTHONPATH=. python experiments/make_results_table.py \\
        --input results/plant_allpairs.csv --second results/generated_allpairs.csv.gz \\
        --include pace pace-opt recourse unconstrained semantic \\
        --output build/results_table.tex

Layout: one row per method, column groups per task type, and within each group
only the metrics that are defined and informative for it. A block of four
separate sub-tables would be more uniform and would not fit the page, and most
of what it would add is columns of n/a.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LABEL = {
    "pace": r"\pace", "pace-opt": r"\pace-Opt",
    "pace-noattr": r"\pace, blockers only", "pace-pooled": r"\pace, claims pooled",
    "recourse": "Recourse", "unconstrained": "Unconstrained", "semantic": "Semantic",
    "random": "Random",
}

#: Rows that are ablations of PACE rather than competing methods. They are set
#: below the others so a reader does not read them as baselines.
ABLATIONS = ("pace-noattr", "pace-pooled")

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
    key = ["scenario", "agent", "start", "goal"]          # what identifies one task
    agg: dict = {"_tasks": {}}
    for cat in {c for c, _ in COLUMNS}:
        sub = df[df.cat == cat]
        agg["_tasks"][cat] = sub.groupby(key).ngroups
        if len(sub):
            agg[cat] = sub.groupby("method")[metrics].mean()
    return agg


def block(agg: dict, wanted) -> list:
    """The body rows for one benchmark: methods first, then the ablations."""
    present = [m for m in wanted if m in agg[COLUMNS[0][0]].index]
    ordered = [m for m in present if m not in ABLATIONS] + [m for m in present if m in ABLATIONS]
    return [f"{LABEL[m]} & " + " & ".join(cells(agg, m)) + r" \\" for m in ordered]


def counts(agg: dict) -> str:
    """The five task-type counts, for the block's own heading row."""
    n = agg["_tasks"]
    fmt = lambda c: f"{n[c]:,}".replace(",", "{,}")       # 3{,}052, as in the text
    return "/".join(fmt(c) for c in ("infeasible, actionable", "infeasible, structural",
                                     "feasible, deviation", "feasible, no change",
                                     "infeasible, unrepairable"))


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
                   help="Restrict the table to these methods, in this order.")
    p.add_argument("--include-second", nargs="+", default=None,
                   help="Method list for the --second block, when it should differ from --include.")
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
                + r", " + counts(agg_) + r" tasks per type} \\")

    labels = args.labels or ["The authored plant", "Generated plants"]
    lines = [heading(labels[0], agg)] + block(agg, wanted)
    if args.second:
        agg2 = aggregate(pd.read_csv(args.second))
        lines += [heading(labels[1], agg2)] + block(agg2, args.include_second or wanted)

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
