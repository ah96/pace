"""One seaborn theme for every figure the project produces.

All plotting - the navigation-graph renderings in ``visualization.py`` and the
metric plots in ``experiments/plot_results.py`` - goes through this module, so
paper figures share a single visual identity instead of each script setting its
own fonts and colours.

Two styles are provided:

``apply_paper_style()``
    Sized for inclusion in the LNCS paper at ``width=\\linewidth``: small type,
    thin lines, no heavy grid. Figures are rendered at their final print size
    rather than rendered large and shrunk by LaTeX, so a 7-point label stays a
    7-point label on the page.

``apply_screen_style()``
    The same palette at screen sizes, for interactive exploration.

The categorical palette is seaborn's colour-blind-safe ``colorblind`` set. The
semantic colours used by the navigation graphs (chosen path, impassable edge,
and so on) are drawn from it in ``visualization.py`` so that a figure and a bar
chart placed on the same page do not clash.
"""

from __future__ import annotations

from typing import Dict

import seaborn as sns

#: Seaborn's colour-blind-safe categorical palette, used project-wide.
PALETTE = "colorblind"


def semantic_colors() -> Dict[str, str]:
    """Named colours for navigation-graph elements, taken from ``PALETTE``.

    Deriving them from the shared palette rather than hard-coding hex values
    keeps the graph figures and the metric plots in the same colour system.
    Purple, not orange, marks alternatives: the palette's orange sits next to
    its red, and "questioned alternative" and "impassable" are precisely the
    two categories a reader must not confuse.
    """
    c = sns.color_palette(PALETTE, 10).as_hex()
    return {
        "chosen":      c[2],   # green  - the path actually taken
        "alternative": c[4],   # purple - a questioned alternative
        "impassable":  c[3],   # red    - blocked for this robot
        "default":     "#b8b8b8",
        # Darkened so the two endpoint discs stay distinguishable from the
        # chosen-path and impassable-edge colours in the legend, where they
        # appear side by side as flat patches with no shape to tell them apart.
        "node_start":  "#016b4e",
        "node_goal":   "#8c3b00",
        "node_active": c[0],   # blue   - on a highlighted path
        "node_base":   "#9aa0a6",
    }


def apply_paper_style() -> None:
    """Seaborn theme for figures rendered at final print size."""
    sns.set_theme(
        context="paper",
        style="white",
        palette=PALETTE,
        font_scale=1.0,
        rc={
            "figure.dpi":        300,
            "savefig.dpi":       300,
            "savefig.bbox":      "tight",
            "axes.linewidth":    0.6,
            "lines.linewidth":   1.2,
            "patch.linewidth":   0.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon":    False,
            "pdf.fonttype":      42,   # embed TrueType, so text stays selectable
            "ps.fonttype":       42,
        },
    )


def apply_screen_style() -> None:
    """Same palette, sized for interactive viewing rather than print."""
    sns.set_theme(
        context="talk",
        style="whitegrid",
        palette=PALETTE,
        rc={"figure.dpi": 110, "pdf.fonttype": 42},
    )
