"""
pace/visualization.py
==============================
Navigation graph visualization using NetworkX and Matplotlib.

Entry point
-----------
    from pace.visualization import draw_navigation_graph

    draw_navigation_graph(
        graph,
        chosen_path=path,
        alternative_paths=alternatives,
        reasoner=nav.reasoner,
        agent=robot,
        title="Hospital - delivery_bot",
    )

Styling comes from ``pace.plotting``: one seaborn theme and one colour-blind-safe
palette for every figure the project produces.

Color coding
------------
Nodes:
  green  - start of the chosen path
  red    - goal of the chosen path
  blue   - intermediate node on any highlighted path
  grey   - unreferenced node

Edges:
  green solid    - edge on the chosen path
  purple solid   - edge on an alternative path (not chosen)
  red dashed     - impassable edge for this agent (cost = ∞)
  light grey     - default / background edge

Edge labels (when *reasoner* and *agent* are provided):
  Finite cost shown as a decimal; ∞ for impassable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")           # non-interactive backend - safe in all envs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from .affordance import AffordanceReasoner
from .navigation import NavigationGraph, NavPath
from .ontology import OntologyIndividual
from .plotting import apply_paper_style, apply_screen_style, semantic_colors


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

# Taken from the project-wide seaborn palette (see pace/plotting.py) so graph
# figures and metric plots share one colour system.
_C = semantic_colors()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_navigation_graph(
    graph:             NavigationGraph,
    chosen_path:       Optional[NavPath]       = None,
    alternative_paths: Optional[List[NavPath]] = None,
    reasoner:          Optional[AffordanceReasoner]   = None,
    agent:             Optional[OntologyIndividual]   = None,
    title:             str                     = "Navigation Graph",
    save_path:         Optional[str]           = None,
    show:              bool                    = True,
    figsize:           Optional[Tuple[float, float]] = None,
    paper_mode:        bool                    = False,
    dpi:               int                     = 150,
    show_title:        Optional[bool]          = None,
    legend:            Optional[str]           = None,
    label_edges:       Optional[str]           = None,
    start:             Optional[str]           = None,
    goal:              Optional[str]           = None,
) -> plt.Figure:
    """
    Render *graph* as a directed 2-D layout, optionally highlighting paths.

    Parameters
    ----------
    graph             : the NavigationGraph to render
    chosen_path       : path to highlight in green
    alternative_paths : list of alternative paths to highlight in orange
    reasoner + agent  : when both supplied, edge labels show affordance costs
    title             : figure title
    save_path         : if given, save the figure to this file path (PNG/SVG/...)
    show              : call plt.show() when True (set False in headless scripts)
    figsize           : (width, height) in inches; defaults per mode (see below)
    paper_mode        : render at true print size for a one-column LaTeX figure
    dpi               : resolution used when saving to file
    show_title        : draw the title inside the axes; defaults to False in
                        paper_mode, where the LaTeX caption carries it and an
                        in-figure title only duplicates text and steals height
    legend            : "below", "inside" or None. Defaults to "below", or to
                        None in paper_mode, where a single shared legend for
                        the whole LaTeX figure is cheaper than one per panel
    label_edges       : "all" or "highlighted". Defaults to "all", or to
                        "highlighted" in paper_mode: numbers on background
                        edges are the main source of clutter at print size and
                        none of them is referenced in the text
    start / goal      : node ids to mark as endpoints. Only needed when there
                        is no feasible path to read them off, which is exactly
                        the case the figure is usually illustrating: without
                        them an infeasible task renders as an undifferentiated
                        grey graph with nothing to anchor the reader

    Sizing
    ------
    ``paper_mode`` renders at roughly the width the figure will occupy on the
    page, so ``\\includegraphics[width=\\linewidth]`` scales it about 1:1 and a
    7-point label stays a 7-point label. Rendering large and letting LaTeX
    shrink the result is what made earlier versions unreadable: a 9-inch canvas
    placed in a 2-inch slot divides every font size by four.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if paper_mode:
        # LNCS \textwidth is 122 mm (4.8 in). Render at that width so the
        # figure is placed at its natural size rather than being downscaled.
        figsize        = figsize or (4.8, 1.55)
        dpi            = max(dpi, 300)
        _title_fs      = 8.0
        _node_label_fs = 7.0
        _edge_label_fs = 6.0
        _legend_fs     = 6.5
        _sz_start_goal = 190
        _sz_active     = 150
        _sz_base       = 110
        _arrow_sz      = 7
        _w_default, _w_impass, _w_alt, _w_chosen = 0.6, 1.7, 1.5, 1.9
        _label_dy      = 0.105
        _margin        = 0.11
        show_title     = False if show_title is None else show_title
        legend         = legend                      # None unless asked for
        label_edges    = label_edges or "highlighted"
    else:
        figsize        = figsize or (13, 8)
        _title_fs      = 14
        _node_label_fs = 8
        _edge_label_fs = 7
        _legend_fs     = 8
        _sz_start_goal = 600
        _sz_active     = 450
        _sz_base       = 350
        _arrow_sz      = 14
        _w_default, _w_impass, _w_alt, _w_chosen = 1.0, 1.5, 2.0, 2.5
        _label_dy      = 0.055
        _margin        = 0.16
        show_title     = True if show_title is None else show_title
        legend         = "below" if legend is None else legend
        label_edges    = label_edges or "all"

    apply_paper_style() if paper_mode else apply_screen_style()

    fig, ax = plt.subplots(figsize=figsize)
    if show_title:
        ax.set_title(title, fontsize=_title_fs, fontweight="bold", pad=6)
    ax.axis("off")

    # Build a networkx DiGraph for drawing
    g = nx.DiGraph()
    pos: Dict[str, Tuple[float, float]] = {}

    for nid, node in graph.nodes.items():
        g.add_node(nid)
        pos[nid] = node.position

    # Normalise positions so nodes fill the axes (avoids large blank areas)
    pos = _normalise_pos(pos)

    for from_id, edge_list in graph.edges.items():
        for edge in edge_list:
            if edge.to_id in graph.nodes:
                g.add_edge(from_id, edge.to_id, edge_obj=edge)

    # Categorise edges
    chosen_edges:      Set[Tuple[str, str]] = set()
    alternative_edges: Set[Tuple[str, str]] = set()
    impassable_edges:  Set[Tuple[str, str]] = set()

    if chosen_path and chosen_path.is_feasible:
        for e in chosen_path.edges:
            chosen_edges.add((e.from_id, e.to_id))

    if alternative_paths:
        for alt in alternative_paths:
            if alt.is_feasible:
                for e in alt.edges:
                    key = (e.from_id, e.to_id)
                    if key not in chosen_edges:
                        alternative_edges.add(key)

    if reasoner and agent:
        for from_id, edge_list in graph.edges.items():
            for edge in edge_list:
                cost, _ = reasoner.navigation_cost(
                    edge.space_individual, agent, base_distance=edge.distance
                )
                if cost == math.inf:
                    impassable_edges.add((from_id, edge.to_id))

    default_edges = [
        (u, v) for u, v in g.edges()
        if (u, v) not in chosen_edges
        and (u, v) not in alternative_edges
        and (u, v) not in impassable_edges
    ]

    # Draw edges by category
    _draw_edges(ax, g, pos, list(default_edges),     _C["default"],     False, _w_default, 0.6, _arrow_sz)
    _draw_edges(ax, g, pos, list(impassable_edges),  _C["impassable"],  True,  _w_impass,  0.95, _arrow_sz)
    _draw_edges(ax, g, pos, list(alternative_edges), _C["alternative"], False, _w_alt,     0.85, _arrow_sz)
    _draw_edges(ax, g, pos, list(chosen_edges),      _C["chosen"],      False, _w_chosen,  0.95, _arrow_sz)

    # Categorise and draw nodes
    start_node = start or (chosen_path.nodes[0]  if (chosen_path and chosen_path.nodes) else None)
    goal_node  = goal  or (chosen_path.nodes[-1] if (chosen_path and chosen_path.nodes) else None)
    active_nodes = set()
    if chosen_path:
        active_nodes.update(chosen_path.nodes)
    if alternative_paths:
        for alt in alternative_paths:
            active_nodes.update(alt.nodes)

    node_colors = []
    node_sizes  = []
    for nid in g.nodes():
        if nid == start_node:
            node_colors.append(_C["node_start"])
            node_sizes.append(_sz_start_goal)
        elif nid == goal_node:
            node_colors.append(_C["node_goal"])
            node_sizes.append(_sz_start_goal)
        elif nid in active_nodes:
            node_colors.append(_C["node_active"])
            node_sizes.append(_sz_active)
        else:
            node_colors.append(_C["node_base"])
            node_sizes.append(_sz_base)

    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

    # Node labels: dark text in a white rounded box, offset just below each node.
    # Drawing the full name on a box (rather than white text centred on the disk)
    # keeps long ids like "icu_main"/"mezzanine"/"corridor_a" fully legible and
    # avoids the white-on-white overflow that truncated earlier figures.
    label_pos = _place_labels(g, pos, _label_dy)
    label_texts = nx.draw_networkx_labels(
        g, label_pos,
        font_size=_node_label_fs, font_color="#2c3e50", font_weight="bold", ax=ax,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.85, "ec": "#bdc3c7"},
    )
    for _t in label_texts.values():         # never clip labels at the axes edge
        _t.set_clip_on(False)
    ax.margins(_margin)                      # padding so edge-node labels are not cut

    # Edge cost labels
    if reasoner and agent:
        # Which edges get a number. Background edges carry no explanatory
        # weight and their labels are the main source of clutter at print
        # size, so "highlighted" keeps only the path and blocker costs.
        labelled = (
            chosen_edges | alternative_edges | impassable_edges
            if label_edges == "highlighted" else None
        )
        edge_labels: Dict[Tuple[str, str], str] = {}
        for from_id, edge_list in graph.edges.items():
            for edge in edge_list:
                if edge.to_id not in graph.nodes:
                    continue
                key = (from_id, edge.to_id)
                if labelled is not None and key not in labelled:
                    continue
                cost, _ = reasoner.navigation_cost(
                    edge.space_individual, agent, base_distance=edge.distance
                )
                label = "∞" if cost == math.inf else f"{cost:.1f}"
                edge_labels[key] = label
        nx.draw_networkx_edge_labels(
            g, pos, edge_labels=edge_labels,
            font_size=_edge_label_fs, font_color="#2c3e50", ax=ax,
            rotate=False,   # keep labels horizontal (so "∞" is not mistaken for "8")
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7, "ec": "none"},
        )

    # Legend. Each panel repeating the same six keys wastes the height that
    # makes the graph legible, so paper figures draw it once via
    # draw_legend_strip() and share it across panels.
    if legend == "below":
        fig.legend(
            handles=legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            fontsize=_legend_fs,
            framealpha=0.9,
            ncol=6,         # all 6 items in one row
            borderaxespad=0.2,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])  # reserve 8 % for the legend
    elif legend == "inside":
        ax.legend(
            handles=legend_handles(),
            loc="upper center",
            fontsize=_legend_fs,
            framealpha=0.85,
            ncol=3,
            borderaxespad=0.1,
            handlelength=1.2,
            columnspacing=1.0,
        )
        fig.tight_layout(pad=0.2)
    else:
        fig.tight_layout(pad=0.2)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Shared legend
# ---------------------------------------------------------------------------

def legend_handles() -> List[mpatches.Patch]:
    """The six colour keys, in the order they are read: edges, then endpoints."""
    return [
        mpatches.Patch(color=_C["chosen"],      label="Chosen path"),
        mpatches.Patch(color=_C["alternative"], label="Alternative path"),
        mpatches.Patch(color=_C["impassable"],  label="Impassable (∞)"),
        mpatches.Patch(color=_C["default"],     label="Background edge"),
        mpatches.Patch(color=_C["node_start"],  label="Start"),
        mpatches.Patch(color=_C["node_goal"],   label="Goal"),
    ]


def draw_legend_strip(
    save_path: Optional[str] = None,
    width:     float = 4.8,
    fontsize:  float = 6.5,
    ncol:      int   = 6,
    dpi:       int   = 300,
) -> plt.Figure:
    """Render the colour key alone, as a wide strip to place under a figure.

    Multi-panel paper figures share one legend rather than repeating it in
    every panel, which returns that height to the graphs themselves.
    """
    fig = plt.figure(figsize=(width, 0.28))
    fig.legend(
        handles=legend_handles(),
        loc="center",
        fontsize=fontsize,
        ncol=ncol,
        frameon=False,
        handlelength=1.1,
        handleheight=0.8,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    return fig

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _draw_edges(
    ax:        plt.Axes,
    g:         nx.DiGraph,
    pos:       Dict[str, Tuple[float, float]],
    edges:     List[Tuple[str, str]],
    color:     str,
    dashed:    bool,
    width:     float,
    alpha:     float,
    arrowsize: int = 14,
) -> None:
    if not edges:
        return
    nx.draw_networkx_edges(
        g, pos,
        edgelist    = edges,
        edge_color  = color,
        style       = "dashed" if dashed else "solid",
        width       = width,
        alpha       = alpha,
        arrows      = True,
        arrowsize   = arrowsize,
        arrowstyle  = "-|>",
        ax          = ax,
        connectionstyle = "arc3,rad=0.08",
    )


def _place_labels(
    g:   nx.DiGraph,
    pos: Dict[str, Tuple[float, float]],
    dy:  float,
) -> Dict[str, Tuple[float, float]]:
    """Offset each node's label to whichever vertical side is less crowded.

    A label pinned below every node sits directly on top of any edge leaving
    that node downwards, which is what buried the impassable ramp edge in the
    warehouse figure. Summing the directions to a node's neighbours gives the
    side the edges lean towards; the label goes to the other one. Nodes whose
    edges are purely horizontal keep the default position below.
    """
    placed: Dict[str, Tuple[float, float]] = {}
    for nid, (x, y) in pos.items():
        neighbours = set(g.successors(nid)) | set(g.predecessors(nid))
        lean = sum(pos[m][1] - y for m in neighbours if m in pos)
        # lean < 0: edges go downwards, so put the label above, and vice versa.
        placed[nid] = (x, y + dy) if lean < -1e-9 else (x, y - dy)
    return placed

def _normalise_pos(
    pos: Dict[str, Tuple[float, float]],
    margin: float = 0.08,
) -> Dict[str, Tuple[float, float]]:
    """Rescale node positions to fill [margin, 1-margin]² so the graph
    uses the full axes area without large blank regions."""
    if len(pos) < 2:
        return pos
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0
    scale = 1.0 - 2 * margin
    return {
        nid: (
            margin + (x - x_min) / x_span * scale,
            margin + (y - y_min) / y_span * scale,
        )
        for nid, (x, y) in pos.items()
    }
