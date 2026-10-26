"""SDF builders for the twin's worlds.

Fuel furniture and people fitted to footprints, walls, doors, floors, the
goal flag, floor halos and dashed routes, and a camera whose projection is
the camera sensor's. Objects are duck-typed; see
:class:`pace.twin.plant_world.Look`.
"""

from __future__ import annotations

import math
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .assets import A

# colours (hex) of the markers the director puts into the world
RED = "#d62728"          # a cause of the failure / deviation
GREEN = "#2ca02c"        # the path once the causes are resolved
BLUE = "#1f77b4"         # the original path
WHITE = "#f4f4f2"
INK = "#202022"


def wrap(model_xml: str) -> str:
    return f'<?xml version="1.0"?><sdf version="1.6">{model_xml}</sdf>'


# ---------------------------------------------------------------- transforms

def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_rpy(R: np.ndarray) -> tuple[float, float, float]:
    p = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    if abs(math.cos(p)) > 1e-6:
        return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])
    return 0.0, p, math.atan2(-R[0, 1], R[1, 1])


def pose_matrix(x=0.0, y=0.0, z=0.0, r=0.0, p=0.0, yaw=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(r, p, yaw)
    T[:3, 3] = (x, y, z)
    return T


def parse_pose(text: str | None) -> np.ndarray:
    if not text or not text.strip():
        return np.eye(4)
    v = [float(t) for t in text.split()]
    return pose_matrix(*v)


def pose_text(T: np.ndarray) -> str:
    r, p, y = matrix_rpy(T[:3, :3])
    x, y_, z = T[:3, 3]
    return f"{x:.5f} {y_:.5f} {z:.5f} {r:.5f} {p:.5f} {y:.5f}"


# --------------------------------------------------------------- mesh extents

def _obj_vertices(path: Path) -> np.ndarray:
    pts = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                pts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(pts, dtype=float)


def _dae_vertices(path: Path) -> np.ndarray:
    root = ET.parse(path).getroot()
    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    q = (lambda t: f"{{{ns}}}{t}") if ns else (lambda t: t)
    unit = root.find(f"./{q('asset')}/{q('unit')}")
    scale = float(unit.get("meter", "1")) if unit is not None else 1.0
    up = root.find(f"./{q('asset')}/{q('up_axis')}")
    up = (up.text or "Z_UP").strip() if up is not None else "Z_UP"
    geoms = {}
    for g in root.iter(q("geometry")):
        mesh = g.find(q("mesh"))
        if mesh is None:
            continue
        sources = {s.get("id"): s for s in mesh.findall(q("source"))}
        src_id = None
        verts = mesh.find(q("vertices"))
        if verts is not None:
            for inp in verts.findall(q("input")):
                if inp.get("semantic") == "POSITION":
                    src_id = (inp.get("source") or "").lstrip("#")
        src = sources.get(src_id) if src_id else None
        if src is None and sources:
            src = next(iter(sources.values()))
        if src is None:
            continue
        arr = src.find(q("float_array"))
        if arr is None or not arr.text:
            continue
        acc = src.find(f"./{q('technique_common')}/{q('accessor')}")
        stride = int(acc.get("stride", "3")) if acc is not None else 3
        v = np.fromstring(arr.text, sep=" ")
        if stride < 3 or len(v) < stride:
            continue
        geoms[g.get("id")] = v[: len(v) - len(v) % stride].reshape(-1, stride)[:, :3]

    def node_T(node) -> np.ndarray:
        T = np.eye(4)
        for ch in node:
            tag = ch.tag.split("}")[-1]
            if tag == "matrix":
                T = T @ np.fromstring(ch.text, sep=" ").reshape(4, 4)
            elif tag == "translate":
                M = np.eye(4); M[:3, 3] = np.fromstring(ch.text, sep=" ")[:3]; T = T @ M
            elif tag == "rotate":
                x, y, z, a = np.fromstring(ch.text, sep=" ")[:4]
                ax = np.array([x, y, z]); n = np.linalg.norm(ax)
                if n > 0 and a:
                    ax /= n; a = math.radians(a)
                    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                    M = np.eye(4); M[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
                    T = T @ M
            elif tag == "scale":
                M = np.diag(list(np.fromstring(ch.text, sep=" ")[:3]) + [1.0]); T = T @ M
        return T

    lib_nodes = {n.get("id"): n for n in root.iter(q("node")) if n.get("id")}
    out = []

    def walk(node, T, depth=0):
        if depth > 32:
            return
        T = T @ node_T(node)
        for ig in node.findall(q("instance_geometry")):
            gid = (ig.get("url") or "").lstrip("#")
            if gid in geoms:
                out.append((T[:3, :3] @ geoms[gid].T).T + T[:3, 3])
        for inn in node.findall(q("instance_node")):
            ref = lib_nodes.get((inn.get("url") or "").lstrip("#"))
            if ref is not None:
                walk(ref, T, depth + 1)
        for child in node.findall(q("node")):
            walk(child, T, depth + 1)

    scene = root.find(f"./{q('library_visual_scenes')}/{q('visual_scene')}")
    if scene is not None:
        for node in scene.findall(q("node")):
            walk(node, np.eye(4))
    if not out:
        out = list(geoms.values())
    if not out:
        return np.zeros((0, 3))
    P = np.concatenate(out) * scale
    if up == "Y_UP":
        P = np.stack([P[:, 0], -P[:, 2], P[:, 1]], axis=1)
    elif up == "X_UP":
        P = np.stack([-P[:, 2], P[:, 1], P[:, 0]], axis=1)
    return P


def _stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        pts = re.findall(rb"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", data)
        return np.asarray([[float(a), float(b), float(c)] for a, b, c in pts])
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(float)


def mesh_vertices(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".obj":
        return _obj_vertices(path)
    if suf == ".dae":
        return _dae_vertices(path)
    if suf == ".stl":
        return _stl_vertices(path)
    raise ValueError(f"unsupported mesh {path}")


def resolve_model_dir(name: str) -> Path | None:
    """The directory a ``model://name`` refers to, aliases included."""
    for n in (name, A.ALIASES.get(name, name)):
        for base in (A.FUEL_DIR, A.PAL_DIR):
            if (base / n).is_dir():
                return base / n
    return None


def resolve_uri(uri: str, model_dir: Path) -> Path:
    uri = uri.strip()
    if uri.startswith("model://"):
        rest = uri[len("model://"):]
        name, _, sub = rest.partition("/")
        d = resolve_model_dir(name)
        if d is not None and (d / sub).exists():
            return d / sub
        return model_dir / sub
    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    cand = model_dir / uri
    if cand.exists():
        return cand
    hits = list(model_dir.rglob(Path(uri).name))       # GazBo's WhiteBoard names meshes/Help.stl for ./Help.stl
    return hits[0] if hits else cand


@dataclass
class Visual:
    xml: ET.Element
    pose: np.ndarray
    kind: str
    scale: np.ndarray
    size: np.ndarray
    mesh: Path | None = None


_VISUALS: dict[Path, list[Visual]] = {}
_VERTS: dict[Path, np.ndarray] = {}
_EXTENT: dict[Path, tuple[np.ndarray, np.ndarray]] = {}


def load_visuals(model_dir: Path) -> list[Visual]:
    if model_dir in _VISUALS:
        return _VISUALS[model_dir]
    sdf = model_dir / "model.sdf"
    if not sdf.exists():
        cfg = ET.parse(model_dir / "model.config").getroot()
        sdf = model_dir / (cfg.findtext("sdf") or "model.sdf").strip()
    root = ET.parse(sdf).getroot()
    model = root if root.tag == "model" else root.find("model")
    if model is None:
        raise ValueError(f"no <model> in {sdf}")
    out = []
    for link in model.findall("link"):
        T_link = parse_pose(link.findtext("pose"))
        for vis in link.findall("visual"):
            T_vis = T_link @ parse_pose(vis.findtext("pose"))
            geom = vis.find("geometry")
            if geom is None:
                continue
            if geom.find("mesh") is not None:
                m = geom.find("mesh")
                scale = np.array([float(t) for t in (m.findtext("scale") or "1 1 1").split()])
                if len(scale) == 1:
                    scale = np.repeat(scale, 3)
                mesh = resolve_uri(m.findtext("uri"), model_dir)
                if not mesh.exists():
                    continue
                out.append(Visual(vis, T_vis, "mesh", scale, np.zeros(3), mesh))
            elif geom.find("box") is not None:
                size = np.array([float(t) for t in geom.find("box").findtext("size").split()])
                out.append(Visual(vis, T_vis, "box", np.ones(3), size))
            elif geom.find("cylinder") is not None:
                c = geom.find("cylinder")
                r, L = float(c.findtext("radius")), float(c.findtext("length"))
                out.append(Visual(vis, T_vis, "cylinder", np.ones(3), np.array([r, r, L])))
            elif geom.find("sphere") is not None:
                r = float(geom.find("sphere").findtext("radius"))
                out.append(Visual(vis, T_vis, "sphere", np.ones(3), np.array([r, r, r])))
    _VISUALS[model_dir] = out
    return out


def _points(v: Visual) -> np.ndarray:
    if v.kind == "mesh":
        if v.mesh not in _VERTS:
            _VERTS[v.mesh] = mesh_vertices(v.mesh)
        P = _VERTS[v.mesh] * v.scale
    elif v.kind == "box":
        h = v.size / 2
        P = np.array([(sx * h[0], sy * h[1], sz * h[2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    else:
        r, _, L = v.size if v.kind == "cylinder" else (v.size[0], 0, 2 * v.size[0])
        P = np.array([(sx * r, sy * r, sz * L / 2) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    if len(P) == 0:
        return P
    return (v.pose[:3, :3] @ P.T).T + v.pose[:3, 3]


def model_extent(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if model_dir in _EXTENT:
        return _EXTENT[model_dir]
    pts = [p for p in (_points(v) for v in load_visuals(model_dir)) if len(p)]
    P = np.concatenate(pts)
    lo, hi = np.percentile(P, 0.5, axis=0), np.percentile(P, 99.5, axis=0)
    _EXTENT[model_dir] = (lo, hi)
    return lo, hi


# ------------------------------------------------------------ SDF pieces

def hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def material(hex_colour: str, emissive: float = 0.0, transparency: float = 0.0) -> str:
    r, g, b = hex_rgb(hex_colour)
    em = f"<emissive>{r * emissive:.3f} {g * emissive:.3f} {b * emissive:.3f} 1</emissive>" if emissive else ""
    tr = f"<transparency>{transparency:.2f}</transparency>" if transparency else ""
    return (f"<material><ambient>{r:.3f} {g:.3f} {b:.3f} 1</ambient><diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse>"
            f"<specular>0.1 0.1 0.1 1</specular>{em}</material>{tr}")


def box_visual(name: str, pose: str, size, mat: str) -> str:
    sx, sy, sz = size
    return (f'<visual name="{name}"><pose>{pose}</pose><geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size>'
            f"</box></geometry>{mat}</visual>")


def cylinder_visual(name: str, pose: str, r: float, L: float, mat: str) -> str:
    return (f'<visual name="{name}"><pose>{pose}</pose><geometry><cylinder><radius>{r:.4f}</radius>'
            f"<length>{L:.4f}</length></cylinder></geometry>{mat}</visual>")


def static_model(name: str, pose: str, body: str) -> str:
    return f'<model name="{name}"><static>true</static><pose>{pose}</pose><link name="link">{body}</link></model>'


def _axis_perm(R: np.ndarray) -> np.ndarray | None:
    Aa = np.abs(R)
    if np.allclose(Aa.max(axis=0), 1, atol=1e-3) and np.allclose(Aa.sum(axis=0), 1, atol=1e-3):
        return Aa.T
    return None


def _visual_xml(v: Visual, name: str, T_extra: np.ndarray, scale_world: np.ndarray) -> str:
    T = (T_extra @ v.pose).copy()
    T[:3, 3] = T[:3, 3] * scale_world
    perm = _axis_perm(T[:3, :3])
    s_mesh = perm @ scale_world if perm is not None else np.repeat(scale_world.min(), 3)
    mat = v.xml.find("material")
    mat_xml = _absolute_uris(ET.tostring(mat, encoding="unicode")) if mat is not None else ""
    tr = v.xml.find("transparency")
    tr_xml = ET.tostring(tr, encoding="unicode") if tr is not None else ""
    if v.kind == "mesh":
        sc = v.scale * s_mesh
        g = (f"<mesh><uri>file://{v.mesh}</uri><scale>{sc[0]:.5f} {sc[1]:.5f} {sc[2]:.5f}</scale></mesh>")
    elif v.kind == "box":
        sz = v.size * s_mesh
        g = f"<box><size>{sz[0]:.4f} {sz[1]:.4f} {sz[2]:.4f}</size></box>"
    elif v.kind == "cylinder":
        r = v.size[0] * float(np.sqrt(s_mesh[0] * s_mesh[1]))
        g = f"<cylinder><radius>{r:.4f}</radius><length>{v.size[2] * s_mesh[2]:.4f}</length></cylinder>"
    else:
        g = f"<sphere><radius>{v.size[0] * float(np.mean(s_mesh)):.4f}</radius></sphere>"
    return f'<visual name="{name}"><pose>{pose_text(T)}</pose><geometry>{g}</geometry>{mat_xml}{tr_xml}</visual>'


def _absolute_uris(xml: str) -> str:
    """``model://name/...`` inside a copied material (texture scripts) as absolute
    ``file://`` paths, so Gazebo needs no model-path lookup and no alias."""
    def sub(m):
        d = resolve_model_dir(m.group(1))
        return f"file://{d}/" if d is not None else m.group(0)
    return re.sub(r"model://([^/<]+)/", sub, xml)


def fuel_visuals(key: str, dx: float, dy: float, height: float | None = None) -> tuple[str, float]:
    """The visuals of a Fuel model fitted to a dx x dy footprint, as link XML
    posed at the footprint's centre on the floor.  Returns (xml, drawn height)."""
    d = A.fuel_dir(key)
    vis = load_visuals(d)
    lo, hi = model_extent(d)
    ext = hi - lo
    stretch = key in A.STRETCH
    yaw_model = 0.0
    if stretch and (dx >= dy) != (ext[0] >= ext[1]) and abs(dx - dy) > 0.05:
        yaw_model = math.pi / 2
    R0 = rpy_matrix(0, 0, yaw_model)
    ext_r = np.abs(R0 @ ext)
    if stretch:
        s = np.array([dx / ext_r[0], dy / ext_r[1], 1.0])
        s[:2] = np.clip(s[:2], 0.2, 4.0)
        if height is not None:
            s[2] = height / ext[2]
    else:
        u = min(dx / ext_r[0], dy / ext_r[1])
        if height is not None:
            u = height / ext[2]
        s = np.array([u, u, u])
    centre = R0 @ ((lo + hi) / 2)
    off = -centre * np.array([s[0], s[1], 1.0])
    off[2] = -lo[2] * s[2]
    T_extra = np.eye(4); T_extra[:3, :3] = R0
    parts = [_visual_xml(v, f"v{i}", T_extra, s) for i, v in enumerate(vis)]
    T_off = np.eye(4); T_off[:3, 3] = off
    xml = f'<link name="look"><pose>{pose_text(T_off)}</pose>{"".join(parts)}</link>'
    return xml, float(ext[2] * s[2])


def hull_link(dx: float, dy: float, h: float, cylinder: bool = False) -> str:
    if cylinder:
        geom = f"<cylinder><radius>{max(dx, dy) / 2:.3f}</radius><length>{h:.3f}</length></cylinder>"
    else:
        geom = f"<box><size>{dx:.3f} {dy:.3f} {h:.3f}</size></box>"
    return (f'<link name="hull"><pose>0 0 {h / 2:.3f} 0 0 0</pose>'
            f'<collision name="c"><geometry>{geom}</geometry></collision></link>')


def _pose(o: "Look", yaw: float | None = None) -> str:
    return f"{o.cx:.4f} {o.cy:.4f} 0 0 0 {o.yaw if yaw is None else yaw:.4f}"


def object_model(o: "Look") -> str:
    """The Gazebo model of one ontology object, posed where the ontology puts it."""
    if o.label == "wall":
        return wall_model(o)
    if o.openability and o.hinge is not None:
        return door_model(o)
    key = o.model or o.label
    if key in PROCEDURAL:
        look = f'<link name="look">{PROCEDURAL[key](o)}</link>'
        return f'<model name="{o.gazebo_label}"><static>true</static><pose>{_pose(o)}</pose>{look}{hull_link(o.dx, o.dy, o.height)}</model>'
    if key in A.FUEL:
        h = o.height if (o.is_human or key in A.STRETCH) else None
        look, drawn_h = fuel_visuals(key, o.dx, o.dy, h)
        facing = A.FACING.get(key, 0.0)
        hull = hull_link(o.dx, o.dy, o.height, cylinder=o.is_human)
        # the hull must not turn with the look: give it the inverse yaw of the facing correction
        yaw = o.yaw - facing
        if facing:
            hull = hull.replace('<link name="hull"><pose>0 0', f'<link name="hull"><pose>0 0').replace(
                f"{o.height / 2:.3f} 0 0 0</pose>", f"{o.height / 2:.3f} 0 0 {facing:.4f}</pose>")
        return f'<model name="{o.gazebo_label}"><static>true</static><pose>{_pose(o, yaw)}</pose>{look}{hull}</model>'
    return generic_model(o)


def board_visuals(o: "Look") -> str:
    """A whiteboard on two posts, its face across the shorter footprint side."""
    along_y = o.dy >= o.dx
    parts = []
    for s_ in (-1, 1):
        x, y = (0, s_ * (o.dy / 2 - 0.05)) if along_y else (s_ * (o.dx / 2 - 0.05), 0)
        parts.append(cylinder_visual(f"post{s_ + 1}", f"{x:.3f} {y:.3f} 0.95 0 0 0", 0.025, 1.9, material("#4a4b4d")))
        parts.append(box_visual(f"foot{s_ + 1}", f"{x:.3f} {y:.3f} 0.02 0 0 0",
                                (0.06, 0.5, 0.04) if along_y else (0.5, 0.06, 0.04), material("#4a4b4d")))
    size = (0.04, o.dy, 1.0) if along_y else (o.dx, 0.04, 1.0)
    parts.append(box_visual("board", "0 0 1.4 0 0 0", size, material("#f2f2ee")))
    parts.append(box_visual("frame", "0 0 1.4 0 0 0", (size[0] + 0.02, size[1] + 0.04, 1.06) if along_y
                            else (size[0] + 0.04, size[1] + 0.02, 1.06), material("#9a9a98")))
    return "".join(parts)


def counter_visuals(o: "Look") -> str:
    """A reception or library counter: a wooden body with a lighter top slab."""
    h = o.height
    return "".join([box_visual("body", f"0 0 {(h - 0.04) / 2:.4f} 0 0 0", (o.dx - 0.06, o.dy - 0.06, h - 0.04),
                               material("#8e7a66")),
                    box_visual("top", f"0 0 {h - 0.02:.4f} 0 0 0", (o.dx, o.dy, 0.04), material("#c9b697")),
                    box_visual("kick", "0 0 0.05 0 0 0", (o.dx - 0.16, o.dy - 0.16, 0.1), material("#4a4540"))])


def bed_visuals(o: "Look") -> str:
    """A hospital bed: frame, mattress, pillow, a rail on each long side."""
    L, W = max(o.dx, o.dy), min(o.dx, o.dy)
    along_x = o.dx >= o.dy
    def sz(a, b, c):
        return (a, b, c) if along_x else (b, a, c)
    parts = [box_visual("frame", "0 0 0.25 0 0 0", sz(L, W, 0.3), material("#b8bcc2")),
             box_visual("mattress", "0 0 0.48 0 0 0", sz(L - 0.1, W - 0.1, 0.16), material("#e9edf2")),
             box_visual("blanket", f"{-0.15 if along_x else 0:.2f} {0 if along_x else -0.15:.2f} 0.57 0 0 0",
                        sz(L * 0.6, W - 0.12, 0.04), material("#7fa7c9")),
             box_visual("pillow", f"{L / 2 - 0.3 if along_x else 0:.2f} {0 if along_x else L / 2 - 0.3:.2f} 0.6 0 0 0",
                        sz(0.4, W - 0.3, 0.1), material("#ffffff"))]
    for s_ in (-1, 1):
        x, y = (0, s_ * (W / 2 - 0.02)) if along_x else (s_ * (W / 2 - 0.02), 0)
        parts.append(box_visual(f"rail{s_ + 1}", f"{x:.3f} {y:.3f} 0.75 0 0 0", sz(L * 0.6, 0.03, 0.3), material("#b8bcc2")))
    return "".join(parts)


def cart_visuals(o: "Look") -> str:
    """A service trolley: four posts on castors, two shelves with books, a push handle.
    (Built here because the Fuel Utility Cart is a rover.)"""
    h = o.height
    hx, hy = o.dx / 2 - 0.03, o.dy / 2 - 0.03
    metal, shelf = material("#9aa3ad"), material("#5f7f9f")
    parts = []
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        parts.append(cylinder_visual(f"post{i}", f"{sx * hx:.3f} {sy * hy:.3f} {(h - 0.05) / 2 + 0.07:.3f} 0 0 0",
                                     0.015, h - 0.12, metal))
        parts.append(cylinder_visual(f"wheel{i}", f"{sx * hx:.3f} {sy * hy:.3f} 0.04 1.5708 0 0", 0.04, 0.03,
                                     material("#26282b")))
    for k, z in enumerate((0.22, h - 0.2)):
        parts.append(box_visual(f"shelf{k}", f"0 0 {z:.3f} 0 0 0", (o.dx, o.dy, 0.025), shelf))
    along_y = o.dy >= o.dx
    handle = (f"0 {hy + 0.06:.3f} {h:.3f} 0 1.5708 0" if along_y else f"{hx + 0.06:.3f} 0 {h:.3f} 1.5708 0 0")
    parts.append(cylinder_visual("handle", handle, 0.015, min(o.dx, o.dy), metal))
    colours = ("#8e2c2c", "#2c5d8e", "#c7a13a", "#3f7d4a", "#6b4a8e")
    n = 5
    for k in range(n):                                # a row of books on the top shelf
        L = (o.dy if along_y else o.dx) - 0.1
        off = -L / 2 + (k + 0.5) * L / n
        x, y = (0.0, off) if along_y else (off, 0.0)
        parts.append(box_visual(f"book{k}", f"{x:.3f} {y:.3f} {h - 0.2 + 0.12:.3f} 0 0 0",
                                (min(o.dx, o.dy) * 0.6, L / n * 0.8, 0.22) if along_y else
                                (L / n * 0.8, min(o.dx, o.dy) * 0.6, 0.22), material(colours[k % len(colours)])))
    return "".join(parts)


PROCEDURAL = {"whiteboard": board_visuals, "counter": counter_visuals, "bed": bed_visuals, "cart": cart_visuals}


def generic_model(o: "Look", colour: str = "#a39d94") -> str:
    body = box_visual("b", f"0 0 {o.height / 2:.3f} 0 0 0", (o.dx, o.dy, o.height), material(colour))
    return (f'<model name="{o.gazebo_label}"><static>true</static><pose>{_pose(o)}</pose>'
            f'<link name="look">{body}</link>{hull_link(o.dx, o.dy, o.height)}</model>')


WALL_H = 1.3          # partitions: the camera sees over them, the laser (at 0.1 m) sees them


def wall_model(o: "Look", height: float | None = None) -> str:
    h = height or o.height or WALL_H
    body = box_visual("w", f"0 0 {h / 2:.4f} 0 0 0", (o.dx, o.dy, h - 0.03), material("#dcd8cf"))
    cap = box_visual("cap", f"0 0 {h - 0.015:.4f} 0 0 0", (o.dx + 0.01, o.dy + 0.01, 0.03), material("#b9b4a9"))
    return (f'<model name="{o.gazebo_label}"><static>true</static><pose>{_pose(o)}</pose>'
            f'<link name="look">{body}{cap}</link>{hull_link(o.dx, o.dy, h)}</model>')


def door_model(o: "Look", height: float = 2.0) -> str:
    """A door leaf on the floor with a handle; the model's pose is the leaf's,
    so opening it is teleporting the model to :meth:`"Look".open_pose`."""
    t = min(o.dx, o.dy)
    w = max(o.dx, o.dy)
    along_y = o.dy >= o.dx
    leaf = box_visual("leaf", f"0 0 {height / 2:.3f} 0 0 0", (o.dx, o.dy, height), material("#8b5a2b"))
    hx, hy = (t / 2 + 0.03, 0.0) if along_y else (0.0, t / 2 + 0.03)
    # the handle opposite the hinge
    sx, sy = o.hinge or (-1, 1)
    ox, oy = (0.0, -sy * (w / 2 - 0.12)) if along_y else (-sx * (w / 2 - 0.12), 0.0)
    handle = cylinder_visual("handle", f"{ox + hx:.3f} {oy + hy:.3f} 1.0 0 {1.5708 if along_y else 0.0} "
                             f"{0.0 if along_y else 1.5708}", 0.012, 0.12, material("#c0c0c0"))
    pose_xy = (o.open_pose() if o.is_open else (o.cx, o.cy, o.yaw))
    pose = f"{pose_xy[0]:.4f} {pose_xy[1]:.4f} 0 0 0 {pose_xy[2]:.4f}"
    return (f'<model name="{o.gazebo_label}"><static>true</static><pose>{pose}</pose>'
            f'<link name="look">{leaf}{handle}</link>{hull_link(o.dx, o.dy, height)}</model>')


def floor_model(X0: float, Y0: float, X1: float, Y1: float) -> str:
    """A light floor slab over the room and a collision plane to drive on."""
    cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
    slab = box_visual("slab", f"{cx:.3f} {cy:.3f} -0.01 0 0 0", (X1 - X0, Y1 - Y0, 0.02), material("#d9cbb0"))
    return ('<model name="ground"><static>true</static><link name="link"><collision name="c">'
            '<geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>'
            '<surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface></collision>'
            f"{slab}</link></model>")


def goal_model(x: float, y: float, name: str = "goal_flag") -> str:
    parts = [cylinder_visual("disc", "0 0 0.006 0 0 0", 0.33, 0.012, material("#8f8a80")),
             cylinder_visual("pole", "0 0 0.55 0 0 0", 0.02, 1.1, material(INK)),
             box_visual("flag", "0.17 0 0.95 0 0 0", (0.34, 0.012, 0.2), material(RED))]
    return static_model(name, f"{x:.4f} {y:.4f} 0 0 0 0", "".join(parts))


def halo_model(name: str, o: "Look", colour: str = RED, pad: float = 0.08, width: float = 0.05) -> str:
    """A flat frame on the floor round an object's footprint: the visual
    explanation's red, in the world."""
    F = o.footprint()
    if F is None:
        return ""
    cx, cy = F.mean(axis=0)
    dx, dy = o.dx, o.dy
    if o.openability and o.is_open:
        dx, dy = (o.dy, o.dx)
    ox, oy = dx / 2 + pad, dy / 2 + pad
    mat = material(colour, emissive=0.7)
    parts = [box_visual("n", f"0 {oy:.3f} 0.006 0 0 0", (2 * ox + width, width, 0.012), mat),
             box_visual("s", f"0 {-oy:.3f} 0.006 0 0 0", (2 * ox + width, width, 0.012), mat),
             box_visual("e", f"{ox:.3f} 0 0.006 0 0 0", (width, 2 * oy + width, 0.012), mat),
             box_visual("w", f"{-ox:.3f} 0 0.006 0 0 0", (width, 2 * oy + width, 0.012), mat)]
    yaw = o.yaw
    if o.openability and o.is_open:
        yaw = o.open_pose()[2] + math.pi / 2
    return static_model(name, f"{cx:.4f} {cy:.4f} 0 0 0 {yaw:.4f}", "".join(parts))


def beacon_model(name: str, o: "Look", colour: str = RED) -> str:
    """A slim post over an object with a glowing cap: readable from the camera."""
    F = o.footprint()
    if F is None:
        return ""
    cx, cy = F.mean(axis=0)
    h = max(o.height, 0.5) + 0.35
    parts = [cylinder_visual("post", f"0 0 {h / 2:.3f} 0 0 0", 0.012, h, material(colour, emissive=0.5)),
             cylinder_visual("cap", f"0 0 {h + 0.05:.3f} 0 0 0", 0.09, 0.1, material(colour, emissive=0.9))]
    return static_model(name, f"{cx:.4f} {cy:.4f} 0 0 0 0", "".join(parts))


def route_model(path: np.ndarray, name: str = "route", colour: str = GREEN, period: float = 0.5,
                dash: float = 0.3, width: float = 0.07) -> str:
    """A path as dashes on the floor."""
    P = np.asarray(path, dtype=float)
    if len(P) < 2:
        return ""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    parts = []
    s = 0.1
    k = 0
    while s + dash < cum[-1] and k < 400:
        i = int(np.searchsorted(cum, s) - 1)
        i = max(0, min(i, len(P) - 2))
        t = (s + dash / 2 - cum[i]) / max(seg[i], 1e-9)
        x, y = P[i] + t * (P[i + 1] - P[i])
        yaw = math.atan2(P[i + 1][1] - P[i][1], P[i + 1][0] - P[i][0])
        parts.append(box_visual(f"u{k}", f"{x:.4f} {y:.4f} 0.006 0 0 {yaw:.4f}", (dash + 0.06, width + 0.05, 0.006),
                                material(WHITE, emissive=0.4)))
        parts.append(box_visual(f"d{k}", f"{x:.4f} {y:.4f} 0.010 0 0 {yaw:.4f}", (dash, width, 0.008),
                                material(colour, emissive=0.5)))
        s += period
        k += 1
    return static_model(name, "0 0 0 0 0 0", "".join(parts))


def ring_model(name: str, cx: float, cy: float, radius: float = 0.5, n: int = 24, width: float = 0.05,
               colour: str = INK) -> str:
    parts = []
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 0.5) / n
        am = (a0 + a1) / 2
        x, y = cx + radius * math.cos(am), cy + radius * math.sin(am)
        parts.append(box_visual(f"d{k}", f"{x:.4f} {y:.4f} 0.009 0 0 {am + math.pi / 2:.4f}",
                                (radius * (a1 - a0), width, 0.006), material(colour if k % 2 else WHITE, emissive=0.4)))
    return static_model(name, "0 0 0 0 0 0", "".join(parts))


# ------------------------------------------------------------------ camera

@dataclass
class Camera:
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    width: int = 1920
    height: int = 1080
    hfov: float = math.radians(60.0)

    @property
    def fx(self) -> float:
        return (self.width / 2) / math.tan(self.hfov / 2)

    def project(self, P) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, dtype=float))
        R = rpy_matrix(0.0, self.pitch, self.yaw)
        pc = (R.T @ (P - np.array([self.x, self.y, self.z])).T).T
        u = self.width / 2 - self.fx * pc[:, 1] / pc[:, 0]
        v = self.height / 2 - self.fx * pc[:, 2] / pc[:, 0]
        return np.stack([u, v], axis=1)

    def sdf(self, frames_dir: Path, rate: float = 25.0, name: str = "film_camera") -> str:
        return (f'<model name="{name}"><static>true</static><pose>{self.x:.4f} {self.y:.4f} {self.z:.4f} '
                f'0 {self.pitch:.5f} {self.yaw:.5f}</pose><link name="link"><sensor name="cam" type="camera">'
                f"<camera><horizontal_fov>{self.hfov:.5f}</horizontal_fov><image><width>{self.width}</width>"
                f"<height>{self.height}</height><format>R8G8B8</format></image><clip><near>0.05</near><far>200</far></clip>"
                f'<save enabled="true"><path>{frames_dir}</path></save></camera>'
                f"<always_on>1</always_on><update_rate>{rate}</update_rate><visualize>false</visualize>"
                f"</sensor></link></model>")


def fit_camera(bounds: tuple[float, float, float, float], width: int, height: int, yaw: float,
               pitch: float = math.radians(45), top_m: float = 1.4, margin: float = 0.03,
               hfov: float = math.radians(60.0), reserve_bottom: float = 0.0) -> Camera:
    """A camera looking along ``yaw`` and down by ``pitch``, pulled back until
    the whole room (its floor corners and their tops at ``top_m``) fits the
    band of the frame above the bottom strip reserved for captions.  The
    camera is tilted a little further down so that the room sits in the
    middle of that band rather than in the middle of the frame."""
    X0, Y0, X1, Y1 = bounds
    corners = np.array([(x, y, z) for x in (X0, X1) for y in (Y0, Y1) for z in (0.0, top_m)])
    fx = (width / 2) / math.tan(hfov / 2)
    tilt = math.atan((reserve_bottom * height / 2) / fx)
    d = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
    target = np.array([(X0 + X1) / 2, (Y0 + Y1) / 2, 0.4])
    v_lo, v_hi = margin * height, (1 - reserve_bottom) * height - margin * height

    def make(dist: float) -> Camera:
        pos = target - dist * d
        return Camera(*pos, pitch + tilt, yaw, width, height, hfov)

    def ok(dist: float) -> bool:
        uv = make(dist).project(corners)
        return bool((uv[:, 0] > margin * width).all() and (uv[:, 0] < (1 - margin) * width).all()
                    and (uv[:, 1] > v_lo).all() and (uv[:, 1] < v_hi).all())

    lo, hi = 1.0, 80.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return make(hi)


# ------------------------------------------------------------------- world

def world_xml(name: str, models: list[str], rate: float = 250.0, gui_camera: str = "") -> str:
    """A world with lights, physics at ``rate`` updates per wall second (250 =
    real time at a 4 ms step), and the gazebo_ros_state plugin the director calls."""
    return f"""<?xml version="1.0"?>
<!-- generated by pace.twin.sdf -->
<sdf version="1.6">
  <world name="{name}">
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.93 0.93 0.93 1</background>
      <shadows>true</shadows><grid>false</grid><origin_visual>false</origin_visual></scene>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 12 0 0 0</pose>
      <diffuse>0.85 0.85 0.85 1</diffuse><specular>0.1 0.1 0.1 1</specular><direction>-0.3 0.2 -0.9</direction></light>
    <light type="directional" name="fill"><cast_shadows>false</cast_shadows><pose>0 0 12 0 0 0</pose>
      <diffuse>0.4 0.4 0.4 1</diffuse><specular>0 0 0 1</specular><direction>0.4 -0.3 -0.7</direction></light>
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>{rate / 250:.3f}</real_time_factor>
      <real_time_update_rate>{rate:.0f}</real_time_update_rate>
    </physics>
    {gui_camera}
{''.join(models)}
  </world>
</sdf>
"""
