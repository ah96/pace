"""The fleet, rendered in the passages that constrain it: the paper's Figure 2.

Each panel puts one robot at the mouth of one passage of the plant and looks
along the passage from behind the robot, so that body and opening are seen
together: the MiR100 in the very-narrow aisle, the forklift at the aisle
narrowed by staged pallets and at the underpass (before and after PACE's
repair), the Husky on the dock ramp -- and TIAGo pushing the fire door open,
its arm on the push bar and one leaf swung part-way, the whole doorway in the
frame.  Titles are bold; the caption carries the numbers.

    PYTHONPATH=. DISPLAY=:0 python experiments/make_fleet_figure.py --out build/figures
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pace.twin import plant_world as PW  # noqa: E402
from pace.twin import render as R  # noqa: E402
from pace.twin import robots as RB  # noqa: E402
from pace.twin import sdf as S  # noqa: E402
from scenarios.plant import ROBOTS, plant_spec  # noqa: E402

#: the renders, before they are put together
RAW_DIR = Path("build/fleet_panels")

# (panel key, robot, segment, the end it is entered from (zone id), title, changes applied first)
PANELS = [
    ("vna_tote", "tote_amr", "seg_wash_store", "wash", "Tote carrier, 1.5 m aisle: fits", ()),
    ("staged_fork", "forklift_agv", "seg_disA_disB", "disassembly_A",
     "Forklift, staged goods: blocked", ()),
    ("underpass_fork", "forklift_agv", "seg_store_ship", "module_store",
     "Forklift, underpass: blocked", ()),
    ("firedoor_tiago", "mobile_manipulator", "seg_ship_sort", "sorting", "TIAGo pushes the fire door open", ()),
    ("ramp_husky", "yard_ugv", "seg_dock_yard", "dock", "Yard UGV, 7° ramp: climbs it", ()),
    ("underpass_after", "forklift_agv", "seg_store_ship", "module_store",
     "Underpass repaired: passes",
     (("seg_store_ship", "width", 1.8), ("seg_store_ship", "height", 2.3))),
]
#: square panels at a 50 deg field of view: the robots large, the passage's full height in view
W_PX, H_PX = 1200, 1200
FOV_DEG = 50.0


def panel(spec, key, robot, seg_id, from_zone, out: Path, width=W_PX, height=H_PX) -> None:
    cors = PW.corridors(spec)
    c = cors[seg_id]
    z = spec.zone(from_zone)
    into = 1.0 if (c.axis == "x" and z.x < c.rect.cx) or (c.axis == "y" and z.y < c.rect.cy) else -1.0
    t_mouth = 0.0 if into > 0 else 1.0
    mx, my = c.along(t_mouth)
    yaw = (0.0 if into > 0 else math.pi) if c.axis == "x" else (math.pi / 2 if into > 0 else -math.pi / 2)
    L = float(ROBOTS[robot].get("length", 1.0))
    ax_x = 1 if c.axis == "x" else 0
    ax_y = 1 - ax_x
    # the robot's front just at the mouth
    back = 0.3 + L / 2
    rx = mx - into * back * ax_x
    ry = my - into * back * ax_y
    level = z.level
    extra = [RB.robot_static(robot, ROBOTS[robot], rx, ry, yaw, level, label="r")]
    # the camera: behind the robot's tail, off to one side, low and close, looking at the mouth
    dist = back + L / 2 + 1.7 + 0.4 * L
    side = 1.6 + 0.3 * L
    cx = mx - into * dist * ax_x + side * ax_y
    cy = my - into * dist * ax_y - side * ax_x
    z_cam = 2.3 + level + 0.35 * L
    # look at a point just inside the mouth of the passage at mid-height
    tx, ty, tz = mx + into * 0.8 * ax_x, my + into * 0.8 * ax_y, level + 1.0
    cyaw = math.atan2(ty - cy, tx - cx)
    cpitch = math.atan2(z_cam - tz, math.hypot(tx - cx, ty - cy))
    cam = S.Camera(cx, cy, z_cam, cpitch, cyaw, width, height, math.radians(FOV_DEG))
    R.render(lambda fdir: PW.build_world(spec, extra_models=extra, camera_xml=cam.sdf(fdir, rate=4.0)).world,
             out=out, frames=5, timeout=300)


def door_panel(spec, seg_id: str, from_zone: str, out: Path, swing_deg: float = 40.0,
               width=W_PX, height=H_PX) -> None:
    """TIAGo pushing a closed fire door open, as a posed still: the leaf on its
    left swung ``swing_deg`` away from it, the gripper on that leaf's push bar
    (the arm in :data:`pace.twin.robots.PUSHING`), the camera behind TIAGo's
    right shoulder, inside the passage, with the whole doorway in the frame."""
    c = PW.corridors(spec)[seg_id]
    z = spec.zone(from_zone)
    a0, a1 = np.array(c.along(0.0)), np.array(c.along(1.0))
    e = (a1 - a0) / np.linalg.norm(a1 - a0)                  # along the passage, towards t = 1
    into = 1.0 if np.hypot(*(np.array([z.x, z.y]) - a0)) < np.hypot(*(np.array([z.x, z.y]) - a1)) else -1.0
    h = into * e                                             # TIAGo's heading, the way the door opens
    left = np.array([-h[1], h[0]])
    m = np.array(c.along(0.5))
    acr = np.array(c.along(0.5, 1.0)) - m                    # unit vector of a positive across offset
    s = 1 if acr @ left > 0 else -1                          # the leaf on TIAGo's left
    hinge = m + s * (c.width / 2 - 0.15) * acr               # 0.15: the frame post (door_models)
    phi = math.radians(swing_deg)
    d = -s * acr * math.cos(phi) + h * math.sin(phi)         # hinge -> free edge, swung away
    n = np.array([-d[1], d[0]])
    n = n if n @ h < 0 else -n                               # the leaf's face towards TIAGo
    bar = hinge + 0.55 * d + 0.07 * n                        # on the bar's face, near its free end
    gx, gy, _gz = RB.PUSHING_GRASP
    base = bar - gx * h - gy * left
    level = z.level
    robot = RB.robot_static("mobile_manipulator", ROBOTS["mobile_manipulator"], float(base[0]), float(base[1]),
                            math.atan2(h[1], h[0]), level, label="r", posture=RB.PUSHING)
    swing = {seg_id: (into * phi, 0.0) if s < 0 else (0.0, into * phi)}
    cam_xy = m - 3.2 * h - 0.65 * left
    tgt = m + 0.1 * left + 0.3 * h
    z_cam, tz = level + 1.95, level + 1.2
    cam = S.Camera(float(cam_xy[0]), float(cam_xy[1]), z_cam, math.atan2(z_cam - tz, np.hypot(*(tgt - cam_xy))),
                   math.atan2(tgt[1] - cam_xy[1], tgt[0] - cam_xy[0]), width, height, math.radians(FOV_DEG))
    R.render(lambda fdir: PW.build_world(spec, extra_models=[robot], camera_xml=cam.sdf(fdir, rate=4.0),
                                         door_swing=swing).world,
             out=out, frames=5, timeout=300)


def assemble(imgs, out: Path, cols: int = 3, width_in: float = 7.2, dpi: int = 400, title_pt: float = 9.0) -> None:
    """The panels in a grid, a bold title over each.  The figure is set at
    ``\\linewidth`` (4.8 in in LNCS), so ``title_pt`` prints at 2/3 of itself."""
    rows = (len(imgs) + cols - 1) // cols
    gut = 0.03                                               # inches between panels
    pw = (width_in - (cols - 1) * gut) / cols
    ph = pw * imgs[0][1].shape[0] / imgs[0][1].shape[1]
    th = title_pt / 72 + 0.07                                # the title band over each panel
    H = rows * (ph + th) + (rows - 1) * gut
    fig = plt.figure(figsize=(width_in, H), dpi=dpi)
    for k, (title, img) in enumerate(imgs):
        r, cidx = divmod(k, cols)
        bottom = H - r * (ph + th + gut) - th - ph
        ax = fig.add_axes([cidx * (pw + gut) / width_in, bottom / H, pw / width_in, ph / H])
        ax.imshow(img, aspect="auto")
        ax.set_axis_off()
        t = ax.set_title(title, fontsize=title_pt, fontweight="bold", pad=2.0)
        room = pw * dpi * 0.98
        wide = t.get_window_extent(fig.canvas.get_renderer()).width
        if wide > room:                                      # shrink a title that does not fit its panel
            t.set_fontsize(title_pt * room / wide)
    fig.savefig(out, dpi=dpi, pil_kwargs={"quality": 92})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/figures")
    ap.add_argument("--only", nargs="*", default=None, help="render only these panels (the rest are reused)")
    ap.add_argument("--reuse-raw", action="store_true", help="render nothing, put the last renders together")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    spec = plant_spec()
    for key, robot, seg, zone, title, changes in PANELS:
        if a.reuse_raw or (a.only and key not in a.only):
            continue
        sp = PW.apply_changes(spec, changes) if changes else spec
        if key == "firedoor_tiago":
            door_panel(sp, seg, zone, RAW_DIR / f"{key}.jpg")
        else:
            panel(sp, key, robot, seg, zone, RAW_DIR / f"{key}.jpg")
        print("rendered", key)
    import imageio.v3 as iio
    imgs = [(title, iio.imread(RAW_DIR / f"{key}.jpg")) for key, *_r, title, _c in PANELS
            if (RAW_DIR / f"{key}.jpg").exists()]
    assemble(imgs, out / "fleet.jpg")
    print("wrote", out / "fleet.jpg")


if __name__ == "__main__":
    main()
