"""Videos of the pick missions from the frames their cameras saved.

    PYTHONPATH=. python experiments/make_mission_videos.py [--root results/missions] [--fps 6]

Each mission run with ``--film`` has ``film_side/`` and ``film_front/``: a
frame every third of a second of simulated time from a camera beside the
pick and one beyond the stand.  For every mission this writes ``pick.mp4``
(the two views side by side; at 6 fps, twice real time) and ``pick_end.jpg``
(the last frame of both: the item held up, or the arm that could not reach
it).  Needs ffmpeg.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def frames(d: Path) -> list:
    return sorted(d.glob("*.jpg"))


def make(ep: Path, fps: float) -> str:
    side, front = frames(ep / "film_side"), frames(ep / "film_front")
    n = min(len(side), len(front))
    if n == 0:
        return "no frames"
    tmp = ep / "_pairs"
    tmp.mkdir(exist_ok=True)
    try:
        for k in range(n):
            a, b = Image.open(side[k]).convert("RGB"), Image.open(front[k]).convert("RGB")
            pair = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), "white")
            pair.paste(a, (0, 0))
            pair.paste(b, (a.width, 0))
            pair.save(tmp / f"{k:05d}.jpg", quality=90)
            if k == n - 1:
                pair.save(ep / "pick_end.jpg", quality=92)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "%05d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(ep / "pick.mp4")], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return f"{n} frames"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/missions")
    ap.add_argument("--fps", type=float, default=6.0, help="frames are 1/3 s of simulated time apart")
    a = ap.parse_args()
    for ep in sorted(p for p in Path(a.root).iterdir() if (p / "film_side").is_dir()):
        print(f"{ep.name}: {make(ep, a.fps)}")


if __name__ == "__main__":
    main()
