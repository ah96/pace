"""Stills of a twin world: ``gzserver`` with a camera sensor, no GUI, no ROS.

The camera model (:meth:`pace.twin.sdf.Camera.sdf`) saves frames to a
directory; we start the server, wait until a few frames are in (the first
ones may miss textures), keep one, and kill the server by process group.
Rendering needs an OpenGL display (``DISPLAY``): the laptop's and the lab
PC's ``:0`` both work over SSH.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .assets import A


def gazebo_env(**extra) -> dict:
    env = dict(os.environ)
    env["GAZEBO_MODEL_PATH"] = os.pathsep.join(
        p for p in [A.model_path(), env.get("GAZEBO_MODEL_PATH", "")] if p)
    env.setdefault("GAZEBO_MODEL_DATABASE_URI", "")          # never stall on the online database
    env.setdefault("RCUTILS_COLORIZED_OUTPUT", "0")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def render(world_xml_for: "callable", out: str | Path | None = None, frames: int = 6,
           timeout: float = 150.0, attempts: int = 3, display: str | None = None) -> np.ndarray:
    """One frame of the world.  ``world_xml_for(frames_dir)`` returns the SDF
    with a camera saving into ``frames_dir``.  Returns an RGB array; also saved
    to ``out``.  gzserver occasionally dies at startup, so it is retried."""
    for k in range(attempts):
        try:
            return _render(world_xml_for, out, frames, timeout, display)
        except RuntimeError:
            if k == attempts - 1:
                raise
            time.sleep(3.0)


def _render(world_xml_for, out, frames, timeout, display) -> np.ndarray:
    import imageio.v3 as iio
    env = gazebo_env()
    if display:
        env["DISPLAY"] = display
    if not env.get("DISPLAY"):
        raise RuntimeError("Gazebo's camera needs an OpenGL display: set DISPLAY (e.g. :0)")
    tmp = Path(tempfile.mkdtemp(prefix="pace_gz_"))
    try:
        fdir = tmp / "frames"
        fdir.mkdir()
        wf = tmp / "world.sdf"
        wf.write_text(world_xml_for(fdir))
        log = open(tmp / "gzserver.log", "w")
        p = subprocess.Popen(["gzserver", str(wf)], env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < timeout:
                shots = sorted(fdir.glob("*.jpg"))
                if len(shots) > frames:
                    break
                if p.poll() is not None:
                    break
                time.sleep(0.25)
        finally:
            for sig in (signal.SIGINT, signal.SIGKILL):
                if p.poll() is not None:
                    break
                try:
                    os.killpg(p.pid, sig)
                    p.wait(10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            log.close()
        shots = sorted(fdir.glob("*.jpg"))
        if not shots:
            raise RuntimeError("gzserver saved no frame:\n" + (tmp / "gzserver.log").read_text()[-3000:])
        img = iio.imread(shots[-2] if len(shots) > 1 else shots[-1])
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(out, img)
        return img
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
