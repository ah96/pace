"""Where the vendored Gazebo assets are (``assets/gazebo/`` in the repository; see its LICENSES.md).

The Fuel models (Open Robotics, CC BY 4.0) and PAL's TIAGo (Apache 2.0) are vendored
unmodified from their sources; the AWS RoboMaker
warehouse models (MIT-0) are fetched into ``data/aws/`` by :mod:`pace.twin.fetch`.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "gazebo"


class A:
    """The asset table the vendored SDF builders read."""

    FUEL_DIR = ASSETS / "fuel"
    PAL_DIR = ASSETS / "pal"
    FUEL = {"table": "Table", "desk": "Desk", "cabinet": "Cabinet", "box": "Cardboard_box",
            "bin": "TrashBin", "person": "FemaleVisitor", "nurse": "Nurse", "scrubs": "Scrubs"}
    ALIASES = {"cardboard_box": "Cardboard_box"}
    STRETCH = {"table", "desk", "cabinet", "box"}
    FACING = {"person": -1.5708, "nurse": -1.5708, "scrubs": -1.5708}

    @staticmethod
    def fuel_dir(key: str) -> Path:
        return A.FUEL_DIR / A.FUEL[key]

    @staticmethod
    def model_path() -> str:
        return os.pathsep.join([str(A.FUEL_DIR), str(A.PAL_DIR), "/usr/share/gazebo-11/models"])
