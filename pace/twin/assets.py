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
DATA = ROOT / "data"


class A:
    """The asset table the vendored SDF builders read."""

    FUEL_DIR = ASSETS / "fuel"
    PAL_DIR = ASSETS / "pal"
    FUEL = {"chair": "WoodenChair", "table": "Table", "desk": "Desk", "bookshelf": "Bookshelf",
            "cabinet": "Cabinet", "couch": "Sofa", "box": "Cardboard_box", "cart": "Utility_Cart",
            "bin": "TrashBin", "vending_machine": "VendingMachine", "person": "FemaleVisitor",
            "nurse": "Nurse", "scrubs": "Scrubs"}
    ALIASES = {"cardboard_box": "Cardboard_box", "person_standing": "Standing_person",
               "hinged_door": "Hinged_door", "door_handle": "Door_handle"}
    STRETCH = {"table", "desk", "bookshelf", "cabinet", "couch", "box", "cart"}
    FACING = {"person": -1.5708, "nurse": -1.5708, "scrubs": -1.5708}

    @staticmethod
    def fuel_dir(key: str) -> Path:
        return A.FUEL_DIR / A.FUEL[key]

    @staticmethod
    def model_path() -> str:
        extra = [str(p) for p in (DATA / "aws" / "small-warehouse" / "models",) if p.exists()]
        return os.pathsep.join([str(A.FUEL_DIR), str(A.PAL_DIR), *extra, "/usr/share/gazebo-11/models"])
