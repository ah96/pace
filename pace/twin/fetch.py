"""Fetch the third-party worlds and props the twin can use (into ``data/``, git-ignored).

* AWS RoboMaker Small Warehouse (MIT-0): the public warehouse benchmark that
  ships with a Nav2 map; ``aws_warehouse()`` clones the ``ros2`` branch.
* Open Robotics Fuel props under CC0 (Euro pallet, cardboard box, hinged door,
  NIST ramp, jersey barrier): ``fuel_props()``.

    python -m pace.twin.fetch
"""

from __future__ import annotations

import io
import subprocess
import time
import zipfile
from pathlib import Path

from .assets import DATA

AWS_REPO = "https://github.com/aws-robotics/aws-robomaker-small-warehouse-world.git"
FUEL = "https://fuel.gazebosim.org/1.0/OpenRobotics/models"
PROPS = ("Euro pallet", "Cardboard box", "Hinged door", "NIST simple ramp 120", "Jersey barrier",
         "Construction Cone", "Construction Barrel", "Standing person", "Walking person")


def aws_warehouse() -> Path:
    dest = DATA / "aws" / "small-warehouse"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", "ros2", AWS_REPO, str(dest)], check=True)
    return dest


def fuel_props(names=PROPS, pause: float = 0.5) -> Path:
    import requests
    dest = DATA / "fuel"
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        folder = dest / name.replace(" ", "_")
        if folder.exists():
            continue
        enc = name.replace(" ", "%20")
        r = requests.get(f"{FUEL}/{enc}/tip/{enc}.zip", timeout=300)
        if r.status_code == 200:
            zipfile.ZipFile(io.BytesIO(r.content)).extractall(folder)
        else:
            print(f"  {name}: HTTP {r.status_code}")
        time.sleep(pause)
    return dest


if __name__ == "__main__":
    print("AWS warehouse ->", aws_warehouse())
    print("Fuel props ->", fuel_props())
