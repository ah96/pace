# Third-party assets

Vendored unmodified from the sources below, so that the twin's worlds and figures
(`pace/twin/`) render without downloads.

## `fuel/` — Gazebo Fuel models

Models by **Open Robotics** from [app.gazebosim.org](https://app.gazebosim.org/OpenRobotics),
licensed **Creative Commons Attribution 4.0 International (CC BY 4.0)**:
WoodenChair, Table, Desk, Bookshelf, Cabinet, Sofa, Cardboard box, Utility Cart,
TrashBin, VendingMachine, FemaleVisitor, Nurse, Scrubs.  Each directory keeps
the model's own `model.config` (author and version) and `model.sdf`; the
thumbnails and older SDF revisions were left out.

## `pal/` — TIAGo by PAL Robotics

`tiago.urdf` is `tiago_description/robots/tiago.urdf.xacro` from PAL Robotics'
[tiago_robot](https://github.com/pal-robotics/tiago_robot) (with
[pmb2_robot](https://github.com/pal-robotics/pmb2_robot),
[pal_gripper](https://github.com/pal-robotics/pal_gripper) and
[pal_urdf_utils](https://github.com/pal-robotics/pal_urdf_utils)), branch
`humble-devel`, expanded with `end_effector:=pal-gripper ft_sensor:=schunk-ft
arm_type:=tiago-arm base_type:=pmb2 laser_model:=sick-571
camera_model:=orbbec-astra is_public_sim:=true`; the commits are in
`VERSION.txt`.  `<package>/meshes/` holds the meshes that URDF refers to.
All of it is licensed **Apache License 2.0** (`LICENSE.<repository>`).

## `tiago_tucked.urdf`

PAL's URDF with the arm, torso, head and gripper joints frozen in a tucked posture,
the model the navigation episodes drive. Same license as above.

## `robots/`

The fleet's meshes: the MiR100-class tote carrier and the forklift AGV from Open
Robotics' Fuel models (CC BY 4.0), the Husky-class yard UGV from Clearpath's
`husky_description` (BSD-3-Clause). The AWS RoboMaker small warehouse models (MIT-0)
are not stored here; `pace/twin/fetch.py` downloads them into `data/` on first use.
