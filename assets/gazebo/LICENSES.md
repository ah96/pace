# Third-party assets

Vendored from the sources below, unmodified unless noted, so that the twin's worlds and figures
(`pace/twin/`) render without downloads.

## `fuel/` — Gazebo Fuel models

Models by **Open Robotics** from [app.gazebosim.org](https://app.gazebosim.org/OpenRobotics),
licensed **Creative Commons Attribution 4.0 International (CC BY 4.0)**:
Table, Desk, Cabinet, Cardboard box, TrashBin, FemaleVisitor, Nurse, Scrubs. Each
directory keeps the model's own `model.config` (author and version) and `model.sdf`;
the thumbnails and older SDF revisions were left out.

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

PAL's URDF (also `tiago_tucked.rsp.urdf`, for `robot_state_publisher`), **modified**: the
arm, torso, head and gripper joints are frozen in a tucked posture, the model the navigation
episodes drive; each file carries a notice. Same license as above.

## `pace/twin/moveit/` — TIAGo's MoveIt configuration

From PAL Robotics' `tiago_moveit_config` (branch `humble-devel`), licensed **Apache License
2.0** (`pace/twin/moveit/LICENSE.tiago_moveit_config`); `tiago_pal-gripper_schunk-ft.srdf`
is the configuration's SRDF expanded for this robot.

## `robots/`

The fleet's meshes:

- `mir100/`: the MiR100-class tote carrier, from DFKI's `mir_description`
  (BSD-3-Clause, `robots/mir100/LICENSE`).
- `forklift/`: the forklift AGV, "Forklift" by Open Robotics from Gazebo Fuel (CC BY 4.0),
  **modified**: converted from glTF to OBJ; attribution in `robots/forklift/ATTRIBUTION.md`.
- `husky/`: the Husky-class yard UGV, the meshes of Clearpath's
  [husky_description](https://github.com/husky/husky) (BSD-3-Clause, `robots/husky/LICENSE`;
  identical on `noetic-devel` 41e15d283a8d and `humble-devel` 729f8aa45ccd).
