"""move_group's parameters for TIAGo in the twin, from PAL's MoveIt configuration.

PAL's ``tiago_moveit_config`` (Apache 2.0, vendored here) provides the SRDF
(generated for the PAL-gripper / Schunk-FT variant), the KDL kinematics,
joint limits, OMPL settings and the controller list. This module assembles
one parameters file for ``move_group`` from them and our URDF, so that no
ament package has to be built.
"""

from __future__ import annotations

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRDF = HERE / "tiago_pal-gripper_schunk-ft.srdf"


def move_group_params(rsp_urdf: Path, out: Path) -> Path:
    kin = yaml.safe_load((HERE / "kinematics_kdl.yaml").read_text())
    limits = yaml.safe_load((HERE / "joint_limits.yaml").read_text())
    ompl = yaml.safe_load((HERE / "ompl_planning.yaml").read_text())
    # PAL checks a path for collisions every 1% of the joint space's extent (~0.2 rad here), coarse
    # enough for OMPL to cut past a tote's corner and MoveIt to reject the smoothed path; 0.2% it is
    for group in ("arm_torso", "arm"):
        ompl[group]["longest_valid_segment_fraction"] = 0.002
    ctrl = yaml.safe_load((HERE / "controllers_pal-gripper.yaml").read_text())
    simple = ctrl["moveit_simple_controller_manager"]
    names = list(simple.get("controller_names", []))
    simple["controller_names"] = names + [n for n in ("gripper_controller",) if n not in names]
    simple["gripper_controller"] = {"action_ns": "follow_joint_trajectory", "type": "FollowJointTrajectory",
                                    "default": True,
                                    "joints": ["gripper_left_finger_joint", "gripper_right_finger_joint"]}
    params = {"move_group": {"ros__parameters": {
        "use_sim_time": True,
        "robot_description": Path(rsp_urdf).read_text(),
        "robot_description_semantic": SRDF.read_text(),
        "robot_description_kinematics": kin,
        "robot_description_planning": limits if "joint_limits" in limits else {"joint_limits": limits},
        "planning_pipelines": {"pipeline_names": ["ompl"]},
        "ompl": ompl,
        "default_planning_pipeline": "ompl",
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": simple,
        "trajectory_execution": ctrl.get("trajectory_execution", {}),
        "planning_scene_monitor": {"publish_planning_scene": True, "publish_geometry_updates": True,
                                   "publish_state_updates": True, "publish_transforms_updates": True},
        "monitor_dynamics": False,
        "allow_trajectory_execution": True,
        "capabilities": "",
        "disable_capabilities": "",
    }}}
    out = Path(out)
    out.write_text(yaml.safe_dump(params, sort_keys=False))
    return out
