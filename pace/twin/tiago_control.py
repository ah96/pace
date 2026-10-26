"""TIAGo with a live arm for the pick missions: PAL's URDF, Gazebo's plugins, ros2_control.

The navigation twin drives TIAGo with its arm frozen (:mod:`pace.twin.robots`).
A mission that ends with a pick needs the arm, the torso and the gripper
under control, so this module builds a second model: PAL's expanded URDF with

* the Gazebo plugins for the base (diff drive with odometry from the
  world pose, joint states, ground truth, the base laser);
* a ``<ros2_control>`` system on ``gazebo_ros2_control/GazeboSystem`` for the
  seven arm joints, the torso, the head and the two gripper fingers, with the
  tucked posture as initial values;
* the ``gazebo_ros2_control`` model plugin and a controllers file: a joint
  state broadcaster and one ``JointTrajectoryController`` each for arm, torso,
  head and gripper -- what MoveIt's simple controller manager expects.

Arm, torso and head take positions.  (Velocity commands for the arm are an
option: gazebo_ros2_control carries them out by setting link velocities,
which pushed TIAGo's base about -- by up to 86 degrees in one swing.)  The
fingers take *efforts*, so that a closed gripper stops on what it holds and
squeezes with at most the URDF's 16 N.  The wheels' joint states come from a
Gazebo plugin, so that MoveIt knows every joint.

Needs ``ros-humble-gazebo-ros2-control`` and ``ros-humble-ros2-controllers``
at run time; building the files does not.
"""

from __future__ import annotations

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from .assets import A, ASSETS

ARM = [f"arm_{i}_joint" for i in range(1, 8)]
TORSO = ["torso_lift_joint"]
HEAD = ["head_1_joint", "head_2_joint"]
GRIPPER = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
WHEELS = ("wheel_left_joint", "wheel_right_joint")
WHEEL_SEPARATION = 0.4044
WHEEL_RADIUS = 0.0985
#: the arm tucked in front of the torso, the head looking slightly down (the demo's posture)
TUCKED = {"torso_lift_joint": 0.15, "arm_1_joint": 0.20, "arm_2_joint": -1.34, "arm_3_joint": -0.20,
          "arm_4_joint": 1.94, "arm_5_joint": -1.57, "arm_6_joint": 1.37, "arm_7_joint": 0.0,
          "head_1_joint": 0.0, "head_2_joint": -0.3,
          "gripper_left_finger_joint": 0.04, "gripper_right_finger_joint": 0.04}
GRIPPER_OPEN, GRIPPER_CLOSED = 0.044, 0.002
#: the fingers' contact: PAL's URDF lets them sink 10 cm into anything before a contact
#: force acts (minDepth 0.1; ODE takes the larger of two surfaces'), which a 3 cm handle
#: passes straight through
FINGER_CONTACT = {"kp": "100000.0", "kd": "50.0", "mu1": "1.5", "mu2": "1.5", "maxVel": "0.05", "minDepth": "0.001"}

BASE_PLUGINS = """
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <ros></ros>
      <update_rate>50</update_rate>
      <left_joint>wheel_left_joint</left_joint>
      <right_joint>wheel_right_joint</right_joint>
      <wheel_separation>{sep}</wheel_separation>
      <wheel_diameter>{dia}</wheel_diameter>
      <max_wheel_torque>60.0</max_wheel_torque>
      <max_wheel_acceleration>1.2</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>false</publish_wheel_tf>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_footprint</robot_base_frame>
      <odometry_source>1</odometry_source>
    </plugin>
    <plugin name="ground_truth" filename="libgazebo_ros_p3d.so">
      <ros><remapping>odom:=ground_truth/odom</remapping></ros>
      <body_name>base_footprint</body_name>
      <frame_name>world</frame_name>
      <update_rate>20.0</update_rate>
    </plugin>
    <plugin name="wheel_states" filename="libgazebo_ros_joint_state_publisher.so">
      <ros><remapping>~/out:=joint_states</remapping></ros>
      <update_rate>50</update_rate>
      <joint_name>wheel_left_joint</joint_name>
      <joint_name>wheel_right_joint</joint_name>
    </plugin>
    <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
      <parameters>{controllers}</parameters>
      <robot_param>robot_description</robot_param>
      <robot_param_node>robot_state_publisher</robot_param_node>
    </plugin>
  </gazebo>
  <gazebo reference="base_laser_link">
    <sensor name="base_laser" type="ray">
      <always_on>true</always_on>
      <visualize>false</visualize>
      <update_rate>10.0</update_rate>
      <pose>0 0 0 0 0 0</pose>
      <ray>
        <scan><horizontal><samples>540</samples><resolution>1</resolution>
          <min_angle>-2.356</min_angle><max_angle>2.356</max_angle></horizontal></scan>
        <range><min>0.12</min><max>10.0</max><resolution>0.01</resolution></range>
      </ray>
      <plugin name="laser_plugin" filename="libgazebo_ros_ray_sensor.so">
        <ros><remapping>~/out:=scan</remapping></ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>base_laser_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>
"""


def command_interface(joint: str, arm: str = "position") -> str:
    """``arm`` is how the arm is commanded: ``position`` (the default) or ``velocity``.
    The torso takes positions: it never moves while the gripper holds something, and
    under velocity commands it never came to rest within the controller's tolerance."""
    return arm if joint in ARM else "effort" if joint in GRIPPER else "position"


def ros2_control_block(arm: str = "position") -> str:
    joints = []
    # parents before children: gazebo_ros2_control sets each joint's velocity against its parent link's
    for j in TORSO + ARM + HEAD + GRIPPER:
        joints.append(f"""    <joint name="{j}">
      <command_interface name="{command_interface(j, arm)}"/>
      <state_interface name="position"><param name="initial_value">{TUCKED.get(j, 0.0)}</param></state_interface>
      <state_interface name="velocity"/>
    </joint>""")
    return ('  <ros2_control name="GazeboSystem" type="system">\n'
            "    <hardware><plugin>gazebo_ros2_control/GazeboSystem</plugin></hardware>\n"
            + "\n".join(joints) + "\n  </ros2_control>\n")


#: the trajectory controllers' position loop: velocity out for the arm (with the
#: trajectory's own velocity fed forward), effort out for the fingers (a stiff,
#: damped spring on 0.1 kg fingers, saturating at the URDF's 16 N)
GAINS = {"velocity": {"p": 8.0, "i": 0.0, "d": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
         "effort": {"p": 3000.0, "i": 0.0, "d": 30.0, "i_clamp": 0.0, "ff_velocity_scale": 0.0}}


def controllers_yaml(arm: str = "position") -> dict:
    def jtc(joints):
        cmd = command_interface(joints[0], arm)
        # a velocity-commanded wrist keeps a small jitter at rest (a few 0.01 rad/s); a joint set by
        # position reads, between two settings, what gravity did in the step (0.25 rad/s for a wrist
        # holding 2.4 kg) -- a velocity check says nothing there (0 = not checked)
        stopped = {"velocity": 0.2, "position": 0.0}.get(cmd, 0.05)
        params = {"joints": joints, "command_interfaces": [cmd],
                  "state_interfaces": ["position", "velocity"],
                  "state_publish_rate": 50.0, "action_monitor_rate": 20.0,
                  "allow_partial_joints_goal": False,
                  "constraints": {"stopped_velocity_tolerance": stopped, "goal_time": 2.0}}
        if cmd in GAINS:
            params["gains"] = {j: dict(GAINS[cmd]) for j in joints}
        return {"ros__parameters": params}
    return {
        "controller_manager": {"ros__parameters": {
            "update_rate": 1000, "use_sim_time": True,
            "joint_state_broadcaster": {"type": "joint_state_broadcaster/JointStateBroadcaster"},
            "arm_controller": {"type": "joint_trajectory_controller/JointTrajectoryController"},
            "torso_controller": {"type": "joint_trajectory_controller/JointTrajectoryController"},
            "head_controller": {"type": "joint_trajectory_controller/JointTrajectoryController"},
            "gripper_controller": {"type": "joint_trajectory_controller/JointTrajectoryController"}}},
        "arm_controller": jtc(ARM), "torso_controller": jtc(TORSO),
        "head_controller": jtc(HEAD), "gripper_controller": jtc(GRIPPER),
    }


def build(out_dir: Path, arm: str = "position") -> dict:
    """Write tiago_control.urdf (model:// meshes, for gz sdf), tiago_control.rsp.urdf
    (file:// meshes, for robot_state_publisher and MoveIt) and controllers.yaml."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring((A.PAL_DIR / "tiago.urdf").read_text())
    for el in list(root):
        if el.tag in ("ros2_control", "transmission"):
            root.remove(el)
        elif el.tag == "gazebo":
            if el.get("reference") is None or el.find("sensor") is not None or el.find("plugin") is not None:
                root.remove(el)
            elif el.get("reference") in ("gripper_left_finger_link", "gripper_right_finger_link"):
                for tag, value in FINGER_CONTACT.items():
                    (el.find(tag) if el.find(tag) is not None else ET.SubElement(el, tag)).text = value
        elif el.tag == "joint" and el.get("name") in GRIPPER:
            # Gazebo zeroes a joint's effort while it moves faster than its velocity limit;
            # pressed against a handle the fingers jitter past PAL's 0.05 m/s, and the squeeze
            # would keep switching off
            el.find("limit").set("velocity", "1.0")

    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    controllers = out_dir / "controllers.yaml"
    controllers.write_text(yaml.safe_dump(controllers_yaml(arm), sort_keys=False))
    tail = ros2_control_block(arm) + BASE_PLUGINS.format(sep=WHEEL_SEPARATION, dia=2 * WHEEL_RADIUS,
                                                       controllers=controllers)
    xml = xml.replace("</robot>", tail + "</robot>")
    xml = re.sub(r'filename="package://', 'filename="model://', xml)
    gz = out_dir / "tiago_control.urdf"
    gz.write_text('<?xml version="1.0"?>\n' + xml)
    rsp = out_dir / "tiago_control.rsp.urdf"
    rsp.write_text('<?xml version="1.0"?>\n' + xml.replace("model://", f"file://{A.PAL_DIR}/"))
    return {"urdf": gz, "rsp_urdf": rsp, "controllers": controllers}


def model_sdf(urdf: Path, x: float, y: float, yaw: float, z: float = 0.0) -> str:
    """The URDF converted to an SDF ``<model name="robot">`` posed in the world."""
    env = dict(os.environ, GAZEBO_MODEL_PATH=A.model_path())
    out = subprocess.run(["gz", "sdf", "-p", str(urdf)], env=env, capture_output=True, text=True, check=True)
    model = ET.fromstring(out.stdout).find("model")
    model.set("name", "robot")
    for old in model.findall("pose"):
        model.remove(old)
    pose = ET.Element("pose")
    pose.text = f"{x:.4f} {y:.4f} {z + 0.02:.4f} 0 0 {yaw:.4f}"
    model.insert(0, pose)
    return ET.tostring(model, encoding="unicode")


if __name__ == "__main__":
    import sys
    files = build(Path(sys.argv[1] if len(sys.argv) > 1 else "build/tiago_control"))
    print({k: str(v) for k, v in files.items()})
