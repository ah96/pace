"""The fleet in Gazebo: real robot models on the bodies the ontology declares.

Every robot is driven the same way -- a differential-drive base with the
standard ``gazebo_ros`` plugins (diff drive with odometry from the world pose,
a p3d ground truth, a 2-D lidar) and a collision hull that is exactly the box
the affordance rules compare against a passage: ``robot_width`` x ``length``
x ``robot_height``.  What the camera sees is the real robot:

* the tote carrier is a MiR100 (DFKI's ``mir_description`` meshes, BSD-3)
  with a tote module on its top plate;
* the forklift AGV is Open Robotics' counterbalance forklift (Fuel, CC BY 4.0,
  converted from glTF to OBJ), its mast up;
* the mobile manipulator is PAL's TIAGo (its vendored URDF, arm tucked), with
  its own laser in the base;
* the yard UGV is a Clearpath Husky (``husky_description`` meshes, BSD-3)
  with a sensor mast.

For Nav2 each parametric robot also gets a minimal URDF for
``robot_state_publisher`` (``base_footprint -> base_link -> laser_link``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .assets import ASSETS, A
from . import sdf as S

CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pace"
ROBOT_ASSETS = ASSETS / "robots"

#: TIAGo's arm tucked in front of the torso, the head looking slightly down:
#: the posture the twin drives with (baked into the vendored ``tiago_tucked.urdf``)
TUCKED = {"torso_lift_joint": 0.15, "arm_1_joint": 0.20, "arm_2_joint": -1.34, "arm_3_joint": -0.20,
          "arm_4_joint": 1.94, "arm_5_joint": -1.57, "arm_6_joint": 1.37, "arm_7_joint": 0.0,
          "head_1_joint": 0.0, "head_2_joint": -0.3,
          "gripper_left_finger_joint": 0.04, "gripper_right_finger_joint": 0.04}
#: the arm stretched out to a fire door's push bar: the grasping frame at
#: :data:`PUSHING_GRASP`, ahead and to the right of the base, approaching
#: straight ahead with the fingers side by side, the elbow out to the right,
#: the head turned to the bar (solved by :func:`pace.twin.tiago_fk.solve_reach`
#: with ``elbow_y=-0.40``)
PUSHING = dict(TUCKED, torso_lift_joint=0.189, arm_1_joint=0.293, arm_2_joint=0.095, arm_3_joint=-1.834,
               arm_4_joint=1.906, arm_5_joint=-1.122, arm_6_joint=-0.688, arm_7_joint=-1.914,
               head_1_joint=-0.25, head_2_joint=-0.4,
               gripper_left_finger_joint=0.02, gripper_right_finger_joint=0.02)
#: where the grasping frame is in ``base_footprint`` at :data:`PUSHING` (x ahead, y left, z up)
PUSHING_GRASP = (0.70, -0.22, 1.0)


def _box(name, cx, cy, cz, dx, dy, dz, colour, yaw=0.0):
    return S.box_visual(name, f"{cx:.4f} {cy:.4f} {cz:.4f} 0 0 {yaw:.4f}", (dx, dy, dz), S.material(colour))


def _cyl(name, cx, cy, cz, r, L, colour, roll=math.pi / 2):
    return S.cylinder_visual(name, f"{cx:.4f} {cy:.4f} {cz:.4f} {roll:.4f} 0 0", r, L, S.material(colour))


def _mesh(name: str, path: Path, pose: str, scale: float = 1.0, colour: str | None = None,
          emissive: float = 0.0) -> str:
    mat = S.material(colour, emissive=emissive) if colour else ""
    return (f'<visual name="{name}"><pose>{pose}</pose><geometry><mesh><uri>file://{path}</uri>'
            f"<scale>{scale} {scale} {scale}</scale></mesh></geometry>{mat}</visual>")


def _inertial(mass: float, dx: float, dy: float, dz: float) -> str:
    ixx = mass / 12 * (dy ** 2 + dz ** 2)
    iyy = mass / 12 * (dx ** 2 + dz ** 2)
    izz = mass / 12 * (dx ** 2 + dy ** 2)
    return (f"<inertial><mass>{mass:.3f}</mass><inertia><ixx>{ixx:.4f}</ixx><iyy>{iyy:.4f}</iyy>"
            f"<izz>{izz:.4f}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>")


# ------------------------------------------------------------------ looks

def look_tote_amr(L: float, W: float, H: float) -> List[str]:
    """A MiR100 with a tote module: the base mesh, its wheels, a grey tote box on top."""
    d = ROBOT_ASSETS / "mir100"
    parts = [_mesh("base", d / "mir_100_base.stl", "0 0 0 0 0 0", colour="#e8e8e8")]
    for s in (-1, 1):
        parts.append(_mesh(f"wheel{s + 1}", d / "wheel.stl", f"0 {s * 0.222:.3f} 0.0625 0 0 0", colour="#2a2a2a"))
    for cx in (0.34, -0.34):
        for s in (-1, 1):
            parts.append(_mesh(f"caster{cx}{s}", d / "caster_wheel_base.stl", f"{cx:.3f} {s * 0.2:.3f} 0.045 0 0 0",
                               colour="#3a3a3a"))
    parts.append(_box("top_plate", 0, 0, 0.345, 0.86, 0.56, 0.02, "#2b4f8f"))
    parts.append(_box("tote", 0, 0, 0.355 + (H - 0.36) / 2, L - 0.16, W - 0.08, H - 0.36, "#5a6b7a"))
    parts.append(_box("tote_rim", 0, 0, H - 0.02, L - 0.12, W - 0.04, 0.04, "#3e4a55"))
    parts.append(_box("beacon", -L / 2 + 0.05, 0, 0.42, 0.05, 0.12, 0.08, "#ff6f1a"))
    return parts


def look_forklift_agv(L: float, W: float, H: float) -> List[str]:
    """Open Robotics' forklift, forks forward, its mast up, a loaded EUR pallet on the forks."""
    d = ROBOT_ASSETS / "forklift"
    # the mesh's forks point -y and its length spans y in [-0.874, 2.294]: yaw it by
    # +90 deg so the forks point +x, then centre it on the hull
    parts = [_mesh("body", d / "forklift.obj", "0.71 0 0.016 0 0 1.5708", colour="#f0b429", emissive=0.5)]
    # a pallet with load on the forks (the forks are at the +x end, about 1.2 m long)
    px = L / 2 - 0.75
    parts.append(_box("pallet", px, 0, 0.10 + 0.075, 0.8, 1.2, 0.15, "#a8754a"))
    parts.append(_box("load", px, 0, 0.25 + 0.55, 0.78, 1.15, 1.1, "#c9a06c"))
    parts.append(_box("strap", px, 0, 0.25 + 0.55, 0.80, 1.17, 0.08, "#2b4f8f"))
    parts.append(_box("beacon", -L / 2 + 0.9, 0, 2.15, 0.12, 0.12, 0.12, "#ff6f1a"))
    return parts


def look_yard_ugv(L: float, W: float, H: float) -> List[str]:
    """A Clearpath Husky with its top plate, rails and a sensor mast."""
    d = ROBOT_ASSETS / "husky"
    z0 = 0.13228                                          # base_link above the ground
    parts = [_mesh("base", d / "base_link.dae", f"0 0 {z0:.4f} 0 0 0"),
             _mesh("top_chassis", d / "top_chassis.dae", f"0 0 {z0:.4f} 0 0 0"),
             _mesh("top_plate", d / "top_plate.dae", f"0.0812 0 {z0 + 0.245:.4f} 0 0 0"),
             _mesh("rail", d / "user_rail.dae", f"0.272 0 {z0 + 0.245:.4f} 0 0 0"),
             _mesh("bumper_f", d / "bumper.dae", f"0.48 0 {z0 + 0.091:.4f} 0 0 0"),
             _mesh("bumper_r", d / "bumper.dae", f"-0.48 0 {z0 + 0.091:.4f} 0 0 3.1416")]
    for cx in (0.256, -0.256):
        for s in (-1, 1):
            parts.append(_mesh(f"wheel{cx}{s}", d / "wheel.dae", f"{cx:.3f} {s * 0.2854:.4f} 0.1651 0 0 0"))
    parts.append(_cyl("mast", -0.1, 0, 0.42, 0.02, 0.28, "#2f2f2f", roll=0.0))
    parts.append(_box("sensor", -0.1, 0, H - 0.05, 0.14, 0.14, 0.1, "#2f2f2f"))
    return parts


LOOKS = {"tote_amr": look_tote_amr, "forklift_agv": look_forklift_agv, "yard_ugv": look_yard_ugv}
MASS = {"tote_amr": 70.0, "forklift_agv": 350.0, "yard_ugv": 50.0}
TORQUE = {"tote_amr": 40.0, "forklift_agv": 150.0, "yard_ugv": 60.0}


# ----------------------------------------------------------------- models

def parametric_model(name: str, props: Dict[str, object], x: float, y: float, yaw: float,
                     z: float = 0.0, scan_topic: str = "/scan") -> str:
    """SDF of a driven robot at (x, y, yaw), as ``<model name="robot">``."""
    W = float(props["robot_width"]); L = float(props.get("length", 1.0)); H = float(props["robot_height"])
    laser_z = float(props.get("laser_height", 0.2))
    wheel_r = 0.10 if name != "yard_ugv" else 0.1651
    wheel_w = 0.06
    body_z0 = 0.06                                        # underside of the hull
    body_h = H - body_z0
    mass = MASS.get(name, 80.0)
    sep = W - wheel_w - 0.02 if name != "yard_ugv" else 0.5708
    parts = []
    parts.append(f'<collision name="hull"><pose>0 0 {body_z0 + body_h / 2:.4f} 0 0 0</pose>'
                 f"<geometry><box><size>{L:.4f} {W:.4f} {body_h:.4f}</size></box></geometry>"
                 "<surface><contact><ode><max_vel>0.1</max_vel><min_depth>0.001</min_depth></ode></contact>"
                 "<friction><ode><mu>0.2</mu><mu2>0.2</mu2></ode></friction></surface></collision>")
    parts.append(_inertial(mass, L, W, body_h).replace("<inertial>", f"<inertial><pose>0 0 {body_z0 + body_h / 3:.3f} 0 0 0</pose>"))
    parts += LOOKS[name](L, W, H)
    # --- lidar, proud of the front face, with a 190 deg field of view: a ray
    # looking back past 90 deg would cross the hull's front face at
    # y = d * tan(theta), which for theta <= 95 deg and d = 0.10 is beyond any
    # body half-width in the fleet, so the body is never its own obstacle
    lx = L / 2 + 0.10
    r_min = 0.2
    parts.append(_cyl("lidar", lx, 0, laser_z, 0.045, 0.06, "#111111", roll=0.0))
    parts.append(_box("lidar_bracket", L / 2 + 0.04, 0, laser_z - 0.05, 0.12, 0.12, 0.04, "#2f2f2f"))
    parts.append(f"""<sensor name="lidar" type="ray"><pose>{lx:.4f} 0 {laser_z:.4f} 0 0 0</pose>
      <always_on>true</always_on><update_rate>10</update_rate><visualize>false</visualize>
      <ray><scan><horizontal><samples>540</samples><resolution>1</resolution><min_angle>-1.658</min_angle><max_angle>1.658</max_angle></horizontal></scan>
      <range><min>{r_min}</min><max>25.0</max><resolution>0.01</resolution></range>
      <noise><type>gaussian</type><mean>0.0</mean><stddev>0.01</stddev></noise></ray>
      <plugin name="laser" filename="libgazebo_ros_ray_sensor.so"><ros><remapping>~/out:={scan_topic}</remapping></ros>
        <output_type>sensor_msgs/LaserScan</output_type><frame_name>laser_link</frame_name></plugin></sensor>""")
    base = f'<link name="base_link"><pose>0 0 0 0 0 0</pose>{"".join(parts)}</link>'
    links, joints = [], []
    for side, s in (("left", 1), ("right", -1)):
        links.append(f"""<link name="wheel_{side}"><pose>0 {s * sep / 2:.4f} {wheel_r:.4f} {-math.pi / 2:.5f} 0 0</pose>
      {_inertial(4.0, 2 * wheel_r, 2 * wheel_r, wheel_w)}
      <collision name="c"><geometry><cylinder><radius>{wheel_r}</radius><length>{wheel_w}</length></cylinder></geometry>
        <surface><friction><ode><mu>1.5</mu><mu2>1.5</mu2><fdir1>1 0 0</fdir1></ode></friction>
        <contact><ode><kp>1e8</kp><kd>1000</kd><min_depth>0.001</min_depth><max_vel>0.1</max_vel></ode></contact></surface></collision></link>""")
        joints.append(f'<joint name="wheel_{side}_joint" type="revolute"><parent>base_link</parent><child>wheel_{side}</child>'
                      f"<axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>")
    for k, cx in enumerate((L / 2 - 0.2, -L / 2 + 0.2)):
        for s in (-1, 1):
            r = wheel_r
            links.append(f"""<link name="caster{k}{s + 1}"><pose>{cx:.4f} {s * (W / 2 - 0.18):.4f} {r:.4f} 0 0 0</pose>
      {_inertial(1.0, 2 * r, 2 * r, 2 * r)}
      <collision name="c"><geometry><sphere><radius>{r}</radius></sphere></geometry>
        <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction>
        <contact><ode><kp>1e8</kp><kd>1000</kd><min_depth>0.001</min_depth><max_vel>0.1</max_vel></ode></contact></surface></collision></link>""")
            joints.append(f'<joint name="caster{k}{s + 1}_joint" type="fixed"><parent>base_link</parent><child>caster{k}{s + 1}</child></joint>')
    plugins = f"""<plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so"><ros></ros>
      <update_rate>50</update_rate><left_joint>wheel_left_joint</left_joint><right_joint>wheel_right_joint</right_joint>
      <wheel_separation>{sep:.4f}</wheel_separation><wheel_diameter>{2 * wheel_r:.4f}</wheel_diameter>
      <max_wheel_torque>{TORQUE.get(name, 40.0)}</max_wheel_torque><max_wheel_acceleration>1.5</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic><publish_odom>true</publish_odom><publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>false</publish_wheel_tf><odometry_topic>odom</odometry_topic><odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_footprint</robot_base_frame><odometry_source>1</odometry_source></plugin>
    <plugin name="ground_truth" filename="libgazebo_ros_p3d.so"><ros><remapping>odom:=ground_truth/odom</remapping></ros>
      <body_name>base_link</body_name><frame_name>world</frame_name><update_rate>20</update_rate></plugin>"""
    footprint = ('<link name="base_footprint"><pose>0 0 0 0 0 0</pose>'
                 f"{_inertial(0.01, 0.1, 0.1, 0.1)}</link>"
                 '<joint name="base_footprint_joint" type="fixed"><parent>base_footprint</parent><child>base_link</child></joint>')
    return (f'<model name="robot"><static>false</static><pose>{x:.4f} {y:.4f} {z + 0.02:.4f} 0 0 {yaw:.4f}</pose>'
            f"{footprint}{base}{''.join(links)}{''.join(joints)}{plugins}</model>")


def parametric_urdf(name: str, props: Dict[str, object]) -> str:
    """A minimal URDF (frames only) for robot_state_publisher."""
    W = float(props["robot_width"]); L = float(props.get("length", 1.0)); H = float(props["robot_height"])
    laser_z = float(props.get("laser_height", 0.2))
    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_footprint"/>
  <link name="base_link"><visual><origin xyz="0 0 {H / 2:.3f}"/><geometry><box size="{L} {W} {H}"/></geometry></visual></link>
  <joint name="base_footprint_joint" type="fixed"><parent link="base_footprint"/><child link="base_link"/><origin xyz="0 0 0"/></joint>
  <link name="laser_link"/>
  <joint name="laser_joint" type="fixed"><parent link="base_link"/><child link="laser_link"/><origin xyz="{L / 2 + 0.10:.4f} 0 {laser_z:.4f}"/></joint>
</robot>
"""


def _urdf_to_sdf(urdf: Path, sdf: Path) -> str:
    """Gazebo's URDF -> SDF conversion, done once per file."""
    if not sdf.exists():
        env = dict(os.environ, GAZEBO_MODEL_PATH=A.model_path())
        out = subprocess.run(["gz", "sdf", "-p", str(urdf)], env=env, capture_output=True, text=True, check=True)
        sdf.write_text(out.stdout)
    return sdf.read_text()


def _model_at(sdf_text: str, x: float, y: float, yaw: float, z: float) -> str:
    model = ET.fromstring(sdf_text).find("model")
    model.set("name", "robot")
    for old in model.findall("pose"):
        model.remove(old)
    pose = ET.Element("pose")
    pose.text = f"{x:.4f} {y:.4f} {z + 0.02:.4f} 0 0 {yaw:.4f}"
    model.insert(0, pose)
    return ET.tostring(model, encoding="unicode")


def tiago_model(x: float, y: float, yaw: float, z: float = 0.0) -> str:
    """PAL's TIAGo with its drive and laser plugins, converted once from the vendored URDF."""
    CACHE.mkdir(parents=True, exist_ok=True)
    return _model_at(_urdf_to_sdf(ASSETS / "tiago_tucked.urdf", CACHE / "tiago_live.sdf"), x, y, yaw, z)


def _freeze(joint: ET.Element, q: float) -> None:
    """Make a URDF joint fixed at value ``q``: compose ``q`` into its origin."""
    o = joint.find("origin")
    xyz = [float(t) for t in ((o.get("xyz") if o is not None else None) or "0 0 0").split()]
    rpy = [float(t) for t in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
    T = S.pose_matrix(*xyz, *rpy)
    ax_el = joint.find("axis")
    ax = np.array([float(t) for t in ((ax_el.get("xyz") if ax_el is not None else None) or "1 0 0").split()])
    ax = ax / max(np.linalg.norm(ax), 1e-9)
    if joint.get("type") in ("revolute", "continuous") and q:
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Tq = np.eye(4)
        Tq[:3, :3] = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K
        T = T @ Tq
    elif joint.get("type") == "prismatic" and q:
        Tq = np.eye(4)
        Tq[:3, 3] = ax * q
        T = T @ Tq
    if o is None:
        o = ET.SubElement(joint, "origin")
    r, p, yw = S.matrix_rpy(T[:3, :3])
    o.set("xyz", " ".join(f"{v:.6f}" for v in T[:3, 3]))
    o.set("rpy", f"{r:.6f} {p:.6f} {yw:.6f}")
    joint.set("type", "fixed")
    for tag in ("limit", "dynamics", "safety_controller", "calibration", "mimic"):
        for el in joint.findall(tag):
            joint.remove(el)


def tiago_posed_sdf(posture: Dict[str, float]) -> str:
    """PAL's TIAGo with every joint frozen at ``posture`` (radians and metres;
    joints it does not name at zero), as SDF, converted once per posture: for
    stills of the robot doing something, such as pushing a door open."""
    key = hashlib.md5(json.dumps(sorted(posture.items())).encode()).hexdigest()[:10]
    CACHE.mkdir(parents=True, exist_ok=True)
    urdf = CACHE / f"tiago_posed_{key}.urdf"
    if not urdf.exists():
        root = ET.parse(A.PAL_DIR / "tiago.urdf").getroot()
        for el in list(root):                              # PAL's control stack and sensor plugins
            if el.tag in ("ros2_control", "transmission") or (
                    el.tag == "gazebo" and (el.get("reference") is None or el.find("sensor") is not None
                                            or el.find("plugin") is not None)):
                root.remove(el)
        for j in root.findall("joint"):
            if j.get("type") in ("revolute", "prismatic", "continuous"):
                _freeze(j, float(posture.get(j.get("name"), 0.0)))
        xml = ET.tostring(root, encoding="unicode").replace('filename="package://', 'filename="model://')
        urdf.write_text('<?xml version="1.0"?>\n' + xml)
    return _urdf_to_sdf(urdf, CACHE / f"tiago_posed_{key}.sdf")


def tiago_urdf() -> str:
    """PAL's tucked URDF for robot_state_publisher, its ``model://`` meshes resolved to files."""
    return (ASSETS / "tiago_tucked.rsp.urdf").read_text().replace("model://", f"file://{ASSETS / 'pal'}/")


def tiago_static(x: float, y: float, yaw: float, z: float = 0.0, name: str = "tiago",
                 posture: Optional[Dict[str, float]] = None) -> str:
    """TIAGo as scenery (no plugins, no collisions): for stills.  ``posture``
    puts its joints somewhere other than tucked."""
    src = tiago_model(x, y, yaw, z) if posture is None else _model_at(tiago_posed_sdf(posture), x, y, yaw, z)
    model = ET.fromstring(src)
    for tag in ("plugin", "sensor", "collision", "inertial", "joint"):
        for parent in model.iter():
            for child in list(parent):
                if child.tag == tag and not (tag == "joint" and parent is not model):
                    parent.remove(child)
    body = "".join(ET.tostring(link, encoding="unicode") for link in model.findall("link"))
    body = re.sub(r"model://([^/<]+)/", lambda m: f"file://{A.PAL_DIR / m.group(1)}/", body)
    return (f'<model name="{name}"><static>true</static><pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.4f}</pose>'
            f"{body}</model>")


def robot_model(name: str, props: Dict[str, object], x: float, y: float, yaw: float, z: float = 0.0) -> str:
    if name == "mobile_manipulator":
        return tiago_model(x, y, yaw, z)
    return parametric_model(name, props, x, y, yaw, z)


def robot_urdf(name: str, props: Dict[str, object]) -> str:
    if name == "mobile_manipulator":
        return tiago_urdf()
    return parametric_urdf(name, props)


def robot_static(name: str, props: Dict[str, object], x: float, y: float, yaw: float, z: float = 0.0,
                 label: str | None = None, posture: Optional[Dict[str, float]] = None) -> str:
    """A robot as scenery, for stills: the look without plugins or collisions
    (``posture``: TIAGo's joints, tucked when not given)."""
    if name == "mobile_manipulator":
        return tiago_static(x, y, yaw, z, label or name, posture)
    model = ET.fromstring(parametric_model(name, props, x, y, yaw, z))
    model.set("name", label or name)
    model.find("static").text = "true"
    for tag in ("plugin", "sensor", "collision", "inertial"):
        for parent in model.iter():
            for child in list(parent):
                if child.tag == tag:
                    parent.remove(child)
    return ET.tostring(model, encoding="unicode")


def nav2_footprint(props: Dict[str, object]) -> str:
    """Nav2's footprint polygon string for the declared body, centred on base_link."""
    W = float(props["robot_width"]); L = float(props.get("length", 1.0))
    hx, hy = L / 2, W / 2
    return f"[[{hx:.3f}, {hy:.3f}], [{hx:.3f}, {-hy:.3f}], [{-hx:.3f}, {-hy:.3f}], [{-hx:.3f}, {hy:.3f}]]"
