"""A pick mission in the twin: Nav2 to the item's zone, then MoveIt at the item.

This is the manipulation half of the protocol (Sect. "In the twin").  The
episode is the navigation episode of :mod:`pace.twin.episode` with TIAGo's
arm alive (:mod:`pace.twin.tiago_control`); once the last zone is reached:

1. **Approach.** Nav2 to a staging pose in front of the item's stand, then a
   slow, straight drive to the pick pose: facing the stand, the base centre
   :data:`STANDOFF` from the item.
2. **Scene.** The stand and the item's body go into MoveIt's planning scene.
3. **Grasp.** Candidates in turn: from above (the fingers straddle the handle
   across its 3 cm, both ways round) and from the front (horizontal, along
   the handle, the base a little further back so that the folded arm clears
   the head).  For each, ``move_group`` plans to a pre-grasp pose for the
   ``arm_torso`` group with OMPL on PAL's configuration
   (:mod:`pace.twin.moveit.params`); the gripper opens; a straight descent;
   the gripper closes; a straight lift.  The trajectories run through
   ``ros2_control`` in Gazebo.
4. **Verdict.** From Gazebo's own pose of the item.

Outcomes, beyond the navigation ones: ``unreachable`` when no candidate gets
to its grasp pose (the case a restock repairs); ``grasped`` when both fingers
closed on the handle (they stop with its 3 cm between them); ``picked`` when,
besides, the item rose with the gripper; ``missed`` when the fingers closed on
nothing; ``approach_failed`` when the base did not get to the pick pose.

The arm is commanded by position.  gazebo_ros2_control carries a velocity
command out by setting the links' velocities, which turned TIAGo's base by up
to 86 degrees as the arm swung out; a position command sets the joint where it
should be, at zero velocity -- which also means a gripper so moved carries
nothing by friction.  So, as is usual in Gazebo Classic, the part is made fast
to the gripper once the grasp is made -- both fingers stopped on the handle --
with the model attachment plugin (``ros-humble-gazebo-model-attachment-plugin``,
a fixed joint to the wrist); without the plugin the lift is attempted and
recorded (``item_rise``) and the grasp is what is judged.

    python -m pace.twin.mission --route dock,receiving,diagnostics,disassembly_A --item module_bench --out build/mission
    python -m pace.twin.mission --route disassembly_A --item module_bench --out build/pick     # the pick alone
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from scenarios.plant import plant_spec
from . import plant_world as PW
from . import tiago_control as TC
from .episode import Episode, route_mask, write_map
from .moveit.params import move_group_params

GRASP_FRAME = "gripper_grasping_frame"
GROUP = "arm_torso"         # to the pre-grasp pose; the straight moves after it are the arm's alone
ARM_GROUP = "arm"
PLANNING_FRAME = "base_footprint"           # the URDF's root: MoveIt plans in the base's frame
CONTROLLERS = ("joint_state_broadcaster", "arm_controller", "torso_controller", "head_controller", "gripper_controller")
STANDOFF = {"front": 0.75, "front_far": 0.85, "top": 0.60}   # base centre to the item, per grasp
BASE_RADIUS = 0.27          # TIAGo's base
STAGING = 0.50              # Nav2 stops this much further out; the rest is driven straight
ATTACH_LIB = Path("/opt/ros/humble/lib/libgazebo_model_attachment_plugin_lib.so")
ATTACH_PLUGIN = f'<plugin name="model_attachment" filename="{ATTACH_LIB.name}"/>'   # /gazebo/attach
WRIST = "arm_7_link"        # in Gazebo PAL's gripper is lumped into the last arm link
FINGER_REACH = 0.10         # PAL's fingertips reach this far beyond gripper_grasping_frame ...
PALM = 0.039                # ... and its palm, 2-14 cm behind it, this far either side of it
LIFT = 0.06                 # straight up from the grasp, then CARRY back towards the robot
CARRY = 0.08


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def grasp_candidates(x: float, y: float, z: float, yaw: float, height: float) -> list:
    """(name, orientation, grasp, pre-grasp, lift, carry, standoff) of the
    grasping frame, in the world, for an item centred at (x, y, z) with its
    handle along ``yaw`` (pointing away from the robot) on top of a body
    ``height`` tall; ``standoff`` is where the base centre stands, from the
    item."""
    top = z + height / 2
    along = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    clear = FINGER_REACH + 0.02                # a pre-grasp keeps the fingertips 2 cm off the handle
    cands = []
    # from the front, horizontal, over the handle from its near end, the palm clear of the body
    g = np.array([x, y, top + PW.FRONT_GRASP_HEIGHT]) - (PW.HANDLE[0] / 2) * along     # PW.front_grasp
    lift = g + [0.0, 0.0, LIFT]
    for name in ("front", "front_far"):      # further back, the forearm comes in flatter over a tall item
        cands.append((name, _rz(yaw), g, g - clear * along, lift, lift - CARRY * along, STANDOFF[name]))
    for name, flip in (("top", 0.0), ("top_flipped", math.pi)):
        # from above: the fingers straddle the handle across its 3 cm, the tips just clear of the body
        g = np.array([x, y, top + FINGER_REACH + 0.005])
        lift = g + [0.0, 0.0, LIFT]
        pre = g + [0.0, 0.0, PW.HANDLE[2] + 0.02]
        cands.append((name, _rz(yaw + flip) @ _ry(math.pi / 2), g, pre, lift, lift - CARRY * along, STANDOFF["top"]))
    return cands


class MissionEpisode(Episode):
    """Nav2 along ``route`` to the item's zone, then MoveIt at ``item_id``."""

    def __init__(self, spec, route: List[str], item_id: str, out: Path, index: int = 0, arm: str = "position",
                 film: bool = False, **kw):
        self.item = next(it for it in spec.items if it.id == item_id)
        self.arm = arm
        self.film = film            # a side camera saving frames of the pick (needs DISPLAY)
        if route[-1] != self.item.zone:
            raise ValueError(f"the route must end in the item's zone {self.item.zone}")
        kw.setdefault("rate", 1000.0)           # 1 ms physics steps for the grasp, in real time
        super().__init__(spec, "mobile_manipulator", route, out, index, **kw)
        self.step = 0.001
        self.moveit_log: List[dict] = []
        self.pick_record: dict = {}

    # ------------------------------------------------------------ files
    def write_files(self) -> None:
        files = TC.build(self.out / "tiago_control", arm=self.arm)
        sx, sy, sz = self.zone_xy(self.route[0])
        nx, ny, _ = self.zone_xy(self.route[1]) if len(self.route) > 1 else (sx + 1.0, sy, 0.0)
        yaw = math.atan2(ny - sy, nx - sx)
        robot_xml = TC.model_sdf(files["urdf"], sx, sy, yaw, sz)
        extra = [robot_xml]
        if self.film:
            for view, cam in self.film_cameras().items():
                extra.append(cam.sdf(self.out / f"film_{view}", rate=3.0, name=f"pick_camera_{view}"))
        built = PW.build_world(self.twin_spec(), seed=self.seed, people=self.people, extra_models=extra,
                               rate=self.rate, step=self.step,
                               world_plugins=ATTACH_PLUGIN if ATTACH_LIB.exists() else "")
        (self.out / "world.sdf").write_text(built.world)
        res = 0.05
        occ = PW.rasterize(built.static, built.bounds, res, min_height=0.12)
        occ |= ~route_mask(self.spec, self.route, self.cors, built.bounds, res)
        write_map(occ, built.bounds, self.out, res)
        rsp = files["rsp_urdf"].read_text()
        (self.out / "rsp.yaml").write_text(yaml.safe_dump({"robot_state_publisher": {"ros__parameters": {
            "robot_description": rsp, "use_sim_time": True}}}))
        from .episode import nav2_params
        nav2_params(self.robot, self.props, self.out)
        move_group_params(files["rsp_urdf"], self.out / "move_group.yaml")
        (self.out / "episode.json").write_text(json.dumps({
            "robot": self.robot, "route": self.route, "item": self.item.id, "tag": self.tag, "plant": self.spec.name,
            "item_props": {"height": self.item.height, "width": self.item.width, "mass": self.item.mass}},
            indent=1, default=str))

    def launch(self) -> None:
        super().launch()
        for c in CONTROLLERS:
            self.spawn(f"spawn_{c}", ["ros2", "run", "controller_manager", "spawner", c,
                                      "--controller-manager", "/controller_manager",
                                      "--controller-manager-timeout", "180",
                                      "--ros-args", "-p", "use_sim_time:=true"])
        # use_sim_time for every node of the process from the start (not only the one named
        # move_group): otherwise an internal node may switch to sim time late and die declaring
        # the /clock QoS parameters (a race; seen as InvalidParameterValueException)
        self.spawn("move_group", ["ros2", "run", "moveit_ros_move_group", "move_group", "--ros-args",
                                  "--params-file", str(self.out / "move_group.yaml"), "-p", "use_sim_time:=true"])

    # --------------------------------------------------------------- run
    def film_cameras(self) -> dict:
        """Cameras on the pick: one beside it, looking across the gripper, and one
        beyond the stand, looking back along the approach (the fingers' gap)."""
        from .sdf import Camera
        x, y, h, yaw, _ = PW.item_pose(self.spec, self.item)
        side = np.array([-math.sin(yaw), math.cos(yaw)])
        along = np.array([math.cos(yaw), math.sin(yaw)])
        item = np.array([x, y])

        def look(cam, cz, target, tz, hfov):
            d = target - cam
            return Camera(float(cam[0]), float(cam[1]), cz, math.atan2(cz - tz, float(np.hypot(*d))),
                          math.atan2(float(d[1]), float(d[0])), width=960, height=720, hfov=math.radians(hfov))
        return {"side": look(item - 0.7 * along + 1.7 * side, h + 0.75, item - 0.3 * along, h + 0.25, 55.0),
                "front": look(item + 0.9 * along + 0.3 * side, h + 0.45, item - 0.1 * along, h + 0.15, 50.0)}

    def pick_pose(self, standoff: float) -> Tuple[float, float, float]:
        """Where the base stands for a grasp: facing the stand, ``standoff`` from
        the item, or further if the stand's front is in the way."""
        x, y, _, yaw, front = PW.item_pose(self.spec, self.item)
        r = max(standoff, front + BASE_RADIUS + 0.10)
        return x - r * math.cos(yaw), y - r * math.sin(yaw), yaw

    @staticmethod
    def _base(tf, rclpy) -> Tuple[float, float, float]:
        tr = tf.lookup_transform("map", "base_footprint", rclpy.time.Time())
        t, q = tr.transform.translation, tr.transform.rotation
        return t.x, t.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def _drive_with(self, rclpy, PoseStamped, BasicNavigator, TaskResult, Parameter, Buffer, TransformListener) -> dict:
        # the navigation legs first (the parent records them and stops on the first failure)
        res = super()._drive_with(rclpy, PoseStamped, BasicNavigator, TaskResult, Parameter, Buffer, TransformListener)
        if res["outcome"] != "arrived":
            return res
        legs = json.loads((self.out / "summary.json").read_text())["legs_detail"]
        node = BasicNavigator("pace_mission")
        node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        tf = Buffer()
        TransformListener(tf, node)
        try:
            outcome = self._approach(node, rclpy, PoseStamped, tf)
            if outcome == "approached":
                outcome = self._pick(node, rclpy, tf)
        except Exception as e:                                        # noqa: BLE001
            self.event("moveit_error", error=repr(e)[:300])
            outcome = "error"
        finally:
            node.destroy_node()
        summary = self.summary(outcome, legs)
        summary.update({k: self.pick_record.get(k) for k in ("grasp", "finger_gap", "made_fast", "item_rise")})
        summary["moveit"] = json.dumps(self.moveit_log)
        data = json.loads((self.out / "summary.json").read_text())
        data.update(outcome=outcome, moveit=self.moveit_log, pick=self.pick_record)
        (self.out / "summary.json").write_text(json.dumps(data, indent=1))
        return summary

    def _approach(self, nav, rclpy, PoseStamped, tf) -> str:
        """Nav2 to the staging pose, then straight to the pick pose for the first grasp."""
        px, py, yaw = self.pick_pose(STANDOFF["front"])
        sx, sy = px - STAGING * math.cos(yaw), py - STAGING * math.sin(yaw)
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = nav.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = float(sx), float(sy)
        goal.pose.orientation.z, goal.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        nav.goToPose(goal)
        self.event("staging", x=round(sx, 2), y=round(sy, 2))
        t0 = time.time()
        while not nav.isTaskComplete():
            rclpy.spin_once(nav, timeout_sec=0.1)
            if time.time() - t0 > 120.0:
                nav.cancelTask()
                break
        t1 = time.time()
        while time.time() - t1 < 1.5:                # until the velocity smoother has fallen silent
            rclpy.spin_once(nav, timeout_sec=0.1)
        d, dyaw = self._drive_to(nav, rclpy, tf, px, py, yaw)
        self.event("approach_end", dist=round(d, 3), yaw_err=round(dyaw, 3))
        return "approached" if d < 0.05 and abs(dyaw) < 0.08 else "approach_failed"

    def _drive_to(self, node, rclpy, tf, gx: float, gy: float, gyaw: float, timeout: float = 60.0):
        """The last half-metre on ``cmd_vel``: turn to the point, drive to it
        (backwards if it is behind), turn to the heading."""
        from geometry_msgs.msg import Twist
        pub = node.create_publisher(Twist, "cmd_vel", 10)
        d, dyaw, there, t0 = 9.9, 9.9, False, time.time()
        try:
            while time.time() - t0 < timeout:
                rclpy.spin_once(node, timeout_sec=0.05)
                try:
                    x, y, th = self._base(tf, rclpy)
                except Exception:                                     # noqa: BLE001
                    continue
                d, dyaw = math.hypot(gx - x, gy - y), _wrap(gyaw - th)
                there = d < (0.05 if there else 0.02)
                cmd = Twist()
                if not there:
                    err, sign = _wrap(math.atan2(gy - y, gx - x) - th), 1.0
                    if abs(err) > math.pi / 2:
                        err, sign = _wrap(err + math.pi), -1.0
                    cmd.angular.z = max(-0.5, min(0.5, 2.0 * err))
                    if abs(err) < 0.2:
                        cmd.linear.x = sign * max(0.03, min(0.15, 0.8 * d))
                elif abs(dyaw) > 0.02:
                    cmd.angular.z = math.copysign(max(0.08, min(0.4, 1.5 * abs(dyaw))), dyaw)
                else:
                    break
                pub.publish(cmd)
        finally:
            pub.publish(Twist())
            node.destroy_publisher(pub)
        return d, dyaw

    def _pick(self, node, rclpy, tf) -> str:
        """The grasp candidates in turn; the first that reaches its grasp pose closes and lifts."""
        from rclpy.action import ActionClient
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTrajectoryControllerState
        from gazebo_msgs.srv import GetEntityState
        from geometry_msgs.msg import Pose
        from moveit_msgs.action import ExecuteTrajectory, MoveGroup
        from moveit_msgs.msg import (AttachedCollisionObject, BoundingVolume, CollisionObject, Constraints,
                                     OrientationConstraint, PlanningScene, PositionConstraint)
        from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
        from sensor_msgs.msg import JointState
        from shape_msgs.msg import SolidPrimitive
        from trajectory_msgs.msg import JointTrajectoryPoint

        move = ActionClient(node, MoveGroup, "/move_action")
        execute = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")
        grip = ActionClient(node, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory")
        cartesian = node.create_client(GetCartesianPath, "/compute_cartesian_path")
        apply_scene = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        entity = node.create_client(GetEntityState, "/gazebo/get_entity_state")
        fingers: dict = {}
        track: list = []                           # arm, torso and fingers during the pick, at 20 Hz

        def on_joints(m):
            fingers.update({n: p for n, p in zip(m.name, m.position) if n in TC.GRIPPER})
            if TC.ARM[0] in m.name and (not track or time.time() - track[-1]["t"] >= 0.05):
                track.append({"t": time.time(), "q": {n: round(p, 4) for n, p in zip(m.name, m.position)},
                              "v": {n: round(v, 3) for n, v in zip(m.name, m.velocity)}})
        node.create_subscription(JointState, "joint_states", on_joints, 10)
        grip_state: list = []                      # the gripper controller's reference and effort, at 10 Hz

        def on_grip(m):
            if not grip_state or time.time() - grip_state[-1]["t"] >= 0.1:
                grip_state.append({"t": time.time(), "ref": [round(v, 4) for v in m.reference.positions],
                                   "err": [round(v, 4) for v in m.error.positions],
                                   "effort": [round(v, 2) for v in m.output.effort]})
        node.create_subscription(JointTrajectoryControllerState, "/gripper_controller/controller_state", on_grip, 10)
        if not move.wait_for_server(timeout_sec=120.0) or not execute.wait_for_server(timeout_sec=20.0):
            self.event("moveit_missing")
            return "error"

        def wait(seconds):
            t = time.time()
            while time.time() - t < seconds:
                rclpy.spin_once(node, timeout_sec=0.05)

        def call(client, req, timeout=30.0):
            if not client.wait_for_service(timeout_sec=timeout):
                return None
            fut = client.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
            return fut.result()

        def run_action(client, goal, timeout):
            fut = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=30.0)
            gh = fut.result()
            if gh is None or not gh.accepted:
                return None
            rf = gh.get_result_async()
            rclpy.spin_until_future_complete(node, rf, timeout_sec=timeout)
            return rf.result()

        def item_state():
            # once made fast the item may be nested in the robot's model: try its scoped names too
            for name in (f"item_{self.item.id}", f"item_{self.item.id}::link",
                         f"robot::item_{self.item.id}", f"robot::item_{self.item.id}::link"):
                r = call(entity, GetEntityState.Request(name=name, reference_frame="world"))
                if r is not None and r.success:
                    p, q = r.state.pose.position, r.state.pose.orientation
                    return p.x, p.y, p.z, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            return None

        def make_fast() -> bool:
            """The item fixed to the wrist (a joint the attachment plugin adds), once gripped."""
            try:
                from gazebo_model_attachment_plugin_msgs.srv import Attach
            except ImportError:
                return False
            from std_srvs.srv import Empty
            attach = node.create_client(Attach, "/gazebo/attach")
            pause, unpause = (node.create_client(Empty, "/pause_physics"), node.create_client(Empty, "/unpause_physics"))
            # the plugin changes the model from a ROS callback and hangs if the physics is stepping
            call(pause, Empty.Request(), 10.0)
            r = call(attach, Attach.Request(joint_name="grasp", model_name_1="robot", link_name_1=WRIST,
                                            model_name_2=f"item_{self.item.id}", link_name_2="link"), 20.0)
            call(unpause, Empty.Request(), 10.0)
            ok = r is not None and r.success
            self.event("made_fast", ok=ok, message=getattr(r, "message", None))
            return ok

        base: dict = {}

        def pose_msg(xyz, R) -> Pose:
            """A world pose in the planning frame (the base stands still during a grasp)."""
            Rb = _rz(-base["th"])
            v = Rb @ (np.asarray(xyz, float) - [base["x"], base["y"], 0.0])
            p = Pose()
            p.position.x, p.position.y, p.position.z = (float(c) for c in v)
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = _quat(Rb @ R)
            return p

        start = item_state()
        if start is None:
            self.event("item_missing")
            return "error"
        ix, iy, iz, iyaw = start
        w, d, mh = PW.TOTE if self.item.kind == "tote" else PW.MODULE
        nx, ny, nh, _, _ = PW.item_pose(self.spec, self.item)
        self.pick_record = {"item_start": [round(v, 3) for v in start],
                            "item_nominal": [round(nx, 3), round(ny, 3), round(nh + mh / 2, 3)]}

        def diag(label) -> None:
            """Where Gazebo has the base, wrist and a finger, and where TF (the URDF on
            the joint states, which MoveIt plans with) has them."""
            rec = {"base_yaw": round(self._base(tf, rclpy)[2], 4)}
            for link in ("base_footprint", "arm_7_link", "gripper_left_finger_link"):
                r = call(entity, GetEntityState.Request(name="robot" if link == "base_footprint" else f"robot::{link}",
                                                        reference_frame="world"))
                p = r.state.pose.position if r is not None and r.success else None
                try:
                    t = tf.lookup_transform("map", link, rclpy.time.Time()).transform.translation
                    tp = [round(t.x, 3), round(t.y, 3), round(t.z, 3)]
                except Exception:                                     # noqa: BLE001
                    tp = None
                rec[link] = {"gazebo": [round(p.x, 3), round(p.y, 3), round(p.z, 3)] if p else None, "tf": tp}
            self.pick_record[f"where_{label}"] = rec

        def rebase() -> None:
            """The planning frame is the base's, and the base gives a little (it slid ~1 cm and
            turned ~2 deg under the arm swinging out): re-read it before every move."""
            base.update(zip(("x", "y", "th"), self._base(tf, rclpy)))

        def stand_at(standoff) -> None:
            """The base to ``standoff`` from the item, then the scene."""
            self._drive_to(node, rclpy, tf, *self.pick_pose(standoff))
            place_scene(standoff)

        def place_scene(standoff=None) -> None:
            """The stand and the item's body (its handle stays free for the fingers) into
            MoveIt's scene, in the base's frame as it is now."""
            rebase()
            scene = PlanningScene(is_diff=True)
            # the item's body a little smaller than it is: resting on the stand, it would otherwise
            # count as a collision once attached to the gripper
            for name, x, y, z, dx, dy, dz, yaw in PW.stand_boxes(self.spec, self.item) + [
                    ("item_body", ix, iy, iz + 0.002, w - 0.004, d - 0.004, mh - 0.004, iyaw)]:
                co = CollisionObject()
                co.header.frame_id = PLANNING_FRAME
                co.id = name
                co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX,
                                                    dimensions=[float(dx), float(dy), float(dz)]))
                co.primitive_poses.append(pose_msg((x, y, z), _rz(yaw)))
                co.operation = CollisionObject.ADD
                scene.world.collision_objects.append(co)
            r = call(apply_scene, ApplyPlanningScene.Request(scene=scene))
            self.event("scene", standoff=standoff, base=[round(base[k], 4) for k in ("x", "y", "th")],
                       ok=bool(r is not None and r.success))

        def log(label, ok, t0, **kw):
            self.moveit_log.append({"step": label, "ok": ok, "seconds": round(time.time() - t0, 1), **kw})
            self.event("moveit", step=label, ok=ok, **kw)
            return ok

        def tracking(traj) -> float:
            """How far (rad or m, worst joint) the arm settled from the end of a trajectory."""
            wait(1.0)
            if traj is None or not traj.joint_trajectory.points or not track:
                return float("nan")
            end = dict(zip(traj.joint_trajectory.joint_names, traj.joint_trajectory.points[-1].positions))
            now = track[-1]["q"]
            return round(max(abs(now[j] - v) for j, v in end.items() if j in now), 4)

        def plan_to(label, xyz, R, group=GROUP, angle=0.15) -> bool:
            c = Constraints()
            pc = PositionConstraint()
            pc.header.frame_id = PLANNING_FRAME
            pc.link_name = GRASP_FRAME
            pc.weight = 1.0
            bv = BoundingVolume()
            bv.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01]))
            bv.primitive_poses.append(pose_msg(xyz, np.eye(3)))
            pc.constraint_region = bv
            oc = OrientationConstraint()
            oc.header.frame_id = PLANNING_FRAME
            oc.link_name = GRASP_FRAME
            oc.orientation = pose_msg(xyz, R).orientation
            oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = angle
            oc.weight = 1.0
            c.position_constraints.append(pc)
            c.orientation_constraints.append(oc)
            goal = MoveGroup.Goal()
            goal.request.group_name = group
            goal.request.num_planning_attempts = 3
            goal.request.allowed_planning_time = 12.0
            goal.request.max_velocity_scaling_factor = 0.3
            goal.request.max_acceleration_scaling_factor = 0.3
            goal.request.goal_constraints.append(c)
            goal.request.start_state.is_diff = True
            goal.planning_options.plan_only = False
            t0 = time.time()
            for attempt in range(3):                    # OMPL's smoothed path may clip an obstacle: try again
                res = run_action(move, goal, 240.0)
                code = res.result.error_code.val if res is not None else None
                if code != -2:                          # PLANNING_FAILED; no IK (99999) is not retried
                    break
            err = tracking(res.result.planned_trajectory) if code == 1 else None
            return log(label, code == 1, t0, code=code, settled_off=err, attempts=attempt + 1)

        def tuck(label) -> bool:
            """Arm and torso back to the tucked posture (a joint-space goal)."""
            from moveit_msgs.msg import JointConstraint
            c = Constraints()
            for j in TC.TORSO + TC.ARM:
                c.joint_constraints.append(JointConstraint(joint_name=j, position=float(TC.TUCKED[j]),
                                                           tolerance_above=0.01, tolerance_below=0.01, weight=1.0))
            goal = MoveGroup.Goal()
            goal.request.group_name = GROUP
            goal.request.num_planning_attempts = 3
            goal.request.allowed_planning_time = 8.0
            goal.request.max_velocity_scaling_factor = 0.3
            goal.request.max_acceleration_scaling_factor = 0.3
            goal.request.goal_constraints.append(c)
            goal.request.start_state.is_diff = True
            goal.planning_options.plan_only = False
            t0 = time.time()
            res = run_action(move, goal, 240.0)
            code = res.result.error_code.val if res is not None else None
            return log(label, code == 1, t0, code=code)

        def straight(label, xyz, R, slow=3.0) -> bool:
            """A straight line of the grasping frame by the arm alone (the torso stays
            put), executed at a third of full speed."""
            req = GetCartesianPath.Request()
            req.header.frame_id = PLANNING_FRAME
            req.start_state.is_diff = True
            req.group_name = ARM_GROUP
            req.link_name = GRASP_FRAME
            req.waypoints = [pose_msg(xyz, R)]
            req.max_step = 0.005
            req.jump_threshold = 0.0
            req.avoid_collisions = True
            t0 = time.time()
            r = call(cartesian, req)
            frac = float(r.fraction) if r is not None else 0.0
            if frac < 0.97:
                return log(label, False, t0, fraction=round(frac, 3))
            traj = r.solution
            for pt in traj.joint_trajectory.points:
                t = (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9) * slow
                pt.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
                pt.velocities = [v / slow for v in pt.velocities]
                pt.accelerations = [a / slow ** 2 for a in pt.accelerations]
            res = run_action(execute, ExecuteTrajectory.Goal(trajectory=traj), 90.0)
            code = res.result.error_code.val if res is not None else None
            return log(label, code == 1, t0, fraction=round(frac, 3), code=code,
                       settled_off=tracking(traj) if code == 1 else None)

        def gripper(label, width, squeeze=True) -> None:
            """Open or close the fingers, as a goal that stays live for minutes: once a goal
            succeeds, the trajectory controller (Humble) holds the fingers where they *are* --
            against a handle, no force at all; knocked by the arm swinging, not open.  While it
            is live they keep pushing to ``width`` with up to the URDF's 16 N."""
            t0 = time.time()
            ok = grip.wait_for_server(timeout_sec=20.0)
            if ok:
                g = FollowJointTrajectory.Goal()
                g.trajectory.joint_names = list(TC.GRIPPER)
                for sec in ((1, 600) if squeeze else (1,)):
                    pt = JointTrajectoryPoint()
                    pt.positions = [float(width)] * len(TC.GRIPPER)
                    pt.time_from_start = Duration(sec=sec, nanosec=500000000)
                    g.trajectory.points.append(pt)
                if squeeze:
                    fut = grip.send_goal_async(g)
                    rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
                    ok = fut.result() is not None and fut.result().accepted
                    wait(2.5)
                else:
                    ok = run_action(grip, g, 20.0) is not None
            wait(1.0)
            log(label, ok, t0, fingers={k: round(v, 4) for k, v in fingers.items()})

        gripper("open", TC.GRIPPER_OPEN)
        reached, here = None, None
        for name, R, g, pre, lift, carry, standoff in grasp_candidates(ix, iy, iz, iyaw, mh):
            if standoff != here:
                stand_at(standoff)
                here = standoff
            for attempt in range(3):
                # the pre-grasp's arm configuration is one of several; in some the forearm meets
                # the item on the way in -- then back to the tucked posture and plan it afresh
                if attempt:
                    tuck(f"{name}:tuck")
                if not plan_to(f"{name}:pre_grasp", pre, R):
                    break
                gripper(f"{name}:open", TC.GRIPPER_OPEN)
                diag(f"{name}_pre")
                place_scene()
                if straight(f"{name}:grasp", g, R):
                    diag(f"{name}_grasp")
                    reached = (name, R, lift, carry)
                    break
            if reached is not None:
                break
        (self.out / "joints.json").write_text(json.dumps({"joints": track, "gripper": grip_state}))
        if reached is None:
            return "unreachable"
        name, R, lift, carry = reached
        self.pick_record.update(grasp=name, base=[round(base[k], 3) for k in ("x", "y", "th")])
        at_grasp = item_state()
        gripper(f"{name}:close", 0.0)
        held = min(fingers.get(j, 0.0) for j in TC.GRIPPER)
        gap = sum(fingers.get(j, 0.0) for j in TC.GRIPPER)
        on_handle = held > 0.004 and abs(gap - PW.HANDLE[1]) < 0.006
        closed = item_state()
        self.pick_record.update(finger_gap=round(gap, 4), on_handle=on_handle,
                                item_at_grasp=[round(v, 3) for v in at_grasp] if at_grasp else None,
                                item_closed=[round(v, 3) for v in closed] if closed else None)
        if not on_handle:
            (self.out / "joints.json").write_text(json.dumps({"joints": track, "gripper": grip_state}))
            return "missed"
        # the item's link frame is where the wrist is in a front grasp (PW.item_anchor)
        self.pick_record["made_fast"] = made_fast = (make_fast() if ATTACH_LIB.exists() and name.startswith("front")
                                                     else False)
        wait(2.0)                                     # the arm settles on its new load; MoveIt's state catches up
        diag("after_attach")
        # the item now moves with the gripper: MoveIt plans the lift with it attached, by the arm
        # alone (the torso stays put), in a slow straight line -- 3 cm where 6 are out of reach
        att = AttachedCollisionObject()
        att.link_name = "gripper_link"
        att.object.id = "item_body"
        att.object.header.frame_id = PLANNING_FRAME
        att.object.operation = CollisionObject.ADD
        att.touch_links = ["gripper_link", "gripper_left_finger_link", "gripper_right_finger_link",
                           "gripper_grasping_frame"]
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(att)
        r = call(apply_scene, ApplyPlanningScene.Request(scene=scene))
        self.event("attach", ok=bool(r is not None and r.success))
        height = LIFT
        if not made_fast:
            rebase()
        # (once the part is fast, TIAGo's model includes it and its odometry pose moves with the
        # combined model: the base frame stays as it was read before -- the base does not move)
        if not straight(f"{name}:lift", lift, R):
            height = LIFT / 2
            if not straight(f"{name}:lift_half", lift - [0.0, 0.0, LIFT / 2], R):
                (self.out / "joints.json").write_text(json.dumps({"joints": track, "gripper": grip_state}))
                return "grasped"
        wait(2.0)
        end = item_state()
        rise = end[2] - iz if end is not None else 0.0
        self.pick_record.update(item_end=[round(v, 3) for v in end] if end else None, item_rise=round(rise, 3),
                                lift=height)
        if rise > 0.6 * height:                       # then carry it back, for the record
            straight(f"{name}:carry", carry - [0.0, 0.0, LIFT - height], R)
            wait(2.0)
            after = item_state()
            self.pick_record["item_carried"] = [round(v, 3) for v in after] if after else None
        (self.out / "joints.json").write_text(json.dumps({"joints": track, "gripper": grip_state}))
        return "picked" if rise > 0.6 * height else "grasped"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, help="comma-separated zone ids, ending in the item's zone")
    ap.add_argument("--item", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--changes", default="", help="individual:property=value;...")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--no-people", action="store_true")
    ap.add_argument("--arm", choices=("position", "velocity"), default="position",
                    help="how ros2_control commands the arm")
    ap.add_argument("--film", action="store_true", help="save frames of the pick from a side camera (needs DISPLAY)")
    a = ap.parse_args(argv)
    spec = plant_spec()
    if a.changes:
        changes = []
        for item in a.changes.split(";"):
            sid, rest = item.split(":")
            prop, val = rest.split("=")
            changes.append((sid, prop, yaml.safe_load(val)))
        spec = PW.apply_changes(spec, changes)
    ep = MissionEpisode(spec, a.route.split(","), a.item, Path(a.out), a.index, arm=a.arm, film=a.film,
                        timeout=a.timeout, people=not a.no_people)
    print(json.dumps(ep.run()))


if __name__ == "__main__":
    main()
