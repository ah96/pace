"""One episode in the twin: a robot drives a questioned route with Nav2, in Gazebo.

The episode answers the question an explanation makes a claim about: can
*this* robot get from A to B along *this* route in *this* state of the plant?
It is run twice per repair, on the plant as it is and on the plant with PACE's
change applied, so that "blocked before, completes after" is measured rather
than asserted.

What happens
------------
1. **World.** The plant of the spec (:mod:`pace.twin.plant_world`), with the
   robot's Gazebo model (:mod:`pace.twin.robots`) standing in the start zone,
   facing the first waypoint.
2. **Map.** The robot's prior map is the static geometry of that world, with
   everything outside the questioned route's passages and zones marked
   occupied, so Nav2 can only take the route being asked about: if the route
   is refused, it is refused because of the route.
3. **Nav2.** map server, planner (NavFn), controller (Regulated Pure Pursuit
   with footprint collision checking), the usual bringup; odometry is the
   world pose and ``map -> odom`` the identity.
4. **Drive.** One ``NavigateToPose`` per leg of the route, zone to zone. The
   outcome of the route is the outcome of its first failing leg: ``arrived``,
   ``aborted`` (Nav2 gave up: no plan, or the controller found no safe
   velocity), ``stuck`` (the robot stopped moving with the goal outstanding,
   the physical case: something the 2-D map cannot see stopped it), or
   ``timeout``.
5. **Record.** Every leg's outcome and time, the track, where the robot came
   to rest and which passage that is, in ``summary.json``.

Each episode has its own ROS domain and Gazebo port and every child process is
in its own session, so episodes run in parallel and a killed episode still
cleans up.

    python -m pace.twin.episode --robot forklift_agv --route dock,receiving,diagnostics --out build/ep
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from scenarios.plant import ROBOTS, ZONE_HALF, PlantSpec, plant_spec
from . import plant_world as PW
from . import robots as RB
from .render import gazebo_env

PARAMS_BASE = Path(__file__).with_name("nav2_params_base.yaml")
OUTCOMES = ("arrived", "aborted", "stuck", "timeout", "error", "interrupted")


# ------------------------------------------------------------- the files

def route_mask(spec: PlantSpec, route: Sequence[str], cors: Dict[str, PW.Corridor],
               bounds, resolution: float, margin: float = 0.0) -> np.ndarray:
    """True where the robot may be: the route's zones and passages."""
    x0, y0, x1, y1 = bounds
    W, H = int(math.ceil((x1 - x0) / resolution)), int(math.ceil((y1 - y0) / resolution))
    ok = np.zeros((H, W), dtype=bool)

    def fill(r: PW.Rect):
        i0, i1 = int((r.y0 - margin - y0) / resolution), int(math.ceil((r.y1 + margin - y0) / resolution))
        j0, j1 = int((r.x0 - margin - x0) / resolution), int(math.ceil((r.x1 + margin - x0) / resolution))
        ok[max(0, i0):min(H, i1), max(0, j0):min(W, j1)] = True

    for zid in route:
        z = spec.zone(zid)
        fill(PW.Rect(z.x - ZONE_HALF, z.y - ZONE_HALF, z.x + ZONE_HALF, z.y + ZONE_HALF))
    for a, b in zip(route, route[1:]):
        for c in cors.values():
            if {c.seg.a, c.seg.b} == {a, b}:
                fill(c.rect)
    return ok


def write_map(occ: np.ndarray, bounds, out: Path, resolution: float) -> None:
    img = np.where(occ, 0, 254).astype(np.uint8)[::-1]
    with open(out / "map.pgm", "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())
    (out / "map.yaml").write_text(yaml.safe_dump({
        "image": "map.pgm", "resolution": resolution, "origin": [float(bounds[0]), float(bounds[1]), 0.0],
        "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196, "mode": "trinary"}))


def nav2_params(robot: str, props: Dict[str, object], out: Path, controller: str = "rpp") -> Path:
    """Nav2's parameters for this robot: its footprint (or TIAGo's radius),
    its speed, and one controller (from nav2_params_base.yaml)."""
    P = yaml.safe_load(PARAMS_BASE.read_text())
    v_max = min(float(props.get("max_speed", 1.0)), 1.0)
    w_max = 1.0 if robot != "forklift_agv" else 0.6
    length = float(props.get("length", 1.0))
    if robot == "mobile_manipulator":
        radius, footprint = 0.275, None
    else:
        radius, footprint = None, RB.nav2_footprint(props)
    # inflation from the half-length for a long vehicle: the planner then keeps
    # the path away from the corners it would otherwise cut when turning into
    # an aisle (inflation raises cost, only the inscribed radius is lethal)
    infl = 0.35 + (max(float(props["robot_width"]), 0.6 * length) / 2 if footprint else radius)

    def walk(d):
        if isinstance(d, dict):
            for k, v in list(d.items()):
                if k == "robot_base_frame":
                    d[k] = "base_footprint"
                elif k == "robot_radius":
                    if footprint:
                        del d[k]
                        d["footprint"] = footprint
                        d["footprint_padding"] = 0.02
                    else:
                        d[k] = radius
                elif k == "inflation_radius":
                    d[k] = round(infl, 3)
                else:
                    walk(v)
    walk(P)
    cs = P["controller_server"]["ros__parameters"]
    cs["FollowPath"] = copy.deepcopy(P["controllers"][controller])
    fp = cs["FollowPath"]
    for key, val in {"desired_linear_vel": v_max, "max_vel_x": v_max, "max_speed_xy": v_max,
                     "vx_max": v_max, "max_vel_theta": w_max, "wz_max": w_max,
                     "rotate_to_heading_angular_vel": min(w_max, 0.8)}.items():
        if key in fp:
            fp[key] = val
    if controller == "rpp":
        fp["lookahead_dist"] = round(max(0.6, 0.35 * length), 2)
        fp["min_lookahead_dist"] = round(max(0.3, 0.25 * length), 2)
        fp["max_lookahead_dist"] = round(max(0.9, 0.5 * length), 2)
        # The controller's own collision veto treats every cell within the
        # footprint's inscribed radius of an obstacle as a collision, which for
        # a 1.2 m body refuses any corridor under 2.4 m whatever physics says.
        # The twin is there to let physics say it: the planner keeps the path
        # centred (inflation), the contact solver decides whether the body
        # gets through.
        fp["use_collision_detection"] = False
    cs["progress_checker"]["movement_time_allowance"] = 15.0
    # the mission cares where the robot gets to, not which way it faces there
    cs["general_goal_checker"]["yaw_goal_tolerance"] = 3.15
    cs["general_goal_checker"]["xy_goal_tolerance"] = 0.35
    cs["progress_checker"]["required_movement_radius"] = 0.3
    del P["controllers"]
    vs = P["velocity_smoother"]["ros__parameters"]
    vs["max_velocity"] = [v_max, 0.0, w_max]
    vs["min_velocity"] = [-v_max, 0.0, -w_max]
    # the local costmap must see the whole robot plus a margin
    lc = P["local_costmap"]["local_costmap"]["ros__parameters"]
    lc["width"] = lc["height"] = int(math.ceil(2 * length + 4))
    # the planner may be strict: no unknown space, no tolerance
    P["planner_server"]["ros__parameters"]["GridBased"]["tolerance"] = 0.5
    path = out / "nav2_params.yaml"
    path.write_text(yaml.safe_dump(P, sort_keys=False))
    return path


# ------------------------------------------------------------- the episode

@dataclass
class Episode:
    spec: PlantSpec
    robot: str
    route: List[str]
    out: Path
    index: int = 0
    timeout: float = 420.0
    leg_timeout: float = 150.0
    rate: float = 250.0
    seed: int = 0
    stuck_after: float = 10.0
    people: bool = True
    tag: str = ""
    procs: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    def __post_init__(self):
        self.out = Path(self.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.props = dict(ROBOTS[self.robot])
        self.domain = 1 + self.index % 29
        self.gz_port = 11345 + 1 + self.index % 50
        self.cors = PW.corridors(self.spec)

    # ----------------------------------------------------------- helpers
    def env(self) -> dict:
        # Fast DDS without shared memory: with it on, a node was seen to stop
        # receiving TF at a random moment and Nav2 then drove blind
        # (pace/twin/fastdds_udp_only.xml)
        return gazebo_env(ROS_DOMAIN_ID=self.domain, ROS_LOCALHOST_ONLY="1",
                          GAZEBO_MASTER_URI=f"http://localhost:{self.gz_port}",
                          FASTRTPS_DEFAULT_PROFILES_FILE=str(Path(__file__).with_name("fastdds_udp_only.xml")))

    def spawn(self, name: str, cmd: list) -> None:
        log = open(self.out / f"{name}.log", "w")
        self.procs[name] = subprocess.Popen(cmd, env=self.env(), stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(self.out))
        log.close()

    def stop_all(self) -> None:
        for name, p in reversed(list(self.procs.items())):
            for sig, wait in ((signal.SIGINT, 8.0), (signal.SIGTERM, 4.0), (signal.SIGKILL, 4.0)):
                if p.poll() is not None:
                    break
                try:
                    os.killpg(p.pid, sig)
                    p.wait(wait)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        self.procs.clear()

    def event(self, kind: str, **kw) -> None:
        self.events.append({"event": kind, "t_wall": round(time.time() - self.t0, 2), **kw})

    def zone_xy(self, zid: str) -> Tuple[float, float, float]:
        z = self.spec.zone(zid)
        return z.x, z.y, z.level

    def where(self, x: float, y: float) -> str:
        """The zone or passage a point lies in (nearest, if none contains it)."""
        best, dist = "?", 1e9
        for z in self.spec.zones:
            d = max(abs(x - z.x), abs(y - z.y)) - ZONE_HALF
            if d < dist:
                best, dist = z.id, d
        for c in self.cors.values():
            r = c.rect
            d = max(r.x0 - x, x - r.x1, r.y0 - y, y - r.y1)
            if d < dist:
                best, dist = c.seg.id, d
        return best

    # ------------------------------------------------------------ launch
    def twin_spec(self) -> PlantSpec:
        """The plant as this robot meets it: a robot that operates doors finds
        every closed (not locked) door open, which is what \textsc{Openable}
        stands for and what the twin has no arm-and-button routine to act out."""
        if not self.props.get("can_open_doors", False):
            return self.spec
        changes = [(seg.id, "door_state", "open") for seg in self.spec.segments
                   if seg.props.get("door_state") == "closed"]
        return PW.apply_changes(self.spec, changes) if changes else self.spec

    def write_files(self) -> None:
        sx, sy, sz = self.zone_xy(self.route[0])
        nx, ny, _ = self.zone_xy(self.route[1])
        yaw = math.atan2(ny - sy, nx - sx)
        robot_xml = RB.robot_model(self.robot, self.props, sx, sy, yaw, sz)
        built = PW.build_world(self.twin_spec(), seed=self.seed, people=self.people, extra_models=[robot_xml],
                               rate=self.rate)
        (self.out / "world.sdf").write_text(built.world)
        res = 0.05
        occ = PW.rasterize(built.static, built.bounds, res, min_height=0.12)
        occ |= ~route_mask(self.spec, self.route, self.cors, built.bounds, res)
        write_map(occ, built.bounds, self.out, res)
        (self.out / "rsp.yaml").write_text(yaml.safe_dump({"robot_state_publisher": {"ros__parameters": {
            "robot_description": RB.robot_urdf(self.robot, self.props), "use_sim_time": True}}}))
        nav2_params(self.robot, self.props, self.out)
        (self.out / "episode.json").write_text(json.dumps({
            "robot": self.robot, "route": self.route, "tag": self.tag, "plant": self.spec.name,
            "segments": {s.id: s.props for s in self.spec.segments}}, indent=1, default=str))

    def launch(self) -> None:
        self.write_files()
        self.spawn("gzserver", ["gzserver", "--verbose", "-s", "libgazebo_ros_init.so", str(self.out / "world.sdf")])
        self.spawn("rsp", ["ros2", "run", "robot_state_publisher", "robot_state_publisher", "--ros-args",
                           "--params-file", str(self.out / "rsp.yaml")])
        self.spawn("map_to_odom", ["ros2", "run", "tf2_ros", "static_transform_publisher", "--frame-id", "map",
                                   "--child-frame-id", "odom", "--ros-args", "-p", "use_sim_time:=true"])
        self.spawn("map_server", ["ros2", "run", "nav2_map_server", "map_server", "--ros-args",
                                  "-p", f"yaml_filename:={self.out / 'map.yaml'}", "-p", "use_sim_time:=true"])
        self.spawn("map_lifecycle", ["ros2", "run", "nav2_lifecycle_manager", "lifecycle_manager", "--ros-args",
                                     "-r", "__node:=lifecycle_manager_map", "-p", "node_names:=['map_server']",
                                     "-p", "autostart:=true", "-p", "use_sim_time:=true"])
        self.spawn("nav2", ["ros2", "launch", "nav2_bringup", "navigation_launch.py", "use_sim_time:=True",
                            f"params_file:={self.out / 'nav2_params.yaml'}", "autostart:=True",
                            "use_composition:=False"])

    @staticmethod
    def wait_active(node, names, budget: float) -> bool:
        import rclpy
        from lifecycle_msgs.srv import GetState
        deadline = time.time() + budget
        for name in names:
            cli = node.create_client(GetState, f"/{name}/get_state")
            while time.time() < deadline:
                if cli.wait_for_service(timeout_sec=1.0):
                    fut = cli.call_async(GetState.Request())
                    rclpy.spin_until_future_complete(node, fut, timeout_sec=2.0)
                    if fut.done() and fut.result() is not None and fut.result().current_state.id == 3:
                        break
                rclpy.spin_once(node, timeout_sec=0.5)
            else:
                return False
        return True

    # --------------------------------------------------------------- run
    def run(self) -> dict:
        os.environ["ROS_DOMAIN_ID"] = str(self.domain)
        os.environ["ROS_LOCALHOST_ONLY"] = "1"
        os.environ.setdefault("RCUTILS_COLORIZED_OUTPUT", "0")
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(Path(__file__).with_name("fastdds_udp_only.xml"))

        self._running = True

        def stop(signum, frame):
            if self._running:
                raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        self.t0 = time.time()
        try:
            self.launch()
            return self._drive()
        except KeyboardInterrupt:
            self.event("interrupted")
            return self.summary("interrupted", [])
        except Exception as e:                                # noqa: BLE001
            self.event("error", error=repr(e)[:400])
            return self.summary("error", [])
        finally:
            self._running = False
            self.stop_all()

    def _drive(self) -> dict:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
        from rclpy.parameter import Parameter
        from tf2_ros import Buffer, TransformListener

        rclpy.init()
        try:
            return self._drive_with(rclpy, PoseStamped, BasicNavigator, TaskResult, Parameter, Buffer, TransformListener)
        finally:
            rclpy.try_shutdown()

    def _drive_with(self, rclpy, PoseStamped, BasicNavigator, TaskResult, Parameter, Buffer, TransformListener) -> dict:
        nav = BasicNavigator("pace_episode")
        nav.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        tf = Buffer()
        TransformListener(tf, nav)
        if not self.wait_active(nav, ("map_server", "planner_server", "controller_server", "bt_navigator"), 150.0):
            nav.destroy_node()
            raise RuntimeError("Nav2 did not come up (see nav2.log)")
        self.event("nav2_active")

        def pose():
            try:
                tr = tf.lookup_transform("map", "base_footprint", rclpy.time.Time())
                return np.array([tr.transform.translation.x, tr.transform.translation.y])
            except Exception:                                  # noqa: BLE001
                return None

        # let TF settle
        t_wait = time.time()
        while pose() is None and time.time() - t_wait < 30.0:
            rclpy.spin_once(nav, timeout_sec=0.2)
        track: List[np.ndarray] = []
        legs: List[dict] = []
        outcome = "arrived"
        for k, (a, b) in enumerate(zip(self.route, self.route[1:])):
            gx, gy, _ = self.zone_xy(b)
            nxt = self.route[k + 2] if k + 2 < len(self.route) else None
            if nxt:
                hx, hy, _ = self.zone_xy(nxt)
                heading = math.atan2(hy - gy, hx - gx)
            else:
                ax, ay, _ = self.zone_xy(a)
                heading = math.atan2(gy - ay, gx - ax)
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = nav.get_clock().now().to_msg()
            goal.pose.position.x, goal.pose.position.y = float(gx), float(gy)
            goal.pose.orientation.z, goal.pose.orientation.w = math.sin(heading / 2), math.cos(heading / 2)
            nav.clearAllCostmaps()
            nav.goToPose(goal)
            t_leg = time.time()
            self.event("leg", k=k, frm=a, to=b)
            last_move, last_xy = time.time(), pose()
            leg_out = "timeout"
            while time.time() - t_leg < self.leg_timeout and time.time() - self.t0 < self.timeout:
                rclpy.spin_once(nav, timeout_sec=0.1)
                xy = pose()
                if xy is not None:
                    if not track or np.linalg.norm(xy - track[-1]) > 0.05:
                        track.append(xy)
                    if last_xy is None or np.linalg.norm(xy - last_xy) > 0.03:
                        last_move, last_xy = time.time(), xy
                if nav.isTaskComplete():
                    res = nav.getResult()
                    leg_out = {TaskResult.SUCCEEDED: "arrived", TaskResult.FAILED: "aborted",
                               TaskResult.CANCELED: "canceled"}.get(res, str(res))
                    break
                if time.time() - last_move > self.stuck_after and time.time() - t_leg > 20.0:
                    nav.cancelTask()
                    leg_out = "stuck"
                    break
            xy = pose()
            at = self.where(*xy) if xy is not None else "?"
            dist_goal = float(np.linalg.norm(xy - np.array([gx, gy]))) if xy is not None else None
            if leg_out == "arrived" and dist_goal is not None and dist_goal > 1.5:
                leg_out = "aborted"                            # Nav2 declared success far from the goal
            if leg_out != "arrived" and dist_goal is not None and dist_goal <= 0.6:
                leg_out = "arrived"                            # it got there, whatever Nav2 concluded
            legs.append({"k": k, "from": a, "to": b, "outcome": leg_out, "seconds": round(time.time() - t_leg, 1),
                         "at": at, "xy": [round(float(v), 2) for v in xy] if xy is not None else None,
                         "dist_goal": round(dist_goal, 2) if dist_goal is not None else None})
            self.event("leg_end", k=k, outcome=leg_out, at=at)
            if leg_out != "arrived":
                outcome = leg_out
                break
        self.track = np.array(track) if track else np.zeros((0, 2))
        nav.destroy_node()
        return self.summary(outcome, legs)

    def summary(self, outcome: str, legs: List[dict]) -> dict:
        track = getattr(self, "track", np.zeros((0, 2)))
        failed = next((l for l in legs if l["outcome"] != "arrived"), None)
        out = {"tag": self.tag, "plant": self.spec.name, "robot": self.robot, "route": self.route,
               "outcome": outcome, "legs_done": sum(1 for l in legs if l["outcome"] == "arrived"),
               "legs": len(self.route) - 1, "stopped_at": failed["at"] if failed else None,
               "failed_leg": failed["k"] if failed else None,
               "seconds": round(time.time() - self.t0, 1),
               "driven": round(float(np.sum(np.linalg.norm(np.diff(track, axis=0), axis=1))), 2) if len(track) > 1 else 0.0}
        (self.out / "summary.json").write_text(json.dumps(
            {**out, "legs_detail": legs, "events": self.events,
             "track": np.round(track[:: max(1, len(track) // 300)], 3).tolist() if len(track) else []}, indent=1))
        return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--route", required=True, help="comma-separated zone ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None, help="a generated plant instead of the authored one")
    ap.add_argument("--changes", default="", help="segment:property=value;... applied before building")
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--rate", type=float, default=250.0)
    ap.add_argument("--no-people", action="store_true")
    a = ap.parse_args(argv)
    from scenarios.plant import random_plant_spec
    spec = plant_spec() if a.seed is None else random_plant_spec(a.seed)
    if a.changes:
        changes = []
        for item in a.changes.split(";"):
            sid, rest = item.split(":")
            prop, val = rest.split("=")
            changes.append((sid, prop, yaml.safe_load(val)))
        spec = PW.apply_changes(spec, changes)
    ep = Episode(spec, a.robot, a.route.split(","), Path(a.out), a.index, timeout=a.timeout, rate=a.rate,
                 people=not a.no_people)
    print(json.dumps(ep.run()))


if __name__ == "__main__":
    main()
