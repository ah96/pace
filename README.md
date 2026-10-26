# PACE: Pathwise Affordance Counterfactual Explanations

Code, scenarios, digital twin and experiments for

> **Actionable Counterfactual Explanations for Heterogeneous Mobile Robot Fleets in
> Remanufacturing Intralogistics** — submitted to the ERF 2027 scientific track.

When a robot in a remanufacturing plant cannot complete a mission — drive to a zone and,
if asked, pick something up there — the people on the floor need to know *what they can do
about it*. PACE grounds the mission in an ontology of typed passages, items and robots with
affordance rules, plans on a semantic navigation graph, reads candidate repairs off the rule
and cost terms that fail along the route a user asks about, verifies every repair set by
re-planning, and labels each change **actionable** (open the door, move the staged pallets,
restock the module lower) or **structural** (widen the aisle, raise the mezzanine slab), with
an effort estimate. Repairs that open a route are kept apart from savings that merely cheapen
it. A Gazebo digital twin of the plant, driven by Nav2, replays the advice: blocked before the
change, driven after it.

## Layout

```text
pace/                    the framework
  ontology.py            OWL 2 vocabulary and the typed property store
  domain.py              classes (Corridor, Doorway, Ramp, Item, Robot, ...), properties, enums
  config.py              every threshold and weight, and the feasibility predicates
  affordance.py          forward-chaining affordance rules (navigation and manipulation), edge cost
  navigation.py          semantic navigation graph, A*, Yen's k-shortest paths
  counterfactual.py      blocking and cost attribution, verified repair sets, greedy and exhaustive
  explanation.py         the three-layer report
  evaluation.py          ground truth, task types, baselines (incl. DiCE-style recourse), metrics
  twin/                  the Gazebo Classic 11 + ROS 2 Humble + Nav2 digital twin
    plant_world.py       one PlantSpec -> SDF world (racking, doors, cordons, staged pallets,
                         underpass, ramp, people), Nav2 map, places
    robots.py            the fleet: MiR100-class tote carrier, forklift AGV, TIAGo, Husky-class UGV
    tiago_fk.py          TIAGo forward kinematics and a reach solver (the door-pushing posture)
    episode.py           one Nav2 episode: a robot drives a questioned route, before/after a repair
    mission.py           a pick mission: the Nav2 episode, then MoveIt picks the item by its handle
    tiago_control.py     TIAGo with a live arm and gripper (ros2_control in Gazebo)
    moveit/              PAL's MoveIt configuration for TIAGo (vendored) and move_group's parameters
    render.py, sdf.py    stills through gzserver's camera; SDF builders
    fetch.py             AWS RoboMaker small warehouse and CC0 Fuel props into data/
scenarios/
  plant.py               the remanufacturing plant (11 zones, 14 passages, 4 items, 4 robots)
                         and the generator of plants nobody tuned (random_plant_spec)
  synthetic.py           lattices for runtime scaling
experiments/
  run_pace.py            the semantic benchmark -> results/*.csv
  make_results_table.py  the paper's table from the CSVs
  run_twin.py            the twin protocol (every selected route before/after its repair)
  run_missions.py        the pick missions (Nav2 + MoveIt), before/after a restock
  run_reach_sweep.py     TIAGo picks a module or a tote at 17 heights, against PACE's reach verdict
  summarise_reach_sweep.py  the sweep's agreement, height by height
  framework_figure.tex   Fig. 1, the framework (TikZ, built at the LNCS text width)
  make_mission_videos.py the missions' films as videos
  summarise_twin.py      the twin's numbers
  make_twin_figures.py   annotated explanation stills (the qualitative figures)
  make_fleet_figure.py   the fleet in the passages that constrain it
  run_scaling.py         runtime vs plant size
assets/gazebo/           vendored Fuel models, TIAGo, and the fleet's meshes (LICENSES.md)
tests/                   137 tests
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.10 or newer
pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The twin needs Ubuntu 22.04 with ROS 2 Humble, Gazebo Classic 11 (`ros-humble-gazebo-ros-pkgs`)
and Nav2 (`ros-humble-navigation2 ros-humble-nav2-bringup`); stills need an OpenGL display
(`DISPLAY=:0`). Pick missions with MoveIt 2 additionally need
`ros-humble-moveit ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-gazebo-ros2-control`,
and `ros-humble-gazebo-model-attachment-plugin` to lift what TIAGo grips.

## The plant, explained

```bash
PYTHONPATH=. python -m scenarios.plant      # every robot asked why it cannot reach shipping from the dock
```

For the forklift AGV the northern route needs three low-effort actions (raise the roller door,
move the staged pallets, open the fire door) and the southern route needs construction (a
very-narrow aisle, an underpass too narrow and too low); the tote carrier only hears about the
doors; the mobile manipulator opens the doors itself. A mission to the module on the upper
rack adds "restock it lower" to the list; a robot without an arm gets no repair at all.

## The model in detail

The paper describes PACE at a high level (Section 3, Fig. 1); this is the model the code implements.

**Ontology** (`pace/domain.py`, `pace/ontology.py`). OWL 2 classes for passages (`Corridor`,
`Doorway`, `Ramp`, ...), `Item` and `Robot`, with typed datatype properties:

| Individual | Properties |
|---|---|
| passage, structural (only construction changes them) | `width`, `height` (overhead clearance), `slope_angle` |
| passage, state | `door_state` (open/closed/locked), `is_accessible` (an aisle closed for repair), `is_hazardous` (an incident cordon), `restricted` (authorized staff only), `blocked_width` (staged goods), `crowd_density`, `obstacle_density`, `illumination`, `surface_type` |
| item | `item_height` (the surface it rests on), `item_width` (grip dimension), `item_mass` |
| robot | `robot_width`, `robot_height`, `min_clearance` (per side), `max_slope_angle`, `mobility_type`, `can_open_doors`, `has_arm`, `reach_height`, `gripper_opening`, `payload` |

**Affordances** (`pace/affordance.py`; every threshold and predicate in `pace/config.py`). Each
reads its own group of properties, so a missing affordance names its cause:

| Affordance | Holds when |
|---|---|
| Traversable | the passage is accessible, not hazardous and not restricted |
| Passable | `width - blocked_width >= robot_width + 2 min_clearance` and `height >= robot_height` |
| Openable | a closed (not locked) door and a robot that can open doors |
| Climbable | `slope_angle <= max_slope_angle`, and no wheeled robot on a stair-like angle (> 15 deg) |
| Observable | `illumination > 0.25` |
| Reachable | the robot has an arm and `item_height <= reach_height` |
| Graspable | the robot has an arm and `item_width <= gripper_opening` |
| Liftable | the robot has an arm and `item_mass <= payload` |

**Edge cost** (`AffordanceReasoner.navigation_cost`). An edge is impassable (cost infinity) for a
robot if it lacks Traversable or Passable, has a locked door or a closed door the robot cannot
open, exceeds the robot's rated slope, or, for an item edge, lacks Reachable, Graspable or
Liftable. Otherwise

    c(e, r) = beta(e) * ( d(e) + w_door [closed door the robot opens] + w_surf(surface)
                          + w_crowd * crowd_density + w_obst * obstacle_density
                          + w_vis [not Observable] + w_slope * slope_angle )

with each weight an equivalent detour in meters: `w_crowd = 4`, `w_obst = 2.5`, `w_door = 2.5`,
`w_vis = 1.8`, `w_slope = 0.06` per degree, `w_surf` from 0 (smooth, tiled) to 1.4 (wet), and
`beta = 0.75` on designated emergency routes. An item edge (the last leg of a pick mission, from the
item's zone to the item) costs its approach plus a handling term of 2 m. Planning is A*.

**Explanations** (`pace/counterfactual.py`). The routes a user can ask about are the `k + 2`
shortest paths in the unit-weight graph (Yen, `k = 3`), so currently blocked routes are among them.
*Blocking attribution* visits every infinite-cost leg of the queried route and emits one change
per violated condition (a *repair*), even when no single change suffices, so a conjunction such as
an underpass both too narrow and too low comes out whole. *Cost attribution* emits changes that
remove soft-cost terms on finite-cost legs of the queried route and the planner's own route (a
*saving*, reported only once the route is feasible; on a route the robot can already take, only
if it would then be preferred). Each change sets its property to a canonical value: the door
open, the cordon lifted, the staged goods removed, the passage widened to the robot's need plus
0.1 m, the clearance raised to its height plus 0.1 m, the ramp regraded to 1 degree inside its
limit, crowds to 0.2, lighting to 0.6, an item restocked 0.1 m below the arm's reach.

A set of changes `D` is a *verified repair set* for route `pi` when, evaluated on clones of the
affected individuals, `c(pi | D) < inf` or `c(pi | {}) - c(pi | D) > eps` with `eps = 1 m`. The
greedy search applies the whole attributed set and keeps it only if it verifies; `pace-opt`
enumerates subsets of the same candidate pool by increasing size (up to a pool of 12).

**Actionability and effort** (`PropertyChange.is_actionable`, `PropertyChange.effort`). A change
is *actionable* when staff can make it on the task's timescale without structural work:
`door_state`, `is_accessible`, `is_hazardous`, `restricted`, `blocked_width`, `crowd_density`,
`obstacle_density`, `illumination` and `item_height`; everything else is *structural*. Effort is
*low* for one operator action (open a door, move staged goods, redirect people, switch on the
lights, restock an item), *medium* for an authorization or work order (lift a restriction or a
cordon, reopen a closed aisle, treat a surface) and *high* for construction or a different
product (`width`, `height`, `slope_angle`, `item_width`, `item_mass`). Both are declared by the
facility, as mutability is in algorithmic recourse; they are not measured with users.

## Prose narrative (optional)

The report is template-based and complete on its own. A prose narrative can be added by any
language model the user has: `ChatCompletionsNarrator` talks, with the standard library only, to
every server that speaks the chat-completions protocol, hosted or local, and any function
`(system_prompt, user_message) -> str` can stand in for it.

```bash
export PACE_LLM_BASE_URL=http://localhost:8000/v1   # the server's API root
export PACE_LLM_MODEL=<model name>                  # and, for hosted providers, PACE_LLM_API_KEY
```

```python
from pace import ExplanationGenerator, ChatCompletionsNarrator
explainer = ExplanationGenerator(narrator=ChatCompletionsNarrator())
print(explainer.format_report(explanation, narrative=True))
```

## Semantic benchmark

```bash
PYTHONPATH=. python experiments/run_pace.py --all-pairs --output results/plant_allpairs.csv
PYTHONPATH=. python experiments/run_pace.py --all-pairs --only-plants --plants 30 \
    --output results/generated_allpairs.csv --methods pace pace-opt pace-noattr pace-pooled recourse unconstrained semantic random
PYTHONPATH=. python experiments/make_results_table.py --input results/plant_allpairs.csv \
    --second results/generated_allpairs.csv --include pace pace-opt recourse unconstrained semantic \
    --no-filter-rows --output build/results_table.tex          # paste over the tabular in main.tex
```

Tasks are every connected (robot, start, goal) triple, missions included, grouped into five
types: feasible with a deviation worth explaining, feasible with nothing to change, infeasible
with a single actionable fix, infeasible but repairable (construction, conjunctions), and
unrepairable (the cause is on the robot's side, e.g. no arm).

## The twin

```bash
PYTHONPATH=. python -m pace.twin.plant_world --out build/plant           # world.sdf, map, places
PYTHONPATH=. python -m pace.twin.episode --robot tote_amr --route dock,receiving,diagnostics \
    --out build/ep --changes "seg_dock_recv:door_state=open"             # one episode
PYTHONPATH=. python experiments/run_twin.py --plan-only                  # the protocol's episodes
PYTHONPATH=. python experiments/run_twin.py --procs 4                    # run them (a machine with the twin)
PYTHONPATH=. python experiments/summarise_twin.py                        # the numbers
PYTHONPATH=. DISPLAY=:0 python experiments/make_twin_figures.py          # Fig. 2 (annotated explanation)
PYTHONPATH=. DISPLAY=:0 python experiments/make_fleet_figure.py          # Fig. 3 (the fleet at its passages)
```

Both scripts keep Gazebo's raw renders under `build/`; `--reuse-raw` re-annotates or
re-assembles them without rendering again. The TIAGo panel is a posed still (arm on the push
bar, one leaf swung open); in the episodes, a robot that operates doors finds them open.

```bash
PYTHONPATH=. python -m pace.twin.tiago_fk 0.70 -0.22 1.0                  # re-solve the pushing posture
```

### Pick missions (Nav2 + MoveIt)

```bash
PYTHONPATH=. python experiments/run_missions.py --plan-only               # the five mission episodes
PYTHONPATH=. DISPLAY=:0 python experiments/run_missions.py --procs 2 --film  # run them (a machine with the twin)
PYTHONPATH=. python -m pace.twin.mission --route disassembly_A --item module_bench \
    --out build/pick                                                      # one pick, no driving
PYTHONPATH=. python experiments/make_mission_videos.py                    # pick.mp4, pick_end.jpg per mission
PYTHONPATH=. python experiments/run_reach_sweep.py --plan-only            # the reach sweep's 70 episodes
PYTHONPATH=. python experiments/run_reach_sweep.py --procs 2              # run them (a machine with the twin)
PYTHONPATH=. python experiments/summarise_reach_sweep.py                  # PACE's reach verdict against the twin
```

TIAGo drives the route with Nav2 and drives the last half-metre straight to the item's stand;
MoveIt (PAL's configuration, OMPL) takes arm and torso to a pre-grasp clear of the handle, the
gripper slides onto the handle and closes, and the arm lifts. The arm is position-controlled and
the fingers squeeze with at most the URDF's 16 N: a mission is `grasped` when both fingers stop
on the 3 cm handle, `picked` when the part then rises with the gripper, `unreachable` when no
grasp can be planned. Friction does not carry a part in Gazebo Classic (a position-commanded
gripper moves at zero velocity; velocity commands turned TIAGo's base as the arm swung out), so
once both fingers are on the handle the part is made fast to the wrist with the model attachment
plugin. That plugin moves a part until its link's frame meets the wrist's, so an item's link
frame is where the wrist is in a front grasp (`plant_world.item_anchor`). `--film` saves a side
and a front view of each pick.

Each episode has its own ROS domain and Gazebo port; Fast DDS runs UDP-only
(`pace/twin/fastdds_udp_only.xml`), without which a node can silently stop receiving TF and
Nav2 drives blind. The robot's prior map marks everything outside the questioned route as
occupied, so Nav2 can only take the route being asked about.

## Citation

```bibtex
@inproceedings{halilovic2027pace,
  title     = {Actionable Counterfactual Explanations for Heterogeneous Mobile Robot Fleets
               in Remanufacturing Intralogistics},
  author    = {Halilovic, Amar and Krivic, Senka},
  booktitle = {European Robotics Forum (ERF 2027), scientific track},
  year      = {2027}
}
```

## License

MIT (`LICENSE`); third-party assets under their own licences (`assets/gazebo/LICENSES.md`).
