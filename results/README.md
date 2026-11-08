# Results of the paper

The result files behind every number in the paper, so that the tables and the text can be
checked without re-running the benchmark or the digital twin.

| File | What | Produced by |
|---|---|---|
| `plant_allpairs.csv` | semantic benchmark on the authored plant: 616 tasks x 8 methods | `experiments/run_pace.py --all-pairs` |
| `generated_allpairs.csv.gz` | semantic benchmark on the 30 generated plants: 16,852 tasks x 8 methods | `experiments/run_pace.py --all-pairs --only-plants --plants 30` |
| `twin_episodes.csv` | digital-twin protocol: 186 Nav2 episodes (89 repaired routes before and after, 8 feasible routes) | `experiments/run_twin.py` |
| `twin_missions.csv` | pick missions with Nav2 and MoveIt: 5 episodes | `experiments/run_missions.py` |

The semantic benchmark has one row per (scenario, agent, start, goal, method): the task type
(`cat`), the ground-truth size, and the method's metrics (`recall_at_k`, `actionable_fraction`,
`feasibility_recovery`, `num_returned`, `runtime_ms`, ...). Runtimes are wall-clock times of a
single-threaded process and differ between machines; every other column is deterministic.
pandas reads the `.gz` file directly.

The twin files have one row per episode: the robot, the route, the phase (`before`/`after`
the repair), the outcome (`arrived`, `aborted`, `stuck`, ...) and where the robot stopped.

From the repository root:

```bash
PYTHONPATH=. python experiments/make_paper_numbers.py --authored results/plant_allpairs.csv \
    --generated results/generated_allpairs.csv.gz --twin results/twin_episodes.csv \
    --out build/numbers.tex                                    # every number in the paper's text
PYTHONPATH=. python experiments/make_results_table.py --input results/plant_allpairs.csv \
    --second results/generated_allpairs.csv.gz --include pace pace-opt recourse unconstrained semantic \
    --output build/results_table.tex                           # Table 1
PYTHONPATH=. python experiments/summarise_twin.py --input results/twin_episodes.csv   # the twin's counts
```
