"""Private feasibility matrix; uses scripted physical contacts, not learned RL."""

import argparse
import json
from pathlib import Path
from physical_ai.scenes import ROBOTS, TASKS
from physical_ai.env import RUNTIME_PROVENANCE
from reviewer.physical_probe import probe
from reviewer.scenarios import scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    rows = []
    for robot in ROBOTS:
        for task in TASKS:
            for seed in [0, 1, 7]:
                row = probe(robot, task, seed)
                row["split"] = "base"
                rows.append(row)
            for scenario_seed, episode_seed in [(739, 0), (113, 7), (914, 12)]:
                row = probe(
                    robot,
                    task,
                    episode_seed,
                    parameters=scenario(task, scenario_seed, robot=robot),
                )
                row.update(split="advanced", scenario_seed=scenario_seed)
                rows.append(row)
            print(
                robot,
                task,
                [(row["success"], row["steps"]) for row in rows[-6:]],
                flush=True,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["runtime_provenance"] = dict(RUNTIME_PROVENANCE)
    path.write_text(json.dumps(rows, indent=2))
    if not all(row["success"] and row["steps"] <= 1000 for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
