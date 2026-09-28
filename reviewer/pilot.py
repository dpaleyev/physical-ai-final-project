"""Run real PPO pilots across the matrix; keeps evidence, makes no convergence claim."""

import argparse, json, time
from pathlib import Path
from physical_ai.rl import train
from physical_ai.evaluate import checkpoint_policy, evaluate
from physical_ai.scenes import ROBOTS, TASKS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=2000000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--episodes", type=int, default=20)
    a = p.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for robot in ROBOTS:
        for task in TASKS:
            path = root / robot / task
            start = time.monotonic()
            if not (path / "last.zip").exists():
                train(robot, task, path, total_steps=a.steps, n_envs=a.n_envs)
            policy, obs, meta = checkpoint_policy(path / "last.zip", "rl", robot, task)
            result = evaluate(
                robot, task, policy, obs, range(60000, 60000 + a.episodes)
            )
            result["elapsed_seconds"] = time.monotonic() - start
            result["training"] = meta
            results.append(result)
            (root / "results.json").write_text(json.dumps(results, indent=2))
            print(robot, task, result["success_rate"], flush=True)


if __name__ == "__main__":
    main()
