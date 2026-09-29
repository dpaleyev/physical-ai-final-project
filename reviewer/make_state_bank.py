"""PRIVATE: physical controller generates reset states, never RL action labels."""

import argparse, json
from pathlib import Path
import numpy as np
from physical_ai.env import ManipulationEnv
from physical_ai.curriculum import scene_fingerprint
from physical_ai.rl import sha256
from reviewer.physical_probe import transfer


def generate(robot, task, out, seeds=12, stride=0):
    phases = [0, 45, 80, 98, 138, 183, 218, 236]
    if stride:
        phases = sorted(set(phases + list(range(0, 237, stride))))
    rows = []
    accepted = []
    with ManipulationEnv(robot, task) as env:
        actual_step = env.step
        for seed in range(seeds * 4):
            episode_rows = []
            env.reset(seed=seed)

            def capture():
                row = {k: getattr(env.data, k).copy() for k in ["qpos", "qvel", "ctrl"]}
                row.update(
                    ee_target=env.ee_target.copy(),
                    goals=env.goals.copy(),
                    rgba=env.model.geom_rgba.copy(),
                    phase=env.steps,
                )
                episode_rows.append(row)

            capture()

            def step(action):
                result = actual_step(action)
                if env.steps in phases:
                    capture()
                return result

            env.step = step
            transfer(env, "cup0", env.goals[0])
            for _ in range(120):
                if env.success():
                    break
                env.step([0, 0, 0, 1])
            if env.success():
                rows.extend(episode_rows)
                accepted.append(seed)
                if len(accepted) == seeds:
                    break
            else:
                print(
                    "Skipped unsuccessful initialization", robot, task, seed, flush=True
                )
        if len(accepted) < seeds:
            raise RuntimeError(
                f"Only {len(accepted)}/{seeds} valid initialization trajectories"
            )
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    metadata = dict(
        robot=robot,
        task=task,
        seeds=seeds,
        source_seeds=accepted,
        stride=stride,
        phases=phases,
        source="physical_controller_initial_states_no_action_labels",
        model_sha256=scene_fingerprint(robot, task),
        scenes_sha256=sha256(
            Path(__file__).resolve().parents[1] / "physical_ai/scenes.py"
        ),
    )
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata=json.dumps(metadata))
    print(path, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--robot", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--stride", type=int, default=0)
    generate(**vars(p.parse_args()))
