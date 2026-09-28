"""Integration only: deliberately tiny PPO/data/BC runs, NOT trained experts."""

import argparse, json, time
from pathlib import Path
from physical_ai.rl import train as train_rl
from physical_ai.data import collect
from physical_ai.bc import train as train_bc
from physical_ai.scenes import ROBOTS, TASKS
from reviewer.grade import grade


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    evidence = []
    start = time.monotonic()
    for robot in ROBOTS:
        for task in TASKS:
            base = root / robot / task
            if not (base / "rl/last.zip").exists():
                meta = train_rl(
                    robot,
                    task,
                    base / "rl",
                    total_steps=64,
                    n_steps=32,
                    n_envs=1,
                    seed=0,
                )
            else:
                meta = json.loads((base / "rl/last.json").read_text())
            for split, seed in [("train", 1000), ("val", 10000)]:
                if not (base / split / "manifest.json").exists():
                    collect(
                        robot,
                        task,
                        base / "rl/last.zip",
                        base / split,
                        episodes=1,
                        max_attempts=1,
                        seed=seed,
                        only_success=False,
                        max_steps=4,
                    )
            entry = dict(
                robot=robot, task=task, rl=str((base / "rl/last.zip").relative_to(root))
            )
            for variant, proprio, aug in [
                ("bc_baseline", False, False),
                ("bc_proprio", True, False),
                ("bc_robust", True, True),
            ]:
                if not (base / variant / "best.ts").exists():
                    train_bc(
                        base / "train",
                        base / "val",
                        base / variant,
                        epochs=1,
                        batch_size=4,
                        use_proprio=proprio,
                        image_augmentation=aug,
                    )
                entry[variant] = str((base / variant / "best.ts").relative_to(root))
            entries.append(entry)
            evidence.append(
                dict(
                    robot=robot,
                    task=task,
                    rl_parameter_delta=meta["parameter_delta_l2"],
                    rl_steps=meta["timesteps"],
                    integration=True,
                )
            )
            print(robot, task, "pipeline smoke complete", flush=True)
    (root / "REPORT.md").write_text(
        "Integration smoke: deliberately inadequate training and failure-inclusive data. No task-success claim."
    )
    (root / "submission.json").write_text(
        json.dumps(dict(report="REPORT.md", models=entries), indent=2)
    )
    grades = grade(root / "submission.json", root / "grading", episodes=1, max_steps=2)
    summary = dict(
        type="integration_smoke_not_learning_validation",
        elapsed_seconds=time.monotonic() - start,
        matrix=evidence,
        grading_cases=len(grades),
        grading_errors=[r for r in grades if "error" in r],
    )
    (root / "evidence.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["grading_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
