"""Private fixed-environment grader. Runs exported student policies only."""

import argparse
import json
from pathlib import Path
import traceback
from physical_ai.evaluate import checkpoint_policy, evaluate, wilson_interval
from physical_ai.rl import load_rl
from physical_ai.scenes import ROBOTS, TASKS
from reviewer.scenarios import scenario

VARIANTS = ("bc_baseline", "bc_proprio", "bc_robust")


def contained(root, relative, must_exist=True):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Submission path escapes root")
    if must_exist and not path.is_file():
        raise ValueError(f"Missing file: {relative}")
    return path


def validate_submission(submission, root):
    entries = submission.get("models", [])
    expected = {(r, t) for r in ROBOTS for t in TASKS}
    if (
        len(entries) != 8
        or {(e.get("robot"), e.get("task")) for e in entries} != expected
    ):
        raise ValueError("Expected exactly eight distinct robot/task entries")
    for entry in entries:
        for key in ("rl",) + VARIANTS:
            contained(root, entry[key], must_exist=False)
    contained(root, submission["report"])
    return entries


def grade(submission_path, out, episodes=50, seed=738921, max_steps=1000, video=False):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    path = Path(submission_path).resolve()
    root = path.parent
    entries = validate_submission(json.loads(path.read_text()), root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in entries:
        robot, task = entry["robot"], entry["task"]
        for variant in ("rl",) + VARIANTS:
            for split in ("base", "advanced"):
                rows = []
                errors = []
                try:
                    artifact = contained(root, entry[variant])
                    if variant != "rl" and artifact.suffix != ".ts":
                        raise ValueError("BC submission must use exported .ts policies")
                    contained(
                        root, str(artifact.with_suffix(".json").relative_to(root))
                    )
                    predict, obs, meta = checkpoint_policy(
                        artifact, "rl" if variant == "rl" else "bc", robot, task
                    )
                    for i in range(episodes):
                        episode_seed = seed + i
                        params = (
                            {}
                            if split == "base"
                            else scenario(task, seed + 100000 + i, robot=robot)
                        )
                        result = evaluate(
                            robot,
                            task,
                            predict,
                            obs,
                            [episode_seed],
                            params,
                            max_steps,
                            (
                                out / "videos" / robot / task / variant / split
                                if video and i == 0
                                else None
                            ),
                        )
                        row = result["episodes"][0]
                        row["parameters"] = params
                        rows.append(row)
                    successes = sum(r["success"] for r in rows)
                    summary = dict(
                        robot=robot,
                        task=task,
                        variant=variant,
                        split=split,
                        n=len(rows),
                        success_rate=successes / len(rows),
                        wilson_95=wilson_interval(successes, len(rows)),
                        episodes=rows,
                    )
                except Exception as exc:
                    summary = dict(
                        robot=robot,
                        task=task,
                        variant=variant,
                        split=split,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                results.append(summary)
                (out / "results.json").write_text(json.dumps(results, indent=2))
    import csv

    with (out / "summary.csv").open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "robot",
                "task",
                "variant",
                "split",
                "n",
                "success_rate",
                "error",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(results)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("submission")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=738921)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--video", action="store_true")
    a = vars(p.parse_args())
    a["submission_path"] = a.pop("submission")
    result = grade(**a)
    raise SystemExit(1 if any("error" in r for r in result) else 0)
