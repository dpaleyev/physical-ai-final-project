"""Private ordinary-reset learning validation and real PPO -> data -> BC check."""

import argparse
import json
from pathlib import Path
import shutil
import time

from physical_ai.bc import train as train_bc
from physical_ai.data import collect, EpisodeDataset, check_disjoint
from physical_ai.evaluate import checkpoint_policy, evaluate, save_result
from physical_ai.rl import sha256


def validate(
    robot,
    task,
    checkpoint,
    out,
    episodes=50,
    train_episodes=20,
    val_episodes=5,
    bc_epochs=3,
    seed=80000,
):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start = time.monotonic()
    checkpoint = Path(checkpoint)
    policy, observation, metadata = checkpoint_policy(checkpoint, "rl", robot, task)
    result = evaluate(robot, task, policy, observation, range(seed, seed + episodes))
    result.update(
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        reset_distribution="ordinary_ManipulationEnv_no_curriculum",
        purpose="independent_base_quality_gate",
    )
    save_result(result, out / "rl_evaluation.json")
    print(robot, task, "ordinary-reset SR", result["success_rate"], flush=True)
    if episodes < 50 or result["success_rate"] < 0.8:
        raise RuntimeError("Expert did not meet >=80% success on >=50 ordinary resets")
    if task == "color_match" and (
        set(result["by_goal_slot"]) != {"0", "1"}
        or min(g["success_rate"] for g in result["by_goal_slot"].values()) < 0.8
    ):
        raise RuntimeError(
            "Both color targets must independently meet the SR threshold"
        )
    shutil.copy2(checkpoint, out / "expert.zip")
    shutil.copy2(checkpoint.with_suffix(".json"), out / "expert.json")
    for split, count, offset in [
        ("train", train_episodes, 1000),
        ("val", val_episodes, 2000),
    ]:
        collect(
            robot,
            task,
            out / "expert.zip",
            out / split,
            episodes=count,
            seed=seed + offset,
            max_attempts=count * 5,
            only_success=True,
        )
        print(robot, task, split, count, "successful PPO episodes", flush=True)
    training, validation = EpisodeDataset(out / "train"), EpisodeDataset(out / "val")
    check_disjoint(training, validation)
    if not all(
        m["source"] == "ppo"
        and m["success"]
        and m["expert_sha256"] == metadata["sha256"]
        for m in training.metadata + validation.metadata
    ):
        raise RuntimeError("Demonstration provenance or success mismatch")
    history = train_bc(
        out / "train",
        out / "val",
        out / "bc",
        epochs=bc_epochs,
        use_proprio=True,
        seed=0,
    )
    bc_policy, bc_obs, _ = checkpoint_policy(out / "bc/best.ts", "bc", robot, task)
    bc_result = evaluate(
        robot, task, bc_policy, bc_obs, range(seed + 3000, seed + 3005)
    )
    save_result(bc_result, out / "bc_evaluation.json")
    summary = dict(
        robot=robot,
        task=task,
        expert_sha256=metadata["sha256"],
        rl_success_rate=result["success_rate"],
        rl_successes=result["successes"],
        rl_episodes=episodes,
        by_goal_slot=result["by_goal_slot"],
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        train_transitions=len(training),
        val_transitions=len(validation),
        bc_epochs=bc_epochs,
        bc_history=history,
        bc_success_rate=bc_result["success_rate"],
        bc_claim="Training/export/inference on real successful PPO data; short BC pilot only",
        elapsed_seconds=time.monotonic() - start,
    )
    (out / "evidence.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["robot", "task", "checkpoint", "out"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--train-episodes", type=int, default=20)
    p.add_argument("--val-episodes", type=int, default=5)
    p.add_argument("--bc-epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=80000)
    validate(**vars(p.parse_args()))
