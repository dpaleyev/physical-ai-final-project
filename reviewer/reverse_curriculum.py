"""Experimental true PPO with a reverse reset curriculum; no action supervision."""

import argparse, json, time
from pathlib import Path
import numpy as np
from physical_ai.rl import train, load_rl
from physical_ai.curriculum import CurriculumEnv
from physical_ai.env import ManipulationEnv


def evaluate_phase(robot, task, checkpoint, bank, phase, episodes=10):
    policy, _ = load_rl(checkpoint, robot, task)
    successes = 0
    with CurriculumEnv(
        robot, task, state_bank=bank, phase=phase, normal_fraction=0, max_steps=400
    ) as env:
        for seed in range(65000, 65000 + episodes):
            obs, _ = env.reset(seed=seed)
            for _ in range(400):
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(action)
                if done or trunc:
                    break
            successes += int(info["is_success"])
    return successes / episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument(
        "--phases", type=int, nargs="+", default=[236, 218, 183, 138, 98, 80, 45, 0]
    )
    p.add_argument("--attempts", type=int, default=6)
    p.add_argument("--transfer")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--normal-fraction", type=float, default=0.05)
    p.add_argument("--use-sde", action="store_true")
    p.add_argument("--phase-window", type=int, default=0)
    p.add_argument("--actor-head-scale", type=float, default=1.0)
    a = p.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    resume = a.resume
    results = []
    for phase in a.phases:
        for attempt in range(a.attempts):
            out = root / f"phase_{phase}_attempt_{attempt}"
            if not (out / "last.zip").exists():
                train(
                    a.robot,
                    a.task,
                    out,
                    total_steps=a.steps,
                    n_envs=a.n_envs,
                    n_steps=512,
                    ent_coef=0.0001,
                    curriculum_bank=a.bank,
                    curriculum_phase=phase,
                    normal_fraction=a.normal_fraction if phase else 1,
                    bounded_policy=True,
                    max_steps=400,
                    learning_rate=a.learning_rate,
                    use_sde=a.use_sde,
                    curriculum_window=a.phase_window,
                    actor_head_scale=a.actor_head_scale if not results else 1.0,
                    resume=resume,
                    transfer=a.transfer if resume is None else None,
                    checkpoint_every=100000,
                )
            resume = out / "last.zip"
            sr = evaluate_phase(a.robot, a.task, resume, a.bank, phase)
            results.append(
                dict(
                    phase=phase,
                    attempt=attempt,
                    checkpoint=str(resume),
                    selection_sr=sr,
                )
            )
            (root / "progress.json").write_text(json.dumps(results, indent=2))
            print(
                a.robot,
                a.task,
                "phase",
                phase,
                "attempt",
                attempt,
                "selection_sr",
                sr,
                flush=True,
            )
            if sr >= 0.8:
                break
        else:
            print("STAGE DID NOT CONVERGE", phase, flush=True)
            raise SystemExit(1)
    print("COMPLETED", resume, flush=True)


if __name__ == "__main__":
    main()
