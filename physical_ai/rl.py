"""Real PPO training; no scripted controller or distillation in this module."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from physical_ai.env import DISCOUNT, REWARD_VERSION, ManipulationEnv
from physical_ai.scenes import ROBOTS, TASKS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rl(path, robot, task, **load_kwargs):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if (meta["robot"], meta["task"]) != (robot, task):
        raise ValueError("Checkpoint robot/task mismatch")
    if meta.get("algorithm") != "PPO" or meta.get("schema") != 1:
        raise ValueError("Expected a schema-1 PPO checkpoint")
    if meta["sha256"] != sha256(path):
        raise ValueError("Checkpoint hash mismatch")
    return PPO.load(path, device="cpu", **load_kwargs), meta


def train(
    robot,
    task,
    out,
    total_steps=2000000,
    n_steps=1024,
    n_envs=4,
    seed=0,
    learning_rate=3e-4,
    ent_coef=0.01,
    reward_stage="place",
    resume=None,
    parameters=None,
):
    if total_steps <= 0 or n_steps < 2 or n_envs < 1:
        raise ValueError("Invalid training budget")
    torch.set_num_threads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.zip").exists():
        raise FileExistsError(
            "Use a new run directory; --resume reads a prior checkpoint"
        )
    kwargs = dict(
        robot=robot, task=task, parameters=parameters, reward_stage=reward_stage
    )

    def make():
        return Monitor(ManipulationEnv(**kwargs))

    env = (
        SubprocVecEnv([make for _ in range(n_envs)], start_method="spawn")
        if n_envs > 1
        else DummyVecEnv([make])
    )
    env.seed(seed)
    config = dict(
        robot=robot,
        task=task,
        total_steps=total_steps,
        n_steps=n_steps,
        n_envs=n_envs,
        seed=seed,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        reward_stage=reward_stage,
        parameters=parameters or {},
        resume=str(resume) if resume else None,
        gamma=DISCOUNT,
        reward_version=REWARD_VERSION,
        environment_sha256=sha256(Path(__file__).with_name("env.py")),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2))
    try:
        if resume:
            policy, _ = load_rl(
                resume,
                robot,
                task,
                env=env,
                gamma=DISCOUNT,
                n_steps=n_steps,
                seed=seed,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                batch_size=min(256, n_steps * n_envs),
            )
        else:
            policy = PPO(
                "MlpPolicy",
                env,
                n_steps=n_steps,
                batch_size=min(256, n_steps * n_envs),
                n_epochs=10,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                gamma=DISCOUNT,
                policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
                seed=seed,
                device="cpu",
                verbose=0,
            )
        policy.set_logger(configure(str(out), ["csv", "tensorboard"]))
        before = torch.cat(
            [p.detach().flatten().cpu() for p in policy.policy.parameters()]
        )
        policy.learn(total_timesteps=total_steps, reset_num_timesteps=not bool(resume))
        policy.save(out / "last.zip")
        after = torch.cat(
            [p.detach().flatten().cpu() for p in policy.policy.parameters()]
        )
        meta = dict(
            schema=1,
            algorithm="PPO",
            robot=robot,
            task=task,
            timesteps=policy.num_timesteps,
            parameter_delta_l2=float(torch.linalg.vector_norm(after - before)),
            seed=seed,
            reward_stage=reward_stage,
            sha256=sha256(out / "last.zip"),
            config=config,
        )
        (out / "last.json").write_text(json.dumps(meta, indent=2))
        return meta
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=ROBOTS, required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--total-steps", type=int, default=2000000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument(
        "--reward-stage", choices=["reach", "lift", "place"], default="place"
    )
    p.add_argument("--resume")
    p.add_argument("--parameters")
    a = vars(p.parse_args())
    a["parameters"] = (
        json.loads(Path(a["parameters"]).read_text()) if a["parameters"] else None
    )
    print(json.dumps(train(**a), indent=2))


if __name__ == "__main__":
    main()
