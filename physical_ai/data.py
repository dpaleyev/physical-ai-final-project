"""Episode-aligned demonstrations and lazy episode loading."""

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from physical_ai.env import ManipulationEnv, RUNTIME_PROVENANCE
from physical_ai.rl import load_rl, sha256


def episode_key(meta):
    # Same reset, even from a different expert, must not cross a data split.
    return json.dumps(
        {k: meta[k] for k in ["robot", "task", "seed", "parameters"]}, sort_keys=True
    )


def save_episode(path, rgb, proprio, actions, meta):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not (len(rgb) == len(proprio) == len(actions)) or not len(actions):
        raise ValueError("Episode length mismatch or empty episode")
    if rgb.shape[1:] != (84, 84, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB (T,84,84,3)")
    if actions.shape[1:] != (4,) or proprio.ndim != 2:
        raise ValueError("Invalid action/proprio shapes")
    if (
        not np.isfinite(actions).all()
        or not np.isfinite(proprio).all()
        or np.max(np.abs(actions)) > 1.00001
    ):
        raise ValueError("Invalid finite/range contract")
    episode_key(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    dones = np.zeros(len(actions), bool)
    dones[-1] = True
    np.savez_compressed(
        path,
        rgb=rgb,
        proprio=proprio.astype(np.float32),
        actions=actions.astype(np.float32),
        dones=dones,
        metadata=json.dumps(dict(meta, schema=1)),
    )


class EpisodeDataset(Dataset):
    def __init__(self, directory):
        self.paths = sorted(Path(directory).glob("*.npz"))
        if not self.paths:
            raise ValueError(f"No episodes in {directory}")
        self.metadata = []
        self.offsets = [0]
        self._cache_index = None
        for path in self.paths:
            with np.load(path, allow_pickle=False) as ep:
                meta = json.loads(str(ep["metadata"]))
                if meta.get("schema") != 1:
                    raise ValueError("Unknown episode schema")
                actions, prop = ep["actions"], ep["proprio"]
                if (
                    actions.shape != (len(prop), 4)
                    or not np.isfinite(actions).all()
                    or not np.isfinite(prop).all()
                ):
                    raise ValueError(f"Invalid episode: {path}")
                self.metadata.append(meta)
                self.offsets.append(self.offsets[-1] + len(actions))
                self.proprio_dim = prop.shape[1]
        keys = [episode_key(m) for m in self.metadata]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate episode resets")
        if len({(m["robot"], m["task"]) for m in self.metadata}) != 1:
            raise ValueError("Expected one robot/task per dataset")

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect_right(self.offsets, index) - 1
        if self._cache_index != episode:
            with np.load(self.paths[episode], allow_pickle=False) as ep:
                self._cache = {k: ep[k] for k in ("rgb", "proprio", "actions")}
            self._cache_index = episode
        i = index - self.offsets[episode]
        return (
            torch.from_numpy(self._cache["rgb"][i].copy()).permute(2, 0, 1).float()
            / 255,
            torch.from_numpy(self._cache["proprio"][i].copy()),
            torch.from_numpy(self._cache["actions"][i].copy()),
        )


def check_disjoint(train, validation):
    if {episode_key(m) for m in train.metadata} & {
        episode_key(m) for m in validation.metadata
    }:
        raise ValueError("Train/validation episode overlap")
    if (train.metadata[0]["robot"], train.metadata[0]["task"]) != (
        validation.metadata[0]["robot"],
        validation.metadata[0]["task"],
    ):
        raise ValueError("Train/validation robot/task mismatch")


def collect(
    robot,
    task,
    checkpoint,
    out,
    episodes=100,
    seed=1000,
    max_attempts=500,
    only_success=True,
    scene_bank=None,
    max_steps=1000,
):
    if episodes < 1 or max_attempts < episodes:
        raise ValueError("Invalid collection budget")
    policy, metadata = load_rl(checkpoint, robot, task)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.iterdir()):
        raise FileExistsError("Collection output must be empty")
    bank = scene_bank or [{}]
    accepted = 0
    attempts = []
    for attempt in range(max_attempts):
        parameters = bank[attempt % len(bank)]
        with ManipulationEnv(
            robot, task, parameters=parameters, max_steps=max_steps
        ) as env:
            state, _ = env.reset(seed=seed + attempt)
            rgb = []
            prop = []
            actions = []
            preflight = None
            if only_success:
                # A deterministic state-only pass avoids expensive RGB rendering
                # for rejected episodes. Accepted episodes are replayed exactly.
                for step in range(max_steps):
                    action, _ = policy.predict(state, deterministic=True)
                    state, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        break
                preflight = dict(
                    success=info["is_success"], steps=step + 1, state=state.copy()
                )
                if preflight["success"]:
                    state, _ = env.reset(seed=seed + attempt)
            if not only_success or preflight["success"]:
                for _ in range(max_steps):
                    obs = env.bc_observation()
                    action, _ = policy.predict(state, deterministic=True)
                    rgb.append(obs["rgb"])
                    prop.append(obs["proprio"])
                    actions.append(action)
                    state, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        break
                if preflight is not None and (
                    not info["is_success"]
                    or len(actions) != preflight["steps"]
                    or not np.allclose(state, preflight["state"], atol=1e-6, rtol=0)
                ):
                    raise RuntimeError(
                        "Deterministic PPO replay differed from preflight; episode not saved"
                    )
            meta = dict(
                robot=robot,
                task=task,
                seed=seed + attempt,
                parameters=parameters,
                expert_sha256=sha256(checkpoint),
                source="ppo",
                success=info["is_success"],
                failure=info["failure"],
                runtime_provenance=dict(RUNTIME_PROVENANCE),
            )
            attempts.append({"seed": seed + attempt, "success": info["is_success"]})
            if info["is_success"] or not only_success:
                save_episode(
                    out / f"episode_{accepted:06d}.npz",
                    np.asarray(rgb),
                    np.asarray(prop),
                    np.asarray(actions),
                    meta,
                )
                accepted += 1
        (out / "manifest.json").write_text(
            json.dumps(
                dict(
                    schema=1,
                    robot=robot,
                    task=task,
                    accepted=accepted,
                    requested=episodes,
                    attempts=attempts,
                    only_success=only_success,
                    state_only_preflight=only_success,
                    runtime_provenance=dict(RUNTIME_PROVENANCE),
                ),
                indent=2,
            )
        )
        if accepted == episodes:
            return
    raise RuntimeError(
        f"Collected {accepted}/{episodes} episodes in {max_attempts} attempts; evaluate/improve the PPO expert first. Partial data and attempt log retained."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["robot", "task", "checkpoint", "out"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-attempts", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--include-failures", action="store_true")
    p.add_argument("--scene-bank")
    a = vars(p.parse_args())
    a["only_success"] = not a.pop("include_failures")
    a["scene_bank"] = (
        json.loads(Path(a["scene_bank"]).read_text()) if a["scene_bank"] else None
    )
    collect(**a)


if __name__ == "__main__":
    main()
