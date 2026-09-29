"""Export the observed stage lineage for replay with the current implementation."""

import argparse
import hashlib
import inspect
import json
import shutil
from pathlib import Path
import numpy as np
import mujoco

from physical_ai.reset_states import bank_digest, read_bank
from physical_ai.rl import train, sha256
from physical_ai.scenes import ROOT, ROBOTS, TASKS, build_scene
from physical_ai.curriculum import scene_fingerprint


def public_parameters(robot, task, parameters):
    """Canonicalize redundant overrides only after compiled-model equivalence."""

    def digest(params):
        model = mujoco.MjModel.from_xml_string(build_scene(robot, task, params))
        buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, buffer)
        return hashlib.sha256(buffer).hexdigest()

    original, default = digest(parameters), digest({})
    if original != default:
        raise ValueError("Reference parameters produce a different base model")
    return {}, default


def export(experts):
    experts = json.loads(Path(experts).read_text())
    required = {f"{robot}/{task}" for robot in ROBOTS for task in TASKS}
    if set(experts) != required:
        raise ValueError("Expected exactly eight robot/task checkpoints")
    for target, filename in experts.items():
        meta = json.loads(Path(filename).with_suffix(".json").read_text())
        if target != f'{meta["robot"]}/{meta["task"]}':
            raise ValueError(f"Checkpoint robot/task does not match target {target}")
    stages, known, active, adaptations = [], {}, set(), []
    training_keys = set(inspect.signature(train).parameters) - {
        "robot",
        "task",
        "out",
        "resume",
        "transfer",
    }
    bank_dir = ROOT / "assets/reset_states"
    bank_dir.mkdir(parents=True, exist_ok=True)

    def visit(filename):
        path = Path(filename).resolve()
        if path in active:
            raise ValueError("Checkpoint lineage cycle")
        if path in known:
            return known[path]
        active.add(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["sha256"] != sha256(path):
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        cfg = meta["config"]
        parents = {}
        for field in ("resume", "transfer"):
            if cfg.get(field):
                parents[field + "_from"] = visit(cfg[field])
        options = {key: value for key, value in cfg.items() if key in training_keys}
        if options.get("parameters"):
            original = options["parameters"]
            options["parameters"], model_hash = public_parameters(
                meta["robot"], meta["task"], original
            )
            adaptations.append(
                dict(
                    checkpoint_sha256=meta["sha256"],
                    robot=meta["robot"],
                    task=meta["task"],
                    original_parameters=original,
                    compiled_model_sha256=model_hash,
                    reason="Overrides equal the current default compiled model",
                )
            )
        if options.get("curriculum_bank"):
            bank = Path(options["curriculum_bank"])
            expected = cfg.get("curriculum_bank_sha256")
            if bank.is_dir():
                if bank_digest(bank) != expected:
                    raise ValueError(f"Bank hash mismatch: {bank}")
            else:
                # Resolve immutable historical metadata to its verified replacement.
                migrations = json.loads(
                    (ROOT / "reviewer/evidence/reset_bank_migration.json").read_text()
                )
                matches = [r for r in migrations if r["source_bank_sha256"] == expected]
                if len(matches) != 1:
                    raise ValueError(
                        f"No verified LeRobot replacement for bank: {bank}"
                    )
                record = matches[0]
                bank = ROOT / record["path"]
                if bank_digest(bank) != record["bank_sha256"]:
                    raise ValueError(f"Migrated bank hash mismatch: {bank}")
            _, bank_meta = read_bank(bank)
            if (bank_meta["robot"], bank_meta["task"]) != (meta["robot"], meta["task"]):
                raise ValueError(f"Bank robot/task mismatch: {bank}")
            if bank_meta.get("model_sha256") != scene_fingerprint(
                meta["robot"], meta["task"]
            ):
                raise ValueError(f"Bank belongs to a different scene: {bank}")
            dest = bank_dir / bank.name
            if bank.resolve() != dest.resolve():
                if dest.exists():
                    if bank_digest(dest) != bank_digest(bank):
                        raise ValueError(f"Exported bank differs: {dest}")
                else:
                    shutil.copytree(bank, dest)
            options["curriculum_bank"] = str(dest.relative_to(ROOT))
        key = f"stage_{len(stages):03d}"
        node = dict(
            id=key,
            robot=meta["robot"],
            task=meta["task"],
            train=options,
            reference_checkpoint_sha256=meta["sha256"],
            reference_code=dict(
                environment_sha256=cfg.get("environment_sha256"),
                scenes_sha256=cfg.get("scenes_sha256"),
                reward_version=cfg.get("reward_version"),
                trainer_sha256=cfg.get("trainer_sha256"),
                policy_source_sha256=cfg.get("policy_source_sha256"),
            ),
            **parents,
        )
        stages.append(node)
        known[path] = key
        active.remove(path)
        return key

    outputs = {target: visit(experts[target]) for target in sorted(required)}
    recipe = dict(
        schema=1,
        description="Observed PPO stage lineage replayed with the current implementation; not a bitwise reproduction claim. Missing historical code hashes are null. Bank states are physical-controller initializations, not action labels. Evaluate ordinary resets independently.",
        unique_requested_transitions=sum(n["train"]["total_steps"] for n in stages),
        stages=stages,
        outputs=outputs,
    )
    dest = ROOT / "configs/ppo_calibrated.json"
    dest.write_text(json.dumps(recipe, indent=2))
    (ROOT / "reviewer/evidence/curriculum_export_adaptations.json").write_text(
        json.dumps(adaptations, indent=2)
    )
    print(dest, "stages", len(stages), flush=True)
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experts",
        required=True,
        help="JSON mapping robot/task to original final checkpoint path",
    )
    export(p.parse_args().experts)
