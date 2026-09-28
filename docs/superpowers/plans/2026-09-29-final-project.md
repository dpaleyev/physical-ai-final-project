# Physical AI final project: approved implementation plan

Goal: independent course repository with eight physical scenes, student assignment,
RL-to-data-to-BC pipeline, robustness experiments and private grading infrastructure.

User approved the eight-step plan in chat on 2026-09-29. Clarification: student
materials must not enumerate the hidden perturbations or direct students to change
shelf height. Offer general-purpose scene editing, no hidden parameter ranges/seeds.
Repository owner is dpaleyev (initial dpalayev was a typo). Repository private.

## Tasks and acceptance

1. Assets and scenes: vendor pinned UR5e/KUKA MJCF and licenses; add physical grippers,
   four tasks each, previews. Test compilation, deterministic reset, stability,
   reachable object/goal positions, physical pick/release via an author-only probe.
2. Environment: Gymnasium API, normalized end-effector delta + jaw action, same
   MuJoCo engine for RL/collection/evaluation. Privileged RL vector and separate
   RGB/proprio BC observation. Test finite bounded actions, robot limits, no state
   leakage and task success (identity, release, stability, multiple objects).
3. RL: PPO runner, seeded vector environments, config/checkpoint/resume/logging,
   staged curriculum support and explicit algorithm provenance. Test actual weight
   update and checkpoint inference. Never represent distillation as RL.
4. Data and BC: bounded collection attempts, episodic NPZ + manifests, split hygiene,
   image BC baseline and configurable proprio branch, reproducible training/eval.
   Test observation/action alignment, dataset validation, overlap rejection,
   training update and checkpoint roundtrip.
5. Evaluation: shared rollout protocol, success rate/Wilson intervals, per-episode
   JSON/CSV and videos, scene modifications from JSON. Test failure termination,
   stable placement vs hover, metadata validation and statistics.
6. Grading: private scenario generator, submission schema, matrix coverage and
   base/advanced runner. Export students via allowlist to a fresh directory/archive,
   without git history, reviewer files, plans or validation evidence. Test export
   and actual base/advanced evaluation, not just presence of files.
7. Documentation/setup: Russian README, ASSIGNMENT.md, one concise MuJoCo guide,
   experiment/report templates, Docker/CI, public generic parameter explorer.
8. Acceptance: clean install, tests, eight scene previews and physical probes,
   PPO/data/BC/eval integration, training feasibility evidence and explicit measured
   hardware/limitations; final independent review; publish under verified dpaleyev.

## Interfaces and design rulings

- `physical_ai.scenes.build_scene(robot, task, parameters)` returns MJCF; committed
  XML uses the same generator. Two robot names and four task names form the matrix.
- `ManipulationEnv(robot, task, observation, parameters, max_steps)` implements Gym
  reset/step; `bc_observation()` has only image/proprio; `get_privileged_state()` is
  separate. Action is Cartesian delta xyz plus gripper, with ready-made IK adapter.
- Per-episode arrays are `rgb[T,H,W,3]`, `proprio[T,P]`, `actions[T,4]`, done flags;
  metadata records seed, scene parameters, expert hash and task/robot/schema version.
- PPO implementation uses Stable Baselines3 with CPU MuJoCo and subprocess workers.
  This preserves the first practice's algorithm while removing MJX/CPU transfer
  mismatch and custom RSL-RL adapter burden. GPU optional for vision BC.
- Existing practice sources are references, not blindly copied into a new license.
- Hidden perturbations live solely in reviewer/. General edits expose all bodies,
  geoms, lights and camera parameters without task-specific robustness recipes.
- Fresh independent repository is the requested isolation; no worktree of a practice.

## Review focus

Physical tasks must not use welded/teleported grasps. Color matching follows current
appearance, identity swap follows reset identity. BC receives no privileged state.
Hidden configs never enter student export/history. Weak expert collection fails
with a useful bounded error. Full training claims require full training evidence.

## Approved scope adjustment
The user authorized simplifying difficult tasks. Replace three-transfer swap with
one-transfer cup_distractor and three-cup sorting with one-cup color_match.
Update private advanced variants accordingly; do not disclose them in public docs.
