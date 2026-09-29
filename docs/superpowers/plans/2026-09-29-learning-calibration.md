# RL calibration within five hours

Authorized by user: achieve useful SR on all eight robot/task pairs, collect BC
training data with those policies; simplify overly difficult tasks if necessary.
Start 2026-09-28 23:13:40 UTC; hard deadline 2026-09-29 04:13:40 UTC.

Acceptance: >=80% success on 50 independent base episodes for each pair; successful
RL-sourced train/validation demonstration collection for each pair; BC training and
inference verified on those real datasets. No scripted policy relabeled as RL.
Record intermediate measurements and training provenance. Keep private validation
variants out of student export; update public task definitions if simplified.

1. Diagnose actual trained-policy trajectories, not only terminal success counts.
2. Fix the identified learning bottleneck with bounded experiments; prefer keeping
   physical pick/place. If progress stalls, simplify tasks within authorization.
3. Train eight policies, evaluate fresh seeds and collect successful demonstrations.
4. Verify BC on actual RL data, tests, task/advanced consistency, publication.

Initial diagnosis: old place policy brings cup to within 1.4-3.3 cm of plate but
keeps gripper closed and close to cup; new potential formulation has no explicit
release/retreat gradient. Lift curriculum has learned intermittent lifts. First
experiment should target release signal and retained exploration, using old PPO
weights as initialization, with unchanged physical success criteria.
