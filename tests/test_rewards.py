"""Regression: actually finishing must beat holding an object near its goal."""

import numpy as np
from physical_ai.env import ManipulationEnv
from reviewer.physical_probe import move


def decision_return(finish):
    with ManipulationEnv("ur5e", "cup_plate") as env:
        source = env.data.body("cup0").xpos.copy() + [0, 0, 0.006]
        goal = env.goals[0]
        prefix = [
            ([*source[:2], 0.65], 1, 45),
            (source, 1, 35),
            (source, -1, 18),
            ([*source[:2], 0.65], -1, 40),
            ([*goal[:2], 0.50], -1, 60),
        ]
        for target, jaw, steps in prefix:
            move(env, target, jaw, steps)
        rewards = []
        actual_step = env.step

        def record(action):
            result = actual_step(action)
            rewards.append(result[1])
            return result

        env.step = record
        if finish:
            offset = env.data.body("cup0").xpos.copy() - env.ee_target
            target = goal - offset
            move(env, target + [0, 0, 0.005], -1, 35)
            move(env, target + [0, 0, 0.005], 1, 18)
            move(env, [*target[:2], 0.65], 1, 40)
        while env.steps < 1000 and not env.success():
            env.step([0, 0, 0, 1 if finish else -1])
        return sum(reward * 0.995**i for i, reward in enumerate(rewards)), env.success()


def test_finishing_beats_stalling_at_goal():
    stall_return, stall_success = decision_return(False)
    finish_return, finish_success = decision_return(True)
    assert finish_success and not stall_success
    assert finish_return > stall_return


def test_potential_guides_release_and_retreat_at_goal():
    import mujoco

    with ManipulationEnv() as env:
        adr = int(env.model.joint("cup0_free").qposadr[0])
        env.data.qpos[adr : adr + 3] = env.goals[0]
        env.data.qpos[env.arm_q] = env._ik(env.goals[0] + [0, 0, 0.008], iterations=140)
        env.data.qpos[env.jaw_q] = 0.02
        mujoco.mj_forward(env.model, env.data)
        holding = env._potential()
        env.data.qpos[env.arm_q] = env._ik(env.goals[0] + [0, 0, 0.16], iterations=140)
        env.data.qpos[env.jaw_q] = 0.045
        mujoco.mj_forward(env.model, env.data)
        released = env._potential()
        assert released > holding


def test_untouched_distractor_contact_jitter_does_not_block_placement():
    import numpy as np
    import mujoco
    from physical_ai.env import ManipulationEnv
    from reviewer.physical_probe import transfer

    with ManipulationEnv("ur5e", "cup_distractor") as env:
        env.reset(seed=1)
        transfer(env, "cup0", env.goals[0])
        for _ in range(120):
            env.step([0, 0, 0, 1])
        neighbor = env.data.body("cup1")
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, neighbor.id, velocity, 0
        )
        assert not env._distractor_moved
        assert np.linalg.norm(neighbor.xpos[:2] - env.goals[1, :2]) < 0.02
        assert neighbor.xmat.reshape(3, 3)[2, 2] > 0.99
        assert np.linalg.norm(velocity[:3]) > 0.6
        assert env.placement_complete()
