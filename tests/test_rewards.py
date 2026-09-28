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
