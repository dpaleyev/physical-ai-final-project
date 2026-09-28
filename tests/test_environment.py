import importlib.util
import numpy as np
import pytest


def test_environment_is_implemented():
    assert importlib.util.find_spec('physical_ai.env') is not None


@pytest.mark.parametrize('robot', ['ur5e', 'iiwa14'])
@pytest.mark.parametrize('task', ['cup_plate', 'cup_shelf', 'swap', 'sort'])
def test_reset_step_and_observation_contract(robot, task):
    from physical_ai.env import ManipulationEnv
    with ManipulationEnv(robot, task) as env:
        first, _ = env.reset(seed=21)
        pose = env.data.qpos.copy()
        second, _ = env.reset(seed=21)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(pose, env.data.qpos)
        np.testing.assert_allclose(env.data.site('grasp').xmat.reshape(3,3), np.diag([1,-1,-1]), atol=.01)
        assert env.action_space.shape == (4,)
        assert env.observation_space.contains(first)
        assert not env.success()
        for _ in range(12):
            obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0, 1]))
            assert np.isfinite(obs).all() and np.isfinite(reward)
        for name in env.object_names:
            assert .41 < env.data.body(name).xpos[2] < .48
        assert np.linalg.norm(env.data.site('grasp').xpos - env.ee_target) < .025
        with pytest.raises(ValueError):
            env.step(np.array([np.nan, 0, 0, 0]))


def test_success_requires_release_stability_and_all_objects():
    from physical_ai.env import ManipulationEnv
    import mujoco
    with ManipulationEnv('ur5e', 'sort') as env:
        env.reset(seed=5)
        for i, name in enumerate(env.object_names):
            adr = env.model.joint(name + '_free').qposadr[0]
            env.data.qpos[adr:adr+3] = env.goals[i]
        mujoco.mj_forward(env.model, env.data)
        env.data.qvel[:] = 0
        assert env.placement_complete()
        assert not env.success()  # one instant is insufficient
        adr = env.model.joint('cup1_free').qposadr[0]
        env.data.qpos[adr] += .12
        mujoco.mj_forward(env.model, env.data)
        assert not env.placement_complete()


def test_bc_observation_does_not_contain_object_state():
    from physical_ai.env import ManipulationEnv
    with ManipulationEnv('iiwa14', 'swap') as env:
        env.reset(seed=0)
        obs = env.bc_observation()
        assert set(obs) == {'rgb', 'proprio'}
        assert obs['rgb'].shape == (84, 84, 3)
        assert obs['rgb'].dtype == np.uint8
        assert obs['proprio'].shape == (18,)
