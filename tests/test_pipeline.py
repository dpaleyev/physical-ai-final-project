import importlib.util
import numpy as np
import pytest


def test_pipeline_modules_exist():
    for name in ['data', 'bc', 'rl', 'evaluate']:
        assert importlib.util.find_spec('physical_ai.'+name) is not None


def test_episode_roundtrip_rejects_misalignment_and_overlap(tmp_path):
    from physical_ai.data import save_episode, EpisodeDataset, check_disjoint
    meta = dict(robot='ur5e',task='cup_plate',seed=42,parameters={},expert_sha256='abc',success=True,source='ppo')
    rgb=np.zeros((3,84,84,3),np.uint8);prop=np.zeros((3,16),np.float32);actions=np.zeros((3,4),np.float32)
    with pytest.raises(ValueError, match='length'):
        save_episode(tmp_path/'bad.npz',rgb,prop,actions[:2],meta)
    save_episode(tmp_path/'a.npz',rgb,prop,actions,meta)
    ds=EpisodeDataset(tmp_path)
    assert len(ds)==3 and ds[0][0].shape==(3,84,84)
    with pytest.raises(ValueError, match='overlap'):
        check_disjoint(ds,ds)


def test_bc_training_and_checkpoint_roundtrip(tmp_path):
    import torch
    from physical_ai.bc import BCPolicy, save_bc, load_bc
    torch.manual_seed(0)
    model=BCPolicy(16,True)
    rgb=torch.rand(4,3,84,84);prop=torch.randn(4,16);target=torch.ones(4,4)*.3
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    before=model(rgb,prop).detach().clone()
    loss=(model(rgb,prop)-target).square().mean();loss.backward();opt.step()
    assert not torch.equal(before,model(rgb,prop))
    path=tmp_path/'bc.pt';save_bc(path,model,dict(robot='ur5e',task='cup_plate'))
    loaded,meta=load_bc(path)
    torch.testing.assert_close(model(rgb,prop),loaded(rgb,prop))


def test_ppo_updates_and_reloads(tmp_path):
    from physical_ai.rl import train, load_rl
    from physical_ai.env import ManipulationEnv
    out=tmp_path/'ppo'
    train('ur5e','cup_plate',out,total_steps=64,n_steps=32,n_envs=1,seed=4)
    policy,meta=load_rl(out/'last.zip','ur5e','cup_plate')
    assert meta['algorithm']=='PPO' and meta['timesteps']>=64
    with ManipulationEnv() as env:
        obs,_=env.reset(seed=5)
        action,_=policy.predict(obs,deterministic=True)
        assert action.shape==(4,) and np.isfinite(action).all()
    with pytest.raises(ValueError,match='robot/task'):
        load_rl(out/'last.zip','iiwa14','cup_plate')


def test_wilson_interval_and_rollout_timeout():
    from physical_ai.evaluate import wilson_interval, evaluate
    assert wilson_interval(0,10)[0]==0
    low,high=wilson_interval(5,10)
    assert .23<low<.24 and .76<high<.77
    result=evaluate('ur5e','cup_plate',lambda obs: np.array([0,0,0,1]),'state',seeds=[1,2],max_steps=2)
    assert result['success_rate']==0
    assert all(x['failure']=='timeout' for x in result['episodes'])
