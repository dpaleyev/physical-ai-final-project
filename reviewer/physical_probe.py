"""Author-only physical feasibility probe. Scripted IK, NOT an RL expert."""
import argparse
import json
from pathlib import Path
import numpy as np
from physical_ai.env import ManipulationEnv


def move(env, target, jaw, steps=35, frames=None):
    info = {}
    for _ in range(steps):
        delta = np.clip((np.asarray(target)-env.ee_target)/.018, -.55, .55)
        _, _, done, truncated, info = env.step(np.r_[delta, jaw])
        if frames is not None:
            frames.append(env.render())
        if info['is_success']:
            return info
    return info


def transfer(env, name, goal, frames=None):
    source = env.data.body(name).xpos.copy()
    source[2] += .006
    high = max(.65, goal[2]+.18)
    move(env, [source[0],source[1],high],1,45,frames)
    move(env, source,1,35,frames)
    move(env, source,-1,18,frames)
    move(env, [source[0],source[1],high],-1,40,frames)
    lifted = float(env.data.body(name).xpos[2])
    move(env, [goal[0],goal[1],high],-1,45,frames)
    move(env, np.asarray(goal)+[0,0,.012],-1,35,frames)
    move(env, np.asarray(goal)+[0,0,.012],1,18,frames)
    result = move(env, [goal[0],goal[1],high],1,40,frames)
    return lifted, result


def probe(robot, task, seed=0, video=None, parameters=None):
    frames = [] if video else None
    with ManipulationEnv(robot, task, parameters=parameters, max_steps=1500) as env:
        env.reset(seed=seed)
        lifted = []
        if task=='swap':
            sequence = [('cup0',np.array([.62,0,.434])), ('cup1',env.goals[1]),('cup0',env.goals[0])]
        else:
            sequence = list(zip(env.object_names,env.goals))
        for name, goal in sequence:
            height, info = transfer(env,name,goal,frames)
            lifted.append(height)
        for _ in range(20):
            if env.success(): break
            env.step([0,0,0,1])
        result = {'robot':robot,'task':task,'seed':seed,'success':env.success(),'lifted_z':lifted,
                  'positions':[env.data.body(n).xpos.tolist() for n in env.object_names], 'goals':env.goals.tolist()}
        if video:
            import imageio.v2 as imageio
            imageio.mimsave(video,frames,fps=20)
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--robot',default='ur5e'); p.add_argument('--task',default='cup_plate'); p.add_argument('--video'); args=p.parse_args()
    print(json.dumps(probe(args.robot,args.task,video=args.video),indent=2))
