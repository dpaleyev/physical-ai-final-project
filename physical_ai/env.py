"""One MuJoCo environment for RL, demonstrations and evaluation."""
import os
if not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
import mujoco
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from physical_ai.scenes import ROBOTS, build_scene


class ManipulationEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, robot='ur5e', task='cup_plate', parameters=None,
                 max_steps=1000, observation='state', render_mode=None, reward_stage='place'):
        super().__init__()
        self.robot, self.task = robot, task
        self.parameters = parameters or {}
        self.max_steps, self.observation, self.render_mode = max_steps, observation, render_mode
        if reward_stage not in ('reach', 'lift', 'place'):
            raise ValueError('reward_stage must be reach, lift or place')
        self.reward_stage = reward_stage
        self.model = mujoco.MjModel.from_xml_string(build_scene(robot, task, parameters))
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self.n_arm = ROBOTS[robot][2]
        self.arm_jids = self.model.actuator_trnid[:self.n_arm, 0]
        self.arm_q = self.model.jnt_qposadr[self.arm_jids]
        self.arm_v = self.model.jnt_dofadr[self.arm_jids]
        self.jaw_jids = [self.model.joint(n+'_jaw').id for n in ('left', 'right')]
        self.jaw_q = self.model.jnt_qposadr[self.jaw_jids]
        self.jaw_v = self.model.jnt_dofadr[self.jaw_jids]
        self.object_names = [f'cup{i}' for i in range(3 if task=='sort' else 2 if task=='swap' else 1)]
        self.grasp_id = self.model.site('grasp').id
        self.action_space = spaces.Box(-1, 1, (4,), np.float32)
        self.renderer = None
        self.reset(seed=0)
        state = self.get_privileged_state()
        self.observation_space = spaces.Box(-np.inf, np.inf, state.shape, np.float32)
        if observation != 'state':
            raise ValueError('Gym observation is state; use bc_observation() for BC')

    @property
    def proprio_dim(self):
        return 2*(self.n_arm+2)

    def _ik(self, target, iterations=8, initial=None):
        d, m = self._ik_data, self.model
        d.qpos[:] = self.data.qpos
        if initial is not None:
            d.qpos[self.arm_q] = initial
        jp, jr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
        desired = np.diag([1., -1., -1.])
        for _ in range(iterations):
            mujoco.mj_forward(m,d)
            r = d.site_xmat[self.grasp_id].reshape(3,3)
            current_q, inverse_q, error_q = np.empty(4), np.empty(4), np.empty(4)
            mujoco.mju_mat2Quat(current_q, r.reshape(-1))
            mujoco.mju_negQuat(inverse_q, current_q)
            mujoco.mju_mulQuat(error_q, np.array([0.,1.,0.,0.]), inverse_q)
            if error_q[0] < 0:
                error_q *= -1
            orient = np.empty(3)
            mujoco.mju_quat2Vel(orient, error_q, 1.)
            error = np.r_[target-d.site_xpos[self.grasp_id], orient]
            if np.linalg.norm(error) < .0005:
                break
            mujoco.mj_jacSite(m,d,jp,jr,self.grasp_id)
            j = np.vstack([jp[:,self.arm_v],jr[:,self.arm_v]])
            dq = j.T @ np.linalg.solve(j@j.T + .001*np.eye(6), error)
            d.qpos[self.arm_q] += np.clip(dq, -.12, .12)
            d.qpos[self.arm_q] = np.clip(d.qpos[self.arm_q], m.jnt_range[self.arm_jids,0], m.jnt_range[self.arm_jids,1])
        return d.qpos[self.arm_q].copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        seed_q = [-1.57,-1.57,1.57,-1.57,-1.57,0] if self.robot=='ur5e' else [0,.7,0,-1.7,0,1.,0]
        self.data.qpos[self.arm_q] = seed_q
        self.data.qpos[self.jaw_q] = .045
        self.ee_target = np.array([.43,0,.68])
        self.data.qpos[self.arm_q] = self._ik(self.ee_target, iterations=140)
        self.data.ctrl[:self.n_arm] = self.data.qpos[self.arm_q]
        self.data.ctrl[self.n_arm:] = .045
        for name in self.object_names:
            adr = self.model.joint(name+'_free').qposadr[0]
            self.data.qpos[adr:adr+2] += self.np_random.uniform(-.008,.008,2)
        mujoco.mj_forward(self.model,self.data)
        self.goals = np.array([self.data.body(f'target{i}').xpos + [0,0,.034] for i in range(len(self.object_names))])
        if self.task=='sort':
            import itertools
            cup_colors = np.array([self.model.geom(f'cup{i}_bottom').rgba[:3] for i in range(len(self.object_names))])
            plate_colors = np.array([self.model.geom(f'plate{i}').rgba[:3] for i in range(len(self.object_names))])
            orders = list(itertools.permutations(range(len(self.object_names))))
            order = min(orders, key=lambda p: np.linalg.norm(cup_colors-plate_colors[list(p)]))
            if np.max(np.abs(cup_colors-plate_colors[list(order)])) > .01:
                raise ValueError('Each cup requires exactly one matching plate color')
            self.goals = self.goals[list(order)]
        if self.task=='swap':
            self.goals = np.array([self.data.body(name).xpos.copy() for name in self.object_names])[::-1]
        self.steps = self.stable_steps = 0
        self._previous_score = self._potential()
        return self.get_privileged_state(), {'is_success':False}

    def proprio(self):
        return np.r_[self.data.qpos[self.arm_q], self.data.qpos[self.jaw_q],
                     self.data.qvel[self.arm_v], self.data.qvel[self.jaw_v]].astype(np.float32)

    def get_privileged_state(self):
        objects = []
        for i,name in enumerate(self.object_names):
            b = self.data.body(name)
            objects.extend([*b.xpos, *b.xquat, *self.goals[i]])
        return np.r_[self.proprio(), self.data.site('grasp').xpos, self.ee_target, objects].astype(np.float32)

    def bc_observation(self):
        return {'rgb':self.render(), 'proprio':self.proprio()}

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=84, width=84)
        self.renderer.update_scene(self.data, camera='overview')
        return self.renderer.render().copy()

    def placement_complete(self):
        for i,name in enumerate(self.object_names):
            b = self.data.body(name)
            if np.linalg.norm(b.xpos[:2]-self.goals[i,:2]) > .035:
                return False
            if abs(b.xpos[2]-self.goals[i,2]) > .012 or b.xmat.reshape(3,3)[2,2] < .92:
                return False
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,b.id,vel,0)
            if np.linalg.norm(vel[:3]) > .3 or np.linalg.norm(vel[3:]) > .03:
                return False
            if np.linalg.norm(self.data.site('grasp').xpos-b.xpos) < .09:
                return False
        return bool(np.min(self.data.qpos[self.jaw_q]) > .03)

    def success(self):
        return self.stable_steps >= 8

    def _potential(self):
        ee = self.data.site('grasp').xpos
        terms = []
        for i,name in enumerate(self.object_names):
            p = self.data.body(name).xpos
            reach = 1-np.tanh(8*np.linalg.norm(ee-p))
            lift = np.clip((p[2]-.435)/.12,0,1)
            place = 1-np.tanh(7*np.linalg.norm(p-self.goals[i]))
            terms.append(reach if self.reward_stage=='reach' else reach+2*lift if self.reward_stage=='lift' else reach+2*lift+4*place)
        return float(sum(terms))

    def step(self, action):
        action = np.asarray(action,dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('Expected finite action of shape (4,)')
        action = np.clip(action,-1,1)
        self.ee_target = np.clip(self.ee_target + action[:3]*.018, [.22,-.32,.435], [.74,.32,.82])
        start_ctrl = self.data.ctrl.copy()
        end_ctrl = np.r_[self._ik(self.ee_target), [(action[3]+1)*.0225]*2]
        for fraction in np.linspace(.1, 1., 10):
            self.data.ctrl[:] = start_ctrl + fraction*(end_ctrl-start_ctrl)
            mujoco.mj_step(self.model,self.data,nstep=5)
        mujoco.mj_forward(self.model,self.data)
        self.steps += 1
        self.stable_steps = self.stable_steps+1 if self.placement_complete() else 0
        score = self._potential()
        reward = 5*(score-self._previous_score) + .02*score - .002*float(action[:3]@action[:3])
        self._previous_score = score
        failed = any(self.data.body(name).xpos[2]<.32 for name in self.object_names)
        ok = self.success()
        reward += 20*ok - 5*failed
        return self.get_privileged_state(), float(reward), bool(ok or failed), self.steps>=self.max_steps, {'is_success':ok,'failure':'dropped' if failed else 'timeout' if self.steps>=self.max_steps else '', 'stable_steps':self.stable_steps}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
