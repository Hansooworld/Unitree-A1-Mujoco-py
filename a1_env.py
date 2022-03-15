import math,sys,os
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import gym

# Convert quaternion to Euler angle 
def quaternion_to_euler_angle(w, x, y, z):
	ysqr = y * y
	
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))
	
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))
	
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))
	
	return X, Y, Z

class A1Env(mujoco_env.MujocoEnv,utils.EzPickle):
    def __init__(self,_VERBOSE=True,_frame_skip=5, _ctrl=0, _body=0, _jump=0, _angle=0, _vel=0, _wandb=True):
        
        # Some parameters


        # Setting path for extracting mujoco file
        xmlPath = os.path.dirname(os.path.realpath(__file__)) + '/xml/a1mjcf.xml'
        mujoco_env.MujocoEnv.__init__(self,xmlPath, frame_skip=_frame_skip)
        utils.EzPickle.__init__(self)

        # Limitation of joint angles
        
        # observation dimension is same as sum of dim(joint)
        self.obsDim = self.observation_space.shape[0]
        # action dimension is the number of actuators
        self.actDim = self.action_space.shape[0]

        # Print out
        self.verbose = _VERBOSE
        if self.verbose:
            print("A1 Environment.")
            # self.dt = self.model.opt.timestep * self.frame_skip
            print("Obs Dim:[{}] Act Dim:[{}], dt:[{}]".format(self.obsDim,self.actDim,self.dt))

    def step(self, a):
        xposbefore = self.get_body_com("Body")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("Body")[0]
        zpos = self.get_body_com("Body")[2]
        # Forward reward
        forward_reward = (xposafter - xposbefore)/self.dt 
        # Control cost 
        ctrl_cost = np.square(a).sum()
        # Survive reward
        survive_reward = 1.0 # 1.0
        reward = forward_reward - ctrl_cost + survive_reward 
        state = self.state_vector()
        r,p,y = quaternion_to_euler_angle(state[3],state[4],state[5],state[6])
        # How to set the done condition?
        notdone = np.isfinite(state).all() \
            and abs(r) < 170
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done,\
            dict(
                reward_forward=forward_reward,
                reward_control=-ctrl_cost,
                reward_survive=survive_reward)

    # body(link)에 작용하는 torque & force 고려X
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        qpos = self.init_qpos + 0*self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + 0*self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

if __name__ == "__main__":
    env = A1Env()
    while True:
        env.render()
        b= np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        a = np.random.randn(12) * 20
        env.step(a)