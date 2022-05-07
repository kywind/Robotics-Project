import cv2
import gym
import numpy as np

from gym import spaces


class LocoTransformerEnv(gym.Env):

    def __init__(self, env: gym.Env):
        super(LocoTransformerEnv, self).__init__()
        self._env = env # (BaseDisplacement[3], Depth[4096], IMU[4], Motor[12])
        self.proprio_history = []
        self.depth_history = []
        proprio_lower_bound = np.concatenate([env.observation_space.low[:3],
                                              env.observation_space.low[-16:],
                                              env.action_space.low])
        proprio_upper_bound = np.concatenate([env.observation_space.high[:3],
                                              env.observation_space.high[-16:],
                                              env.action_space.high])
        depth_lower_bound = env.observation_space.low[3:-16]
        depth_upper_bound = env.observation_space.high[3:-16]
        lower_bound = np.concatenate([proprio_lower_bound] * 3 + [depth_lower_bound] * 4)
        upper_bound = np.concatenate([proprio_upper_bound] * 3 + [depth_upper_bound] * 4)
        self.observation_space = spaces.Box(lower_bound, upper_bound)
        self.action_space = env.action_space
        self.n_iter = 0
    
    def step(self, action, **kwargs):
        raw_observation, reward, done, info = self._env.step(action, **kwargs)
        proprio_observation = np.concatenate([raw_observation[:3],
                                              raw_observation[-16:],
                                              action])
        depth_observation = raw_observation[3:-16]
        # cv2.imwrite('temp/%d.png' % np.random.randint(100), np.maximum(depth_observation.reshape(64, 64), 0) * 255)
        self.proprio_history.pop(0)
        self.proprio_history.append(proprio_observation)
        self.depth_history.pop(0)
        self.depth_history.append(depth_observation)
        observation = np.concatenate(self.proprio_history + self.depth_history)
        self.n_iter += 1
        if self.n_iter == 1000:
            done = True
        # print(self.n_iter)
        return observation, reward, done, info

    def reset(self, **kwargs):
        self.n_iter = 0
        raw_observation, _ = self._env.reset(**kwargs)
        proprio_observation = np.concatenate([raw_observation[:3],
                                              raw_observation[-16:],
                                              np.zeros(self.action_space.shape[0])])
        depth_observation = raw_observation[3:-16]
        self.proprio_history = [proprio_observation] * 3
        self.depth_history = [depth_observation] * 4
        observation = np.concatenate(self.proprio_history + self.depth_history)
        return observation

    def render(self, mode):
        return self._env.render(mode)

    def close(self):
        self._env.close()