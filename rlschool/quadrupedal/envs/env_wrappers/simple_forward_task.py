# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation/blob/master/motion_imitation/envs/env_wrappers/simple_forward_task.py

"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from rlschool.quadrupedal.envs.utilities import pose3d
from pybullet_utils import transformations

class SimpleForwardTask(object):
  """Default empy task."""
  def __init__(self,param):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.param = param
    self.last_foot = np.zeros(4)

  def __call__(self, env,action,torques):
    return self.reward(env,action,torques)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    return False


  def reward(self, env,action,torques):
    """Get the reward without side effects."""
    vel_reward = self._calc_vel_reward()
    return vel_reward
  
  def _calc_vel_reward(self):
      return self.current_base_pos[0] - self.last_base_pos[0]
  

class RegularizedForwardTask_old(object):
  """Regularized task."""
  def __init__(self,param):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.param = param
    self.last_foot = np.zeros(4)

  def __call__(self, env,action,torques):
    return self.reward(env,action,torques)

  def reset(self, env):
    """Resets the internal state of the task."""
    # print('reseted')
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    rot_quat = env._robot.GetBaseOrientation()
    rot_mat = env._pybullet_client.getMatrixFromQuaternion(rot_quat)
    foot_pos = env._robot.GetFootPositionsInBaseFrame()
    pose = env._robot.GetBaseRollPitchYaw()

    return (
      rot_mat[-1] < 0.5 or
      np.mean(foot_pos[:, -1]) > -0.1 or
      np.max(foot_pos[:, -1]) > 0 or
      abs(pose[-1]) > np.pi / 2
    )

    # info = {}
    # info["rot_quat"] = env._robot.GetBaseOrientation()
    # info["rot_mat"] = env._pybullet_client.getMatrixFromQuaternion(info["rot_quat"])
    # info["base"] = env._robot.GetBasePosition()
    # info["footposition"] = env._robot.GetFootPositionsInBaseFrame()
    # info["pose"] = env._robot.GetBaseRollPitchYaw()
    # info["real_contact"] = env._robot.GetFootContacts()
    # info["joint_angle"] = env._robot.GetMotorAngles()
    # info["drpy"] = env._robot.GetBaseRollPitchYawRate()
    # info["env_info"] = env.env_info
    # info["energy"] = env._robot.GetEnergyConsumptionPerControlStep()
    # info["latency"] = env._robot.GetControlLatency()
    # return rot_mat[-1] < 0.5  or np.mean(footz) > -0.1 or np.max(footz) > 0 or abs(pose[-1]) > np.pi / 2


  def reward(self, env, action, torques):
    """Get the reward without side effects."""
    vel_reward = self._calc_vel_reward() / 0.026
    # print(vel_reward)
    # print(self.current_base_pos[0])
    energy = - 5e-4 * (torques ** 2).sum() / 13
    # energy = - 5e-3 * env._robot.GetEnergyConsumptionPerControlStep()
    # print(energy)
    alive = 0.1
    return vel_reward + energy + alive
  
  def _calc_vel_reward(self):
      return self.current_base_pos[0] - self.last_base_pos[0]

  
class RegularizedForwardTask(object):
  """Regularized task."""
  def __init__(self,param):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.param = param
    self.last_foot = np.zeros(4)

  def __call__(self, env,action,torques):
    return self.reward(env,action,torques)

  def reset(self, env):
    """Resets the internal state of the task."""
    # print('reseted')
    self._env = env
    self.last_base_pos = np.array(env.robot.GetBasePosition())
    self.current_base_pos = self.last_base_pos

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = np.array(env.robot.GetBasePosition())

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    rot_quat = env._robot.GetBaseOrientation()
    rot_mat = env._pybullet_client.getMatrixFromQuaternion(rot_quat)
    foot_pos = env._robot.GetFootPositionsInBaseFrame()
    pose = env._robot.GetBaseRollPitchYaw()

    return (
      rot_mat[-1] < 0.5 or
      np.mean(foot_pos[:, -1]) > -0.1 or
      np.max(foot_pos[:, -1]) > 0 or
      abs(pose[-1]) > np.pi / 2
    )

  def reward(self, env, action, torques):
    """Get the reward without side effects."""

    """new_new"""
    vel = self._calc_vel() 
    vel_reward = min(vel[0], 1)
    yaw = env._robot.GetBaseRollPitchYaw()[-1]
    drift_reward = - np.abs(np.sin(yaw) * vel[0] - np.cos(yaw) * vel[1])
    energy_reward = - 5e-5 * (torques ** 2).sum(1).mean()
    y_reward = - 0.1 * np.abs(env._robot.GetBasePosition()[1]) ** 2
    alive_reward = 0.1
    return vel_reward + drift_reward + energy_reward + y_reward + alive_reward

    # """modified_new"""
    # vel = self._calc_vel() 
    # vel_reward = min(vel, 1)
    # energy_reward = - 5e-5 * (torques ** 2).sum(1).mean()
    # y_reward = - 0.2 * np.abs(env._robot.GetBasePosition()[1]) ** 2
    # alive_reward = 0.1
    # return vel_reward + energy_reward + y_reward + alive_reward

    # """modified_new_005"""
    # vel = self._calc_vel() 
    # vel_reward = min(vel, 1)
    # energy_reward = - 5e-5 * (torques ** 2).sum(1).mean()
    # y_reward = - 0.05 * np.abs(env._robot.GetBasePosition()[1]) ** 2
    # alive_reward = 0.1
    # return vel_reward + energy_reward + y_reward + alive_reward

    # """big_reward"""
    # vel = self._calc_vel() 
    # vel_reward = min(vel[0], 50)
    # yaw = env._robot.GetBaseRollPitchYaw()[-1]
    # drift_reward = - np.abs(np.sin(yaw) * vel[0] - np.cos(yaw) * vel[1])
    # # print(env._robot.GetMotorGains())
    # energy_reward = - 5e-3 * (torques ** 2).sum(1).mean()
    # y_reward = - 0.1 * np.abs(env._robot.GetBasePosition()[1]) ** 2
    # alive_reward = 0.1
    # # print(vel_reward, drift_reward, energy_reward, y_reward, alive_reward)
    # return vel_reward + drift_reward + energy_reward + y_reward + alive_reward

  def _calc_vel(self):
    """new_new"""
    return (self.current_base_pos - self.last_base_pos) / 0.024

    # """big_reward"""
    # return (self.current_base_pos - self.last_base_pos) / 0.04 * 100

    # """modified_new & modified_new_005"""
    # return (self.current_base_pos[0] - self.last_base_pos[0]) / 0.024
