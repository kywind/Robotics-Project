# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from rlschool.quadrupedal.envs import locomotion_gym_env
from rlschool.quadrupedal.envs import locomotion_gym_config
from rlschool.quadrupedal.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from rlschool.quadrupedal.envs.env_wrappers import trajectory_generator_wrapper_env
from rlschool.quadrupedal.envs.env_wrappers import simple_openloop
from rlschool.quadrupedal.envs.env_wrappers import simple_forward_task
from rlschool.quadrupedal.envs.sensors import robot_sensors
from rlschool.quadrupedal.robots import a1
from rlschool.quadrupedal.robots import robot_config
from rlschool.quadrupedal.envs.env_wrappers.gait_generator_env import GaitGeneratorWrapperEnv


SENSOR_MODE = {"dis":1,"motor":2,"imu":3,"contact":0,"footpose":0,"ETG":0,"visual":1}
# 
def build_regular_env(robot_class,
                      motor_control_mode,
                      param,
                      sensor_mode = {"dis":1,"imu":1,"motor":1,"contact":1},
                      gait = 0,
                      normal=0,
                      enable_rendering=False,
                      task_mode = "plane",
                      on_rack=False,
                      filter = 0,
                      action_space = 0,
                      random = False,
                      action_limit=(0.75, 0.75, 0.75),
                      wrap_trajectory_generator=True):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  # sim_params.sim_time_step_s = 0.0025
  # sim_params.num_action_repeat = 16
  sim_params.sim_time_step_s = 0.004
  sim_params.num_action_repeat = 6
  # sim_params.sim_time_step_s = 0.002
  # sim_params.num_action_repeat = 13
  sim_params.enable_action_interpolation = False
  if filter:
    sim_params.enable_action_filter = True
  else:
    sim_params.enable_action_filter = False
  sim_params.enable_clip_motor_commands = False
  sim_params.robot_on_rack = on_rack
  dt = sim_params.num_action_repeat*sim_params.sim_time_step_s

#   sim_params.sim_time_step_s = 1./500.

  gym_config = locomotion_gym_config.LocomotionGymConfig(
      simulation_parameters=sim_params)
  sensors = []
  noise = True if ("noise" in sensor_mode and sensor_mode["noise"]) else False
  # print(sensor_mode)
  if sensor_mode["dis"]:
    sensors.append(robot_sensors.BaseDisplacementSensor(convert_to_local_frame=True,normal=normal,noise=noise))
  if sensor_mode["imu"]==1:
    sensors.append(robot_sensors.IMUSensor(channels=["R", "P", "Y","dR", "dP", "dY"],normal=normal,noise=noise))
  elif sensor_mode["imu"]==2:
    sensors.append(robot_sensors.IMUSensor(channels=["dR", "dP", "dY"],noise=noise))
  elif sensor_mode["imu"]==3:
    sensors.append(robot_sensors.IMUSensor(channels=["R", "P", "dR", "dP"],noise=noise))
  if sensor_mode["motor"]==1:
    sensors.append(robot_sensors.MotorAngleAccSensor(num_motors=a1.NUM_MOTORS,normal=normal,noise=noise,dt=dt))
  elif sensor_mode["motor"]==2:
    sensors.append(robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS,noise=noise))
  if sensor_mode["contact"] == 1:
    sensors.append(robot_sensors.FootContactSensor())
  elif sensor_mode["contact"] == 2:
    sensors.append(robot_sensors.SimpleFootForceSensor())
  if sensor_mode["footpose"]:
    sensors.append(robot_sensors.FootPoseSensor(normal=normal))
  if sensor_mode["visual"]:
    sensors.append(robot_sensors.DepthSensor())
  # print(sensors)

  task = simple_forward_task.RegularizedForwardTask(param)
  # task = simple_forward_task.SimpleForwardTask(param)

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                            param = param,
                                            robot_class=robot_class,
                                            robot_sensors=sensors,
                                            random=random,
                                            task=task,
                                            task_mode=task_mode)
  
  env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(
      env)
      
  if gait!=0 and (motor_control_mode
      == robot_config.MotorControlMode.POSITION):
    env = GaitGeneratorWrapperEnv(env,gait_mode=gait)
  elif (motor_control_mode
      == robot_config.MotorControlMode.POSITION) and wrap_trajectory_generator:
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
        env,
        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
            action_limit=0.75,action_space=action_space)) #origin action_limit=action_limit

  return env