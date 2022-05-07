# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation
"""Simple sensors related to the robot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import typing

from robots import minitaur_pose_utils
from rlschool.quadrupedal.envs.sensors import sensor

_ARRAY = typing.Iterable[float] #pylint: disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY] #pylint: disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any] #pylint: disable=invalid-name


class MotorAngleSensor(sensor.BoxSpaceSensor):
  """A sensor that reads motor angles from the robot."""

  def __init__(self,
               num_motors: int,
               noisy_reading: bool = True,
               observe_sine_cosine: bool = False,
               noise: bool = False,
               lower_bound: _FLOAT_OR_ARRAY = -np.pi,
               upper_bound: _FLOAT_OR_ARRAY = np.pi,
               name: typing.Text = "MotorAngle",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs MotorAngleSensor.
    Args:
      num_motors: the number of motors in the robot
      noisy_reading: whether values are true observations
      observe_sine_cosine: whether to convert readings to sine/cosine values for
        continuity
      lower_bound: the lower bound of the motor angle
      upper_bound: the upper bound of the motor angle
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._num_motors = num_motors
    self._noisy_reading = noisy_reading
    self._observe_sine_cosine = observe_sine_cosine
    self.noise = noise

    if observe_sine_cosine:
      super(MotorAngleSensor, self).__init__(
          name=name,
          shape=(self._num_motors * 2,),
          lower_bound=-np.ones(self._num_motors * 2),
          upper_bound=np.ones(self._num_motors * 2),
          dtype=dtype)
    else:
      super(MotorAngleSensor, self).__init__(
          name=name,
          shape=(self._num_motors,),
          lower_bound=lower_bound,
          upper_bound=upper_bound,
          dtype=dtype)

  def _get_observation(self) -> _ARRAY:
    if self._noisy_reading:
      motor_angles = self._robot.GetMotorAngles()
    else:
      motor_angles = self._robot.GetTrueMotorAngles()
    if self.noise:
      motor_angles += np.random.normal(0,5e-3,size=self._num_motors)
    if self._observe_sine_cosine:
      return np.hstack((np.cos(motor_angles), np.sin(motor_angles)))
    else:
      return motor_angles
  def reset(self):
    pass

class MotorAngleAccSensor(sensor.BoxSpaceSensor):
  """A sensor that reads motor angles from the robot."""

  def __init__(self,
               num_motors: int,
               noisy_reading: bool = True,
               observe_sine_cosine: bool = False,
               noise: bool = False,
               normal: int = 0,
               dt: float = 0.026,
               lower_bound: _FLOAT_OR_ARRAY = -np.pi,
               upper_bound: _FLOAT_OR_ARRAY = np.pi,
               name: typing.Text = "MotorAngleAcc",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs MotorAngleSensor.

    Args:
      num_motors: the number of motors in the robot
      noisy_reading: whether values are true observations
      observe_sine_cosine: whether to convert readings to sine/cosine values for
        continuity
      lower_bound: the lower bound of the motor angle
      upper_bound: the upper bound of the motor angle
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._num_motors = num_motors
    self._noisy_reading = noisy_reading
    self._observe_sine_cosine = observe_sine_cosine
    self.last_angle = np.zeros(self._num_motors)
    self.normal = normal
    self.first_time = True
    self.noise = noise
    self._mean = np.array([0,0.9,-1.8]*4+[0]*12)
    self._std = np.array([0.1]*12+[1]*12)
    self.dt = dt
    if observe_sine_cosine:
      super(MotorAngleAccSensor, self).__init__(
          name=name,
          shape=(self._num_motors * 2*2,),
          lower_bound=-np.ones(self._num_motors * 2),
          upper_bound=np.ones(self._num_motors * 2),
          dtype=dtype)
    else:
      super(MotorAngleAccSensor, self).__init__(
          name=name,
          shape=(self._num_motors*2,),
          lower_bound=lower_bound,
          upper_bound=upper_bound,
          dtype=dtype)

  def _get_observation(self) -> _ARRAY:
    if self._noisy_reading:
      motor_angles = np.asarray(self._robot.GetMotorAngles())
    else:
      motor_angles = np.asarray(self._robot.GetTrueMotorAngles())
    if self.first_time:
      motor_acc = np.zeros(self._num_motors)
      self.first_time = False
    else:
      motor_acc = (motor_angles - self.last_angle)/self.dt
    if self.noise:
      motor_angles += np.random.normal(0,1e-2,size=self._num_motors)
      motor_acc += np.random.normal(0,0.5,size=self._num_motors)
    self.last_angle = motor_angles
    # print(motor_acc)
    if self._observe_sine_cosine:
      return np.hstack((np.cos(motor_angles), np.sin(motor_angles)))
    else:
      if self.normal:
        return (np.concatenate((motor_angles,motor_acc))-self._mean)/self._std
      else:
        return np.concatenate((motor_angles,motor_acc))
  
  def reset(self):
    # print("reset now!")
    self.last_angle = np.zeros(self._num_motors)
    self.first_time = True

class MinitaurLegPoseSensor(sensor.BoxSpaceSensor):
  """A sensor that reads leg_pose from the Minitaur robot."""

  def __init__(self,
               num_motors: int,
               noisy_reading: bool = True,
               observe_sine_cosine: bool = False,
               lower_bound: _FLOAT_OR_ARRAY = -np.pi,
               upper_bound: _FLOAT_OR_ARRAY = np.pi,
               name: typing.Text = "MinitaurLegPose",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs MinitaurLegPoseSensor.

    Args:
      num_motors: the number of motors in the robot
      noisy_reading: whether values are true observations
      observe_sine_cosine: whether to convert readings to sine/cosine values for
        continuity
      lower_bound: the lower bound of the motor angle
      upper_bound: the upper bound of the motor angle
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._num_motors = num_motors
    self._noisy_reading = noisy_reading
    self._observe_sine_cosine = observe_sine_cosine

    if observe_sine_cosine:
      super(MinitaurLegPoseSensor, self).__init__(
          name=name,
          shape=(self._num_motors * 2,),
          lower_bound=-np.ones(self._num_motors * 2),
          upper_bound=np.ones(self._num_motors * 2),
          dtype=dtype)
    else:
      super(MinitaurLegPoseSensor, self).__init__(
          name=name,
          shape=(self._num_motors,),
          lower_bound=lower_bound,
          upper_bound=upper_bound,
          dtype=dtype)

  def _get_observation(self) -> _ARRAY:
    motor_angles = (
        self._robot.GetMotorAngles()
        if self._noisy_reading else self._robot.GetTrueMotorAngles())
    leg_pose = minitaur_pose_utils.motor_angles_to_leg_pose(motor_angles)
    if self._observe_sine_cosine:
      return np.hstack((np.cos(leg_pose), np.sin(leg_pose)))
    else:
      return leg_pose


class BaseDisplacementSensor(sensor.BoxSpaceSensor):
  """A sensor that reads displacement of robot base."""

  def __init__(self,
               lower_bound: _FLOAT_OR_ARRAY = -0.1,
               upper_bound: _FLOAT_OR_ARRAY = 0.1,
               convert_to_local_frame: bool = False,
               noise: bool = False,
               normal: int =0,
               dt: float = 0.026,
               name: typing.Text = "BaseDisplacement",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs BaseDisplacementSensor.

    Args:
      lower_bound: the lower bound of the base displacement
      upper_bound: the upper bound of the base displacement
      convert_to_local_frame: whether to project dx, dy to local frame based on
        robot's current yaw angle. (Note that it's a projection onto 2D plane,
        and the roll, pitch of the robot is not considered.)
      name: the name of the sensor
      dtype: data type of sensor value
    """

    self._channels = ["x", "y", "z"]
    self._num_channels = len(self._channels)

    super(BaseDisplacementSensor, self).__init__(
        name=name,
        shape=(self._num_channels,),
        lower_bound=np.array([lower_bound] * 3),
        upper_bound=np.array([upper_bound] * 3),
        dtype=dtype)

    datatype = [("{}_{}".format(name, channel), self._dtype)
                for channel in self._channels]
    self._datatype = datatype
    self._convert_to_local_frame = convert_to_local_frame
    self.dt = dt
    self.noise = noise
    self._last_yaw = 0
    self._last_base_position = np.zeros(3)
    self._current_yaw = 0
    self._current_base_position = np.zeros(3)
    self._mean = np.array([0]*3)
    self._std = np.array([0.1]*3)
    self.normal = normal


  def get_channels(self) -> typing.Iterable[typing.Text]:
    """Returns channels (displacement in x, y, z direction)."""
    return self._channels

  def get_num_channels(self) -> int:
    """Returns number of channels."""
    return self._num_channels

  def get_observation_datatype(self) -> _DATATYPE_LIST:
    """See base class."""
    return self._datatype

  def _get_observation(self) -> _ARRAY:
    """See base class."""
    dx, dy, dz = (self._current_base_position - self._last_base_position)/self.dt
    if self.noise:
      dx += np.random.normal(0,1e-2)
      dy += np.random.normal(0,1e-2)
      dz += np.random.normal(0,1e-2)
    if self._convert_to_local_frame:
      dx_local = np.cos(self._last_yaw) * dx + np.sin(self._last_yaw) * dy
      dy_local = -np.sin(self._last_yaw) * dx + np.cos(self._last_yaw) * dy
      if self.normal:
        return (np.array([dx_local, dy_local, dz])-self._mean)/self._std
      else:
        return np.array([dx_local, dy_local, dz])
    else:
      if self.normal:
        return (np.array([dx, dy, dz])-self._mean)/self._std
      else:
        return np.array([dx, dy, dz])

  def on_reset(self, env):
    """See base class."""
    self._current_base_position = np.array(self._robot.GetBasePosition())
    self._last_base_position = np.array(self._robot.GetBasePosition())
    self._current_yaw = self._robot.GetBaseRollPitchYaw()[2]
    self._last_yaw = self._robot.GetBaseRollPitchYaw()[2]

  def on_step(self, env):
    """See base class."""
    self._last_base_position = self._current_base_position
    self._current_base_position = np.array(self._robot.GetBasePosition())
    self._last_yaw = self._current_yaw
    self._current_yaw = self._robot.GetBaseRollPitchYaw()[2]
  def reset(self):
    pass

class IMUSensor(sensor.BoxSpaceSensor):
  """An IMU sensor that reads orientations and angular velocities."""

  def __init__(self,
               channels: typing.Iterable[typing.Text] = None,
               noisy_reading: bool = True,
               normal: int = 0,
               noise: bool = False,
               lower_bound: _FLOAT_OR_ARRAY = None,
               upper_bound: _FLOAT_OR_ARRAY = None,
               name: typing.Text = "IMU",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs IMUSensor.

    It generates separate IMU value channels, e.g. IMU_R, IMU_P, IMU_dR, ...

    Args:
      channels: value channels wants to subscribe. A upper letter represents
        orientation and a lower letter represents angular velocity. (e.g. ['R',
        'P', 'Y', 'dR', 'dP', 'dY'] or ['R', 'P', 'dR', 'dP'])
      noisy_reading: whether values are true observations
      lower_bound: the lower bound IMU values
        (default: [-2pi, -2pi, -2000pi, -2000pi])
      upper_bound: the lower bound IMU values
        (default: [2pi, 2pi, 2000pi, 2000pi])
      name: the name of the sensor
      dtype: data type of sensor value
    """
    self._channels = channels if channels else ["R", "P", "dR", "dP"]
    self._num_channels = len(self._channels)
    self._noisy_reading = noisy_reading
    self._mean = np.array([0]*6)
    self._std = np.array([0.1]*3+[0.5]*3)
    self.normal = normal
    self.noise = noise
    # Compute the default lower and upper bounds
    if lower_bound is None and upper_bound is None:
      lower_bound = []
      upper_bound = []
      for channel in self._channels:
        if channel in ["R", "P", "Y"]:
          lower_bound.append(-2.0 * np.pi)
          upper_bound.append(2.0 * np.pi)
        elif channel in ["Rcos", "Rsin", "Pcos", "Psin", "Ycos", "Ysin"]:
          lower_bound.append(-1.)
          upper_bound.append(1.)
        elif channel in ["dR", "dP", "dY"]:
          lower_bound.append(-2000.0 * np.pi)
          upper_bound.append(2000.0 * np.pi)

    super(IMUSensor, self).__init__(
        name=name,
        shape=(self._num_channels,),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        dtype=dtype)

    # Compute the observation_datatype
    datatype = [("{}_{}".format(name, channel), self._dtype)
                for channel in self._channels]

    self._datatype = datatype

  def get_channels(self) -> typing.Iterable[typing.Text]:
    return self._channels

  def get_num_channels(self) -> int:
    return self._num_channels

  def get_observation_datatype(self) -> _DATATYPE_LIST:
    """Returns box-shape data type."""
    return self._datatype

  def _get_observation(self) -> _ARRAY:
    if self.first_time:
      self.first_rpy = self._robot.GetBaseRollPitchYaw()
      self.first_time = False

    if self._noisy_reading:
      rpy = self._robot.GetBaseRollPitchYaw() - self.first_rpy
      drpy = self._robot.GetBaseRollPitchYawRate()
    else:
      rpy = self._robot.GetTrueBaseRollPitchYaw() - self.first_rpy
      drpy = self._robot.GetTrueBaseRollPitchYawRate()

    if self.noise:
      # print("noise!")
      rpy += np.random.normal(0,6e-2,size=3)
      drpy += np.random.normal(0,1e-1,size=3)
    assert len(rpy) >= 3, rpy
    assert len(drpy) >= 3, drpy

    observations = np.zeros(self._num_channels)
    for i, channel in enumerate(self._channels):
      if channel == "R":
        observations[i] = rpy[0]
      if channel == "Rcos":
        observations[i] = np.cos(rpy[0])
      if channel == "Rsin":
        observations[i] = np.sin(rpy[0])
      if channel == "P":
        observations[i] = rpy[1]
      if channel == "Pcos":
        observations[i] = np.cos(rpy[1])
      if channel == "Psin":
        observations[i] = np.sin(rpy[1])
      if channel == "Y":
        observations[i] = rpy[2]
      if channel == "Ycos":
        observations[i] = np.cos(rpy[2])
      if channel == "Ysin":
        observations[i] = np.sin(rpy[2])
      if channel == "dR":
        observations[i] = drpy[0]
      if channel == "dP":
        observations[i] = drpy[1]
      if channel == "dY":
        observations[i] = drpy[2]
    if self.normal:
      return (observations-self._mean)/self._std
    else:
      return observations
  def reset(self):
    self.first_time = True

class BasePositionSensor(sensor.BoxSpaceSensor):
  """A sensor that reads the base position of the Minitaur robot."""

  def __init__(self,
               lower_bound: _FLOAT_OR_ARRAY = -100,
               upper_bound: _FLOAT_OR_ARRAY = 100,
               name: typing.Text = "BasePosition",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs BasePositionSensor.

    Args:
      lower_bound: the lower bound of the base position of the robot.
      upper_bound: the upper bound of the base position of the robot.
      name: the name of the sensor
      dtype: data type of sensor value
    """
    super(BasePositionSensor, self).__init__(
        name=name,
        shape=(3,),  # x, y, z
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        dtype=dtype)

  def _get_observation(self) -> _ARRAY:
    return self._robot.GetBasePosition()
  def reset(self):
    pass

class PoseSensor(sensor.BoxSpaceSensor):
  """A sensor that reads the (x, y, theta) of a robot."""

  def __init__(self,
               lower_bound: _FLOAT_OR_ARRAY = -100,
               upper_bound: _FLOAT_OR_ARRAY = 100,
               name: typing.Text = "PoseSensor",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs PoseSensor.

    Args:
      lower_bound: the lower bound of the pose of the robot.
      upper_bound: the upper bound of the pose of the robot.
      name: the name of the sensor.
      dtype: data type of sensor value.
    """
    super(PoseSensor, self).__init__(
        name=name,
        shape=(3,),  # x, y, orientation
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        dtype=dtype)

  def _get_observation(self) -> _ARRAY:
    return np.concatenate((self._robot.GetBasePosition()[:2],
                           (self._robot.GetTrueBaseRollPitchYaw()[2],)))
  def reset(self):
    pass
class FootForceSensor(sensor.BoxSpaceSensor):
  """A sensor that reads the contact force of a robot."""

  def __init__(self,
               lower_bound: _FLOAT_OR_ARRAY = -1000,
               upper_bound: _FLOAT_OR_ARRAY = 1000,
               name: typing.Text = "FootForceSensor",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs PoseSensor.

    Args:
      lower_bound: the lower bound of the pose of the robot.
      upper_bound: the upper bound of the pose of the robot.
      name: the name of the sensor.
      dtype: data type of sensor value.
    """
    super(FootForceSensor, self).__init__(
        name=name,
        shape=(16,),  # x, y, orientation
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        dtype=dtype)

  def _get_observation(self) -> _ARRAY:
    # print(self._robot.GetFootContactsForce())
    return self._robot.GetFootContactsForce( mode='full')
  def reset(self):
    pass
class SimpleFootForceSensor(sensor.BoxSpaceSensor):
  """A sensor that reads the contact force of a robot."""

  def __init__(self,
               lower_bound: _FLOAT_OR_ARRAY = -1000,
               upper_bound: _FLOAT_OR_ARRAY = 1000,
               name: typing.Text = "FootForceSensor",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs PoseSensor.

    Args:
      lower_bound: the lower bound of the pose of the robot.
      upper_bound: the upper bound of the pose of the robot.
      name: the name of the sensor.
      dtype: data type of sensor value.
    """
    super(SimpleFootForceSensor, self).__init__(
        name=name,
        shape=(8,),  # x, y, orientation
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        dtype=dtype)

  def _get_observation(self) -> _ARRAY:
    # print(self._robot.GetFootContactsForce())
    return self._robot.GetFootContactsForce( mode='simple')
  def reset(self):
    pass

class FootContactSensor(sensor.BoxSpaceSensor):
  """A sensor that reads the contact force of a robot."""

  def __init__(self,
               lower_bound: _FLOAT_OR_ARRAY = -1000,
               upper_bound: _FLOAT_OR_ARRAY = 1000,
               name: typing.Text = "FootContactSensor",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs PoseSensor.

    Args:
      lower_bound: the lower bound of the pose of the robot.
      upper_bound: the upper bound of the pose of the robot.
      name: the name of the sensor.
      dtype: data type of sensor value.
    """
    super(FootContactSensor, self).__init__(
        name=name,
        shape=(4,),  # x, y, orientation
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        dtype=dtype)

  def _get_observation(self) -> _ARRAY:
    # print(self._robot.GetFootContactsForce())
    return np.array(self._robot.GetFootContactsForce( mode='simple')[:4]).reshape(-1)
  def reset(self):
    pass


class FootPoseSensor(sensor.BoxSpaceSensor):
  """A sensor that reads the contact force of a robot."""

  def __init__(self,
               lower_bound: _FLOAT_OR_ARRAY = -1000,
               upper_bound: _FLOAT_OR_ARRAY = 1000,
               normal: int = 0,
               name: typing.Text = "FootPoseSensor",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    """Constructs PoseSensor.

    Args:
      lower_bound: the lower bound of the pose of the robot.
      upper_bound: the upper bound of the pose of the robot.
      name: the name of the sensor.
      dtype: data type of sensor value.
    """
    super(FootPoseSensor, self).__init__(
        name=name,
        shape=(12,),  # x, y, orientation
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        dtype=dtype)
    self.normal = normal
    self._mean = np.array([1.7454079e-01, -1.5465108e-01 ,-2.0661314e-01  ,1.7080666e-01,
                  1.6490668e-01, -2.0865265e-01 ,-1.9902834e-01 ,-1.2880404e-01,
                  -2.3593837e-01, -2.0215839e-01,  1.3673349e-01, -2.3642859e-01 ])
    self._std = np.array([3.9058894e-02,2.4757426e-02,4.2747084e-02,
                  4.1128017e-02, 2.7591322e-02, 4.3003809e-02, 4.3018311e-02, 2.8423777e-02,
                  4.7990609e-02, 4.6113804e-02, 2.8037265e-02, 4.9409315e-02])
  def _get_observation(self) -> _ARRAY:
    if self.normal:
      return (np.array(self._robot.GetFootPositionsInBaseFrame()).reshape(-1)-self._mean)/self._std
    else:
      return np.array(self._robot.GetFootPositionsInBaseFrame()).reshape(-1)
  
  def reset(self):
    pass


class DepthSensor(sensor.BoxSpaceSensor):
  """By GXZ."""

  def __init__(self,
               lower_bound: _FLOAT_OR_ARRAY = 0.0,
               upper_bound: _FLOAT_OR_ARRAY = 1.0,
               name: typing.Text = "DepthSensor",
               dtype: typing.Type[typing.Any] = np.float32) -> None:
    
    """Constructs DepthSensor.

    Args:
      lower_bound: the lower bound of the depth of the robot.
      upper_bound: the upper bound of the depth of the robot.
      name: the name of the sensor.
      dtype: data type of sensor value.
    """
    super(DepthSensor, self).__init__(
        name=name,
        shape=(64, 64),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        dtype=dtype)
  
  def _get_observation(self) -> _ARRAY:
    # base_pos = self._robot.GetBasePosition()
    # rpy = self._robot.GetBaseRollPitchYaw()
# 
    # view_matrix = self._robot._pybullet_client.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=base_pos,
    #     distance=0.5,
    #     yaw=rpy[2]*180/np.pi,
    #     pitch=rpy[0]*180/np.pi,
    #     roll=rpy[1]*180/np.pi,
    #     upAxisIndex=2)
    # proj_matrix = self._robot._pybullet_client.computeProjectionMatrixFOV(
    #     fov=120,
    #     aspect=1,
    #     nearVal=0.1,
    #     farVal=100.0)
    # (_, _, img, depth, _) = self._robot._pybullet_client.getCameraImage(
    #     width=640,
    #     height=640,
    #     renderer=self._robot._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
    #     viewMatrix=view_matrix,
    #     projectionMatrix=proj_matrix)

    base_pos, rpy = self._robot._pybullet_client.getBasePositionAndOrientation(self._robot.quadruped)
    matrix = self._robot._pybullet_client.getMatrixFromQuaternion(rpy)
    tx_vec = np.array([matrix[0], matrix[3], matrix[6]])
    ty_vec = np.array([matrix[1], matrix[4], matrix[7]])
    tz_vec = np.array([matrix[2], matrix[5], matrix[8]])

    base_pos = np.array(base_pos)

    FAR = 100
    NEAR = 0.1

    CAMERA_X = 0.26
    CAMERA_Y = 0
    CAMERA_Z = 0
    CAMERA_PITCH = 0

    camera_pos = base_pos + CAMERA_X * tx_vec + CAMERA_Y * ty_vec + CAMERA_Z * tz_vec
    target_pos = camera_pos + np.cos(CAMERA_PITCH) * tx_vec + np.sin(CAMERA_PITCH) * tz_vec

    # x_vec = np.array([1, 0, 0])
    # y_vec = np.array([0, 1, 0])
    # z_vec = np.array([0, 0, 1])
    # camera_pos = base_pos + CAMERA_X * x_vec + CAMERA_Y * y_vec + CAMERA_Z * z_vec
    # target_pos = camera_pos + np.cos(CAMERA_PITCH) * x_vec + np.sin(CAMERA_PITCH) * z_vec

    viewMatrix = self._robot._pybullet_client.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=tz_vec  # z_vec or tz_vec
    )
    projectionMatrix = self._robot._pybullet_client.computeProjectionMatrixFOV(
        fov=50.0,
        aspect=1.0,
        nearVal=NEAR,
        farVal=FAR,
    )
    _, _, img, depth, _ = self._robot._pybullet_client.getCameraImage(
        width=64, 
        height=64,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer=self._robot._pybullet_client.ER_BULLET_HARDWARE_OPENGL
    )

    depth = FAR * NEAR / (FAR - (FAR - NEAR) * depth)
    alpha = 1.5
    depth = (FAR - NEAR + alpha) / (FAR - NEAR) * (1 - alpha / (depth - NEAR + alpha))

    # import cv2
    # cv2.imwrite('temp/img_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f.png' % (base_pos[0], base_pos[1], base_pos[2], rpy[0], rpy[1], rpy[2]), img)
    # cv2.imwrite('temp/img_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f.png' % (base_pos[0], base_pos[1], base_pos[2], rpy[0], rpy[1], rpy[2]), 255.0 * depth)
    # depth = np.array(depth[::10, ::10])
    return depth
  
  def reset(self):
    pass