#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rlschool.liftsim import *
from rlschool.quadrotor import *
from rlschool.quadrupedal import *

def make_env(env_id, **kwargs):
    if env_id == 'LiftSim':
        return LiftSim(**kwargs)
    elif env_id == 'Quadrotor':
        return Quadrotor(**kwargs)
    elif env_id == 'Quadrupedal':
        return A1GymEnv(**kwargs)
    else:
        raise ValueError('Unknown environment ID.')
