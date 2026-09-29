# Copyright 2026 Enactic, Inc.
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

"""OpenArm move-puck task, privileged and vision variants."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .puck_env_cfg import openarm_puck_env_cfg
from .rl_cfg import openarm_puck_ppo_runner_cfg, openarm_puck_vision_ppo_runner_cfg

register_mjlab_task(
    task_id="OpenArm-Puck",
    env_cfg=openarm_puck_env_cfg(),
    play_env_cfg=openarm_puck_env_cfg(play=True),
    rl_cfg=openarm_puck_ppo_runner_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)

# Same task/rewards/terminations/events -- only the actor's observations
# change (privileged puck-position terms swapped for a fixed overhead
# depth camera). Separate task_id so the privileged checkpoint is never
# touched.
register_mjlab_task(
    task_id="OpenArm-Puck-Vision",
    env_cfg=openarm_puck_env_cfg(vision=True),
    play_env_cfg=openarm_puck_env_cfg(play=True, vision=True),
    rl_cfg=openarm_puck_vision_ppo_runner_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)
