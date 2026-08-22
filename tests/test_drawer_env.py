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

"""Integration tests for the OpenArm-Drawer environment.

Note: building the env compiles mujoco-warp CPU kernels; the first run can
take a few minutes.
"""

import pytest
import torch

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.tasks.registry import list_tasks, load_env_cfg

# joint_pos/joint_vel report ALL 18 bimanual joints (both arms), regardless
# of which arm this task actually actuates.
OBS_DIM = 18 + 18 + 1 + 3 + 1 + 8  # joint_pos, joint_vel, drawer_pos,
# ee_to_handle, handle_contact, actions


def test_task_is_registered():
    assert "OpenArm-Drawer" in list_tasks()


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg("OpenArm-Drawer")
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_action_and_observation_dims(env):
    assert env.action_manager.total_action_dim == 8
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, OBS_DIM)
    assert obs["critic"].shape == (2, OBS_DIM)


def test_env_steps_with_finite_signals(env):
    env.reset()
    for _ in range(10):
        action = torch.zeros(2, env.action_manager.total_action_dim)
        obs, rew, terminated, truncated, _ = env.step(action)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(rew).all()
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)


def test_reset_along_pull_spawns_within_max_opening(env):
    """`reset_along_pull` must spawn every env within its declared max opening."""
    from openarm_mjlab.tasks.drawer.drawer_env_cfg import CABINET_JOINT_CFG
    from openarm_mjlab.tasks.drawer.mdp import drawer_opening

    env.reset()
    opening = drawer_opening(env, CABINET_JOINT_CFG)
    assert (opening >= -1e-6).all() and (opening <= 0.08 + 1e-6).all()


def test_drawer_start_opening_recorded_on_reset(env):
    """`record_drawer_start` must snapshot each env's opening at its own reset."""
    from openarm_mjlab.tasks.drawer.drawer_env_cfg import CABINET_JOINT_CFG
    from openarm_mjlab.tasks.drawer.mdp import _start_opening, drawer_opening

    env.reset()
    opening = drawer_opening(env, CABINET_JOINT_CFG)
    assert torch.allclose(_start_opening(env), opening)


def test_left_arm_stays_at_default_under_zero_action(env):
    """The parked left arm must hold its default pose, not drift to qpos 0."""
    env.reset()
    action = torch.zeros(2, env.action_manager.total_action_dim)
    for _ in range(20):
        env.step(action)
    robot = env.scene["robot"]
    left_j4 = robot.find_joints("openarm_left_joint4")[0][0]
    # Home pose has joint4 at 1.5708 rad; qpos 0 would mean the hold failed.
    assert torch.allclose(
        robot.data.joint_pos[:, left_j4], torch.tensor(1.5708), atol=0.05
    )
