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

"""Integration tests for the OpenArm-Door environment.

Note: building the env compiles mujoco-warp CPU kernels; the first run can
take a few minutes.
"""

import pytest
import torch

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.tasks.registry import list_tasks, load_env_cfg

# joint_pos/joint_vel report ALL 18 bimanual joints (both arms), regardless
# of which arm this task actually actuates.
OBS_DIM = 18 + 18 + 1 + 3 + 1 + 8  # joint_pos, joint_vel, door_angle,
# ee_to_handle, handle_contact, actions


def test_task_is_registered():
    assert "OpenArm-Door" in list_tasks()


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg("OpenArm-Door")
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


def test_episode_starts_caged_at_the_handle(env):
    """Reset must land near the scripted caged-grasp pose, not the arm's own default.

    `reset_robot_joints` adds a uniform (-0.05, 0.05) offset on top of the
    home pose, so the tolerance must cover that jitter, not just solver
    noise.
    """
    from openarm_mjlab.tasks.door.door_env_cfg import CAGED_HOME

    env.reset()
    robot = env.scene["robot"]
    j1 = robot.find_joints("openarm_right_joint1")[0][0]
    assert torch.allclose(
        robot.data.joint_pos[:, j1],
        torch.tensor(CAGED_HOME.joint_pos["openarm_right_joint1"]),
        atol=0.06,
    )


def test_door_start_angle_recorded_on_reset(env):
    """`record_door_start` must snapshot each env's angle at its own reset."""
    from openarm_mjlab.tasks.door.door_env_cfg import DOOR_JOINT_CFG
    from openarm_mjlab.tasks.door.mdp import _start_angle, door_angle

    env.reset()
    angle = door_angle(env, DOOR_JOINT_CFG)
    assert torch.allclose(_start_angle(env), angle)


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
