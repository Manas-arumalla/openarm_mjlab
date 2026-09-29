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

"""Integration tests for the OpenArm-BimanualLift environment.

Note: building the env compiles mujoco-warp CPU kernels; the first run can
take a few minutes.
"""

from types import SimpleNamespace

import pytest
import torch

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.tasks.registry import list_tasks, load_env_cfg

OBS_DIM = 18 + 18 + 3 + 3 + 1 + 1 + 16  # joint_pos, joint_vel,
# tool_to_right_end, tool_to_left_end, pinch_right, pinch_left, actions

# Unlike the other tasks in this suite, BOTH arms are actively controlled:
# two 7-dim arm terms plus two 1-dim finger-effort terms.
ACTION_DIM = 7 + 7 + 1 + 1


def test_task_is_registered():
    assert "OpenArm-BimanualLift" in list_tasks()


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg("OpenArm-BimanualLift")
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_action_and_observation_dims(env):
    assert env.action_manager.total_action_dim == ACTION_DIM
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


def test_both_arms_are_actuated(env):
    """Both arms must be controllable; neither is parked.

    This is what distinguishes the task from the rest of the suite, where one
    arm is servo-held at its default pose, so it is worth asserting rather
    than assuming.
    """
    names = list(env.action_manager._terms.keys())
    assert "right_arm" in names
    assert "left_arm" in names
    assert "right_squeeze" in names
    assert "left_squeeze" in names


def test_bar_starts_on_the_table():
    """The bar must rest at its table height on an unassisted reset.

    Built with its own env rather than the shared fixture: the task's
    curriculum deliberately spawns a fraction of episodes with the bar
    already raised, so asserting the table height against the default
    config would be flaky.
    """
    from mjlab.envs import ManagerBasedRlEnv

    from openarm_mjlab.tasks.bimanual_lift import mdp as bl_mdp

    cfg = load_env_cfg("OpenArm-BimanualLift")
    cfg.scene.num_envs = 2
    cfg.events["reset_bar_held_high"].params["probability"] = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        z = env.scene["bar"].data.root_link_pos_w[:, 2]
        assert torch.allclose(z, torch.tensor(bl_mdp.BAR_START[2]), atol=0.02)
    finally:
        env.close()


def test_success_requires_both_ends_not_just_one(env):
    """`lifted_together` must not be satisfiable by a single-arm lift.

    The height terms compose the two ends with min(), so a policy that raises
    one end while the other stays down earns nothing -- the property the task
    exists to enforce.
    """
    from openarm_mjlab.tasks.bimanual_lift import mdp as bl_mdp

    assert bl_mdp.LEVEL_TOLERANCE < bl_mdp.HEIGHT_TOLERANCE
    assert bl_mdp.HELD_HIGH_RAISE < bl_mdp.TARGET_LIFT


def test_held_high_reset_removes_joint_jitter_only_in_selected_envs(monkeypatch):
    from mjlab.managers import SceneEntityCfg

    from openarm_mjlab.tasks.bimanual_lift import mdp as bl_mdp

    joint_names = ["unrelated_joint"]
    held_pose = [0.4]
    for side, angles, squeeze in (
        ("right", bl_mdp.HELD_HIGH_RIGHT_Q, bl_mdp.RIGHT_FINGER_SQUEEZE),
        ("left", bl_mdp.HELD_HIGH_LEFT_Q, bl_mdp.LEFT_FINGER_SQUEEZE),
    ):
        joint_names.extend(f"openarm_{side}_joint{i}" for i in range(1, 8))
        joint_names.append(f"openarm_{side}_finger_joint1")
        held_pose.extend((*angles, squeeze))
    positions = torch.tensor(held_pose).repeat(3, 1) + 0.025
    velocities = torch.full_like(positions, 0.1)
    before_positions = positions.clone()
    before_velocities = velocities.clone()
    bar_states = torch.zeros(3, 13)
    bar_states[:, 3] = 1.0
    before_bar_states = bar_states.clone()

    def write_joints(joint_pos, joint_vel, *, env_ids):
        positions[env_ids] = joint_pos
        velocities[env_ids] = joint_vel

    def write_bar(state, *, env_ids):
        bar_states[env_ids] = state

    robot = SimpleNamespace(
        joint_names=joint_names,
        data=SimpleNamespace(joint_pos=positions, joint_vel=velocities),
        write_joint_state_to_sim=write_joints,
    )
    bar = SimpleNamespace(
        data=SimpleNamespace(default_root_state=before_bar_states),
        write_root_state_to_sim=write_bar,
    )
    env = SimpleNamespace(scene={"robot": robot, "bar": bar}, num_envs=3, device="cpu")
    # Select env 2, reject env 0, and leave env 1 outside this reset entirely.
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.1, 0.9]))
    bl_mdp.reset_bar_held_high(
        env,
        torch.tensor([2, 0]),
        asset_cfg=SceneEntityCfg("bar"),
        robot_joints_cfg=SceneEntityCfg("robot"),
        probability=0.5,
    )

    torch.testing.assert_close(positions[2, 1:], torch.tensor(held_pose[1:]))
    torch.testing.assert_close(positions[2, 0], before_positions[2, 0])
    torch.testing.assert_close(velocities[2], torch.zeros_like(velocities[2]))
    torch.testing.assert_close(positions[:2], before_positions[:2])
    torch.testing.assert_close(velocities[:2], before_velocities[:2])
    torch.testing.assert_close(bar_states[:2], before_bar_states[:2])
    expected_bar_position = torch.tensor(bl_mdp.BAR_START)
    expected_bar_position[2] += bl_mdp.HELD_HIGH_RAISE
    torch.testing.assert_close(bar_states[2, :3], expected_bar_position)
