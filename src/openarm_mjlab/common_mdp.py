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

"""MDP helper functions shared across the valve/door/drawer/puck/lift tasks.

All of them gate progress on genuine finger-pad contact rather than a proxy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from .robot_bimanual import GRASP_LOCAL_OFFSET

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def fingers_on_handle(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return True where a finger pad touches the handle/grip.

    Any-pad contact, not a two-pad pinch: this gripper's closed cage leaves
    a real gap for slim handles, so even a scripted approach caging the
    handle presses only one pad against it. Progress rewards are gated on
    this (sustained contact required, a flick loses contact and forfeits
    every later step) rather than on the pinch itself.
    """
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    return (found.view(env.num_envs, -1) > 0).any(dim=1)


def fingers_on_handle_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Observation wrapper for :func:`fingers_on_handle`."""
    return fingers_on_handle(env, sensor_name).float().unsqueeze(-1)


def both_pads_on_block(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return True where BOTH finger pads touch the object.

    A genuine squeeze, not a poke. Used by tasks whose object is wide enough
    for a real friction pinch (unlike a slim handle bar).
    """
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    return (found.view(env.num_envs, 2, -1).amax(dim=-1) > 0).all(dim=1)


def pinch_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Observation wrapper for :func:`both_pads_on_block`."""
    return both_pads_on_block(env, sensor_name).float().unsqueeze(-1)


def ee_to_target(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the vector from the finger-cage center to a target site, base frame.

    ``robot_cfg`` selects the end-effector site and ``target_cfg`` the site to
    reach for -- a drawer handle, a door handle, a valve grip. The cage center
    is the EE site displaced by ``GRASP_LOCAL_OFFSET``, so the vector is zero
    when the target sits in the middle of the open fingers rather than at the
    wrist. Expressed in the robot base frame so the observation does not move
    with the world origin.
    """
    robot: Entity = env.scene[robot_cfg.name]
    target: Entity = env.scene[target_cfg.name]
    ee_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    ee_quat_w = robot.data.site_quat_w[:, robot_cfg.site_ids].squeeze(1)
    offset = torch.tensor(GRASP_LOCAL_OFFSET, device=ee_pos_w.device).expand_as(
        ee_pos_w
    )
    tool_pos_w = ee_pos_w + quat_apply(ee_quat_w, offset)
    target_pos_w = target.data.site_pos_w[:, target_cfg.site_ids].squeeze(1)
    vec_w = target_pos_w - tool_pos_w
    base_quat_w = robot.data.root_link_quat_w
    return quat_apply(quat_inv(base_quat_w), vec_w)


def reach_target_reward(
    env: ManagerBasedRlEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a Gaussian-kernel reward on the tool-to-target distance."""
    d2 = torch.sum(torch.square(ee_to_target(env, robot_cfg, target_cfg)), dim=-1)
    return torch.exp(-d2 / std**2)


def contact_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return a dense reward for holding the handle/grip in contact."""
    return fingers_on_handle(env, sensor_name).float()


def terminated_by(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
    """Return 1.0 on the step the named termination term fires.

    Terminations are computed before rewards each step, so the manager's
    cached result is current; referencing it keeps the success bonus and the
    success condition structurally identical, instead of restating the
    predicate in two places that can drift apart.
    """
    return env.termination_manager.get_term(term_name).float()
