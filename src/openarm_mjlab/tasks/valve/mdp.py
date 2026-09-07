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

"""Valve-turning MDP terms.

Contact-gated RATE reward (state rewards pay retroactively), gained progress
vs. episode start (a randomized start would otherwise pay free income),
anti-reverse penalty (no pumping), and a terminal success bonus (a success
termination without one makes the optimum avoid success).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...common_mdp import (
    contact_reward,
    ee_to_target,
    fingers_on_handle,
    reach_target_reward,
    terminated_by,
)


if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "contact_reward",
    "ee_to_target",
    "fingers_on_handle",
    "reach_target_reward",
    "terminated_by",
]

TARGET_TURN = 1.35  # rad (77.4 deg): the single-grasp kinematic ceiling.
MAX_TURN_RATE = 1.0  # rad/s


def valve_angle(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the valve hinge angle, radians."""
    valve: Entity = env.scene[asset_cfg.name]
    return valve.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)


def valve_rate(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the valve hinge angular velocity, rad/s."""
    valve: Entity = env.scene[asset_cfg.name]
    return valve.data.joint_vel[:, asset_cfg.joint_ids].squeeze(-1)


def _start_angle(env) -> torch.Tensor:
    """Return the per-env angle recorded at episode start."""
    if not hasattr(env, "_valve_start_angle"):
        env._valve_start_angle = torch.zeros(env.num_envs, device=env.device)
    return env._valve_start_angle


def _gained_contact(env) -> torch.Tensor:
    """Return rotation accumulated only while fingers touch the grip."""
    if not hasattr(env, "_valve_gained_contact"):
        env._valve_gained_contact = torch.zeros(env.num_envs, device=env.device)
    return env._valve_gained_contact


def _max_angle(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the per-env running-max angle reached this episode."""
    if not hasattr(env, "_valve_max_angle"):
        env._valve_max_angle = valve_angle(env, asset_cfg).clone()
    return env._valve_max_angle


def _prev_progress(env) -> torch.Tensor:
    """Return the previous step's clamped progress fraction, for shaping."""
    if not hasattr(env, "_valve_prev_progress"):
        env._valve_prev_progress = torch.zeros(env.num_envs, device=env.device)
    return env._valve_prev_progress


def record_valve_start(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset all per-episode valve bookkeeping for the given envs."""
    a = valve_angle(env, asset_cfg)
    _start_angle(env)[env_ids] = a[env_ids]
    _gained_contact(env)[env_ids] = 0.0
    _max_angle(env, asset_cfg)[env_ids] = a[env_ids]
    _prev_progress(env)[env_ids] = 0.0


def turn_gained(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return rotation gained past the episode start (positive direction), rad."""
    return torch.clamp(valve_angle(env, asset_cfg) - _start_angle(env), min=0.0)


def turn_rate_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a contact-gated reward for new angle above the running max.

    Also accumulates gained-under-contact for the honesty check in
    :func:`turned_target`. Credited progress is capped at ``TARGET_TURN`` so
    continuing to turn past the target under contact is never separately
    incentivized once the target is already met.
    """
    contact = fingers_on_handle(env, sensor_name).float()
    a = valve_angle(env, asset_cfg)
    maxa = _max_angle(env, asset_cfg)
    a_capped = torch.clamp(a, max=TARGET_TURN)
    maxa_capped = torch.clamp(maxa, max=TARGET_TURN)
    new = torch.clamp(a_capped - maxa_capped, min=0.0)
    new_uncapped = torch.clamp(a - maxa, min=0.0)
    _gained_contact(env).add_(new_uncapped * contact)
    maxa.copy_(torch.maximum(maxa, a))
    capped = torch.clamp(new / env.step_dt, 0.0, MAX_TURN_RATE) / MAX_TURN_RATE
    return capped * contact


def turn_progress_shaping_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Return potential-based shaping (Ng, Harada & Russell 1999) on gained-turn fraction.

    ``reward = gamma * Phi(s') - Phi(s)``. Camping-negative by construction: a
    fixed state pays ``(gamma - 1) * Phi < 0`` every step instead of paying
    nothing-or-positive, so holding still under contact is never free.
    """
    gate = fingers_on_handle(env, sensor_name).float()
    cur = torch.clamp(turn_gained(env, asset_cfg) / TARGET_TURN, 0.0, 1.0)
    prev = _prev_progress(env)
    shaping = gamma * cur - prev
    prev.copy_(cur)
    return shaping * gate


def reverse_rate_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an anti-pump penalty: backing the valve up is never free."""
    return torch.clamp(-valve_rate(env, asset_cfg), min=0.0)


def overspeed_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a penalty for turning faster than ``MAX_TURN_RATE``."""
    return torch.clamp(valve_rate(env, asset_cfg).abs() - MAX_TURN_RATE, min=0.0)


def turned_target(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return success: target rotation gained, quasi-static, still in contact.

    Requires at least 85% of the gained turn to have been produced under
    contact, so a policy that flings the valve open without a real grip does
    not count as success.
    """
    done = turn_gained(env, asset_cfg) >= TARGET_TURN
    honest = _gained_contact(env) >= 0.85 * TARGET_TURN
    slow = valve_rate(env, asset_cfg).abs() < MAX_TURN_RATE
    return done & honest & slow & fingers_on_handle(env, sensor_name)
