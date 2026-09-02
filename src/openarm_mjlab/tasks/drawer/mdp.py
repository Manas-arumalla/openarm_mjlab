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

"""Observation, reward, and termination terms for the drawer-pulling task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from ...common_mdp import (
    contact_reward,
    ee_to_target,
    fingers_on_handle,
    fingers_on_handle_obs,
    reach_target_reward,
    terminated_by,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "contact_reward",
    "ee_to_target",
    "fingers_on_handle",
    "fingers_on_handle_obs",
    "reach_target_reward",
    "terminated_by",
]

# The cabinet's slide joint runs -0.10 (open) .. 0 (closed); opening in
# meters is -qpos.
DRAWER_TRAVEL = 0.10


def drawer_opening(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the drawer opening in meters, shape ``(num_envs,)``."""
    cabinet: Entity = env.scene[asset_cfg.name]
    return -cabinet.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)


def drawer_speed(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the absolute slide velocity in m/s, shape ``(num_envs,)``."""
    cabinet: Entity = env.scene[asset_cfg.name]
    return cabinet.data.joint_vel[:, asset_cfg.joint_ids].squeeze(-1).abs()


def _engaged(env) -> torch.Tensor:
    """Return whether this episode's engagement depth has been frozen yet."""
    if not hasattr(env, "_drawer_engaged"):
        env._drawer_engaged = torch.zeros(env.num_envs, device=env.device)
    return env._drawer_engaged


def _engage_frac(env) -> torch.Tensor:
    """Return the opening fraction frozen at first genuine handle contact.

    This task uses a "caged start" (fingers already on the handle at
    spawn), so a plain boolean engaged/not-engaged latch would fire on
    the very first step of nearly every episode. Freezing the DEPTH at
    first contact instead preserves a real, spawn-depth-appropriate
    reward value while still removing the growing per-step opportunity
    cost of pulling deep early vs. late in an episode.
    """
    if not hasattr(env, "_drawer_engage_frac"):
        env._drawer_engage_frac = torch.zeros(env.num_envs, device=env.device)
    return env._drawer_engage_frac


def handle_contact_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a dense reward for touching the handle; also freezes engagement depth."""
    contact = contact_reward(env, sensor_name)
    newly = (contact > 0) & (_engaged(env) == 0)
    if newly.any():
        frac = torch.clamp(drawer_opening(env, asset_cfg) / DRAWER_TRAVEL, 0.0, 1.0)
        _engage_frac(env)[newly] = frac[newly]
        _engaged(env)[newly] = 1.0
    return contact


def _start_opening(env) -> torch.Tensor:
    """Return the per-env drawer opening recorded at episode start."""
    if not hasattr(env, "_drawer_start_opening"):
        env._drawer_start_opening = torch.zeros(env.num_envs, device=env.device)
    return env._drawer_start_opening


def _max_opening(env) -> torch.Tensor:
    """Return the per-env running-max opening reached this episode."""
    if not hasattr(env, "_drawer_max_opening"):
        env._drawer_max_opening = torch.zeros(env.num_envs, device=env.device)
    return env._drawer_max_opening


def _prev_progress(env) -> torch.Tensor:
    """Return the previous step's clamped progress fraction, for shaping."""
    if not hasattr(env, "_drawer_prev_progress"):
        env._drawer_prev_progress = torch.zeros(env.num_envs, device=env.device)
    return env._drawer_prev_progress


def record_drawer_start(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset all per-episode drawer bookkeeping for the given envs.

    Recording the start opening ensures progress rewards pay only for
    opening the policy PRODUCED: with a randomized initial opening,
    absolute-opening rewards would otherwise pay free income at spawn.
    """
    cabinet: Entity = env.scene[asset_cfg.name]
    op = -cabinet.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    _start_opening(env)[env_ids] = op[env_ids]
    _max_opening(env)[env_ids] = op[env_ids]
    _prev_progress(env)[env_ids] = 0.0
    if not hasattr(env, "_drawer_gained_contact"):
        env._drawer_gained_contact = torch.zeros(env.num_envs, device=env.device)
    env._drawer_gained_contact[env_ids] = 0.0
    _engaged(env)[env_ids] = 0.0
    _engage_frac(env)[env_ids] = 0.0


def open_gained_progress(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return opening gained beyond the episode's start, 0..1 over full travel."""
    gained = drawer_opening(env, asset_cfg) - _start_opening(env)
    return torch.clamp(gained / DRAWER_TRAVEL, 0.0, 1.0)


def open_progress_shaping_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Return potential-based shaping (Ng, Harada & Russell 1999) on gained-opening fraction.

    ``reward = gamma * Phi(s') - Phi(s)``. A reward that pays the same
    every step for a fixed state makes camping strictly profitable, but
    removing the dense per-depth signal outright regresses value-function
    credit assignment. Potential-based shaping keeps the density while
    being camping-negative by construction: if the state doesn't change
    between steps, the term is exactly ``(gamma - 1) * Phi(s) < 0``, so
    standing still always costs a little, everywhere along the pull.
    Contact-gated like every other progress term here (anti-flick).
    """
    gate = fingers_on_handle(env, sensor_name).float()
    cur = open_gained_progress(env, asset_cfg)
    prev = _prev_progress(env)
    shaping = gamma * cur - prev
    prev.copy_(cur)
    return shaping * gate


def closing_speed_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return an anti-pump penalty: closing the drawer is never free."""
    cabinet: Entity = env.scene[asset_cfg.name]
    closing_rate = cabinet.data.joint_vel[:, asset_cfg.joint_ids].squeeze(-1)
    return torch.clamp(closing_rate, min=0.0)


def drawer_speed_penalty(
    env: ManagerBasedRlEnv,
    max_speed: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a penalty for slide speed beyond a grasp-pull-plausible rate."""
    return torch.clamp(drawer_speed(env, asset_cfg) - max_speed, min=0.0)


def pull_rate_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_speed: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a contact-gated pull rate, capped at the speed limit (0..1).

    This is the anti-flick term: opening gained without finger contact or
    above the cap earns nothing per-mm, so knocking the drawer open is
    strictly worse than pulling it at or below the cap while touching.
    New-progress-only (a plain rate reward pays for oscillation).
    """
    h = drawer_opening(env, asset_cfg)
    maxh = _max_opening(env)
    new = torch.clamp(h - maxh, min=0.0)
    maxh.copy_(torch.maximum(maxh, h))
    capped = torch.clamp(new / env.step_dt, 0.0, max_speed) / max_speed
    contact = fingers_on_handle(env, sensor_name).float()
    if not hasattr(env, "_drawer_gained_contact"):
        env._drawer_gained_contact = torch.zeros(env.num_envs, device=env.device)
    env._drawer_gained_contact.add_(new * contact)
    return capped * contact


def approach_precision_reward(
    env: ManagerBasedRlEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    cabinet_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a tight cage-on-bar kernel, active during the approach phase only.

    An always-on tight kernel taxes deep pulls (cage alignment degrades
    near full arm extension), so precision is rewarded only at grasp
    ACQUISITION; once the drawer is moving under contact, the pull terms
    own the behavior. Faded by the depth FROZEN at first genuine handle
    contact (:func:`_engage_frac`), not by continuous absolute progress,
    so the fade never grows with how much further the policy has pulled
    since first contact.
    """
    fine = reach_target_reward(env, std, robot_cfg, cabinet_cfg)
    return fine * (1.0 - _engage_frac(env))


def frontal_grasp_reward(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg,
    pitch: float = 0.2618,
    fade: float = 0.0,
) -> torch.Tensor:
    """Return a reward for a human-like frontal wrist orientation.

    The tool axis (site -z) points world +x pitched down, and the closing
    axis stays in the vertical plane so the cage straddles the horizontal
    handle bar top/bottom, the way a person pulls a drawer. Without this
    term the policy grabs the bar sideways. Returns the product of both
    axis alignments, each mapped to 0..1.
    """
    robot: Entity = env.scene[robot_cfg.name]
    quat = robot.data.site_quat_w[:, robot_cfg.site_ids].squeeze(1)
    device = quat.device
    z_world = quat_apply(
        quat, torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(quat[:, :3])
    )
    x_world = quat_apply(
        quat, torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(quat[:, :3])
    )
    target_z = torch.tensor([-math.cos(pitch), 0.0, math.sin(pitch)], device=device)
    target_x = torch.tensor([0.0, -1.0, 0.0], device=device)
    tool_align = (z_world @ target_z + 1.0) / 2.0
    closing_align = (x_world @ target_x + 1.0) / 2.0
    r = tool_align * closing_align
    if fade > 0.0:
        # Human wrists orient frontally to GRAB, then rotate a little as
        # the pull comes toward the body; holding full frontal through a
        # deep pull costs workspace. Faded by the depth frozen at first
        # contact, so this is a fixed, spawn-depth-appropriate discount,
        # not a growing per-step cost that punishes pulling deep early.
        r = r * (1.0 - fade * _engage_frac(env))
    return r


def drawer_held_fully_open(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
    max_speed: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return success: fully open, quasi-static, in contact, and honestly earned.

    Requires at least 85% of the opening produced to have happened under
    contact, so knocking the drawer open does not count as success.
    """
    opening_ok = drawer_opening(env, asset_cfg) > threshold
    slow = drawer_speed(env, asset_cfg) < max_speed
    contact = fingers_on_handle(env, sensor_name)
    gained = drawer_opening(env, asset_cfg) - _start_opening(env)
    if not hasattr(env, "_drawer_gained_contact"):
        env._drawer_gained_contact = torch.zeros(env.num_envs, device=env.device)
    honest = env._drawer_gained_contact >= 0.85 * torch.clamp(gained, min=1e-6)
    return opening_ok & slow & contact & honest


# Pull-manifold spawns (last-resort drawer curriculum): both branch-
# consistent IK poses, lerped by the spawn opening so the cage tracks
# the handle at every depth.
PULL_POSE_CLOSED = (0.3023, 0.171, 0.1442, 1.3204, 0.0122, 0.0955, -0.0358)
PULL_POSE_OPEN = (-0.1476, 0.1145, 0.2265, 1.7283, 0.002, -0.0623, -0.0892)
_RIGHT_JOINTS = tuple(f"openarm_right_joint{i}" for i in range(1, 8))


def reset_along_pull(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    robot_joints_cfg: SceneEntityCfg,
    max_opening: float = 0.08,
) -> None:
    """Spawn at a random opening with the cage already tracking the handle.

    Uses linear pose interpolation between the two IK anchors. Random
    exploration cannot sustain a 90mm straight-line pull, so the policy
    must see the whole pull manifold from step 0.
    """
    robot: Entity = env.scene[robot_joints_cfg.name]
    cabinet: Entity = env.scene[asset_cfg.name]
    n = len(env_ids)
    o = torch.rand(n, device=env.device) * max_opening
    jp_c = cabinet.data.joint_pos[env_ids].clone()
    jv_c = torch.zeros_like(cabinet.data.joint_vel[env_ids])
    jp_c[:, asset_cfg.joint_ids] = (-o).unsqueeze(-1)
    cabinet.write_joint_state_to_sim(jp_c, jv_c, env_ids=env_ids)
    frac = (o / DRAWER_TRAVEL).unsqueeze(-1)
    closed = torch.tensor(PULL_POSE_CLOSED, device=env.device)
    opened = torch.tensor(PULL_POSE_OPEN, device=env.device)
    pose = closed * (1 - frac) + opened * frac
    jp = robot.data.joint_pos[env_ids].clone()
    jv = torch.zeros_like(robot.data.joint_vel[env_ids])
    names = robot.joint_names
    for k, jname in enumerate(_RIGHT_JOINTS):
        j = names.index(jname)
        jp[:, j] = pose[:, k]
    for j, jname in enumerate(names):
        if "right_finger" in jname:
            jp[:, j] = -0.25
    robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
    # JointPositionAction's use_default_offset caches offset =
    # default_joint_pos ONCE at build time and never re-reads it on
    # reset. Teleporting the arm here without also re-anchoring that
    # offset would leave a zero-mean action still demanding a return to
    # the static default pose. Re-anchoring the action term's per-env
    # offset to the actual spawn pose is what makes a zero action HOLD
    # position, the same guarantee reset_joints_by_offset gets for free
    # by using small in-place deltas instead of a teleport.
    joint_pos_term = env.action_manager.get_term("joint_pos")
    col_of = {name: i for i, name in enumerate(joint_pos_term.target_names)}
    for k, jname in enumerate(_RIGHT_JOINTS):
        joint_pos_term._offset[env_ids, col_of[jname]] = pose[:, k]
