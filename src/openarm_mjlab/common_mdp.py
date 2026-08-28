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
