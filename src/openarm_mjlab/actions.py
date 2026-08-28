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

"""Custom action terms shared by the reach/valve/door/drawer/puck/lift task family.

All tasks in the family actively control only one arm; the parked arm needs
an explicit hold action so it is not pulled toward qpos 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class HoldDefaultPositionAction(JointPositionAction):
    """Zero-dim action term that servo-holds targets at their default pose.

    Every task in this family actively controls only one arm; the parked
    arm needs an explicit hold, since an actuated-but-uncontrolled joint is
    otherwise pulled toward qpos 0 on every reset rather than its default
    pose. Matched by ``actuator_names``, commanded to ``default_joint_pos``
    every step, contributing zero dimensions to the policy's action space.
    """

    @property
    def action_dim(self) -> int:
        """Return zero: this term takes no policy input."""
        return 0

    def process_actions(self, actions: torch.Tensor) -> None:
        """Do nothing; the hold target is constant, not policy-driven."""
        del actions  # Nothing to process; targets are constant.

    def apply_actions(self) -> None:
        """Command the held joints back to their default position."""
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(
            self._offset - encoder_bias, joint_ids=self._target_ids
        )


@dataclass(kw_only=True)
class HoldDefaultPositionActionCfg(JointPositionActionCfg):
    """Config for :class:`HoldDefaultPositionAction`."""

    def build(self, env: ManagerBasedRlEnv) -> HoldDefaultPositionAction:
        """Build the :class:`HoldDefaultPositionAction` term."""
        return HoldDefaultPositionAction(self, env)
