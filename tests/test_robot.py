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

"""Tests for the OpenArm Cell robot entity config."""

import mujoco
import numpy as np

from openarm_mjlab.robot import (
    LEFT_GRASP_SITE,
    LEFT_JOINT4_HOME,
    OPENARM_CELL_XML,
    get_openarm_cell_spec,
    get_openarm_robot_cfg,
)

EXPECTED_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)] + [
    "openarm_left_finger_joint1",
    "openarm_left_finger_joint2",
]
EXPECTED_ACTUATORS = [f"left_joint{i}_ctrl" for i in range(1, 8)] + [
    "left_finger1_ctrl"
]


def test_grasp_site_at_fingertip_centroid():
    """The grasp site must sit at the centroid of the fingertip collision geoms."""
    model = get_openarm_cell_spec().compile()
    data = mujoco.MjData(model)
    data.qpos[model.joint("openarm_left_joint4").qposadr[0]] = LEFT_JOINT4_HOME
    mujoco.mj_forward(model, data)

    fingertip_geoms = [
        f"finger_{k}_left_collision_{i:02d}"
        for k in ("inner", "outer")
        for i in range(4)
    ]
    centroid = np.mean([data.geom(name).xpos for name in fingertip_geoms], axis=0)
    np.testing.assert_allclose(data.site(LEFT_GRASP_SITE).xpos, centroid, atol=1e-3)


def test_spec_freezes_right_arm_and_lifter():
    model = get_openarm_cell_spec().compile()
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    assert joint_names == EXPECTED_JOINTS
    assert actuator_names == EXPECTED_ACTUATORS
    assert model.nkey == 0  # Stale 19-dim home keyframe removed.


def test_frozen_right_arm_matches_home_pose():
    # Reference: unmodified cell.xml at its home keyframe.
    ref_model = mujoco.MjSpec.from_file(str(OPENARM_CELL_XML)).compile()
    ref_data = mujoco.MjData(ref_model)
    mujoco.mj_resetDataKeyframe(ref_model, ref_data, 0)
    mujoco.mj_forward(ref_model, ref_data)
    ref_right_ee = ref_data.site("right_ee_control_point").xpos.copy()
    ref_left_ee = ref_data.site("left_ee_control_point").xpos.copy()

    model = get_openarm_cell_spec().compile()
    data = mujoco.MjData(model)
    data.qpos[model.joint("openarm_left_joint4").qposadr[0]] = LEFT_JOINT4_HOME
    mujoco.mj_forward(model, data)

    np.testing.assert_allclose(
        data.site("right_ee_control_point").xpos, ref_right_ee, atol=1e-6
    )
    np.testing.assert_allclose(
        data.site("left_ee_control_point").xpos, ref_left_ee, atol=1e-6
    )


def test_fingertip_collision_overrides():
    """Fingertips get condim=6 grasp contacts; other geoms keep their XML values."""
    from mjlab.entity import Entity

    model = Entity(get_openarm_robot_cfg()).spec.compile()
    tip = model.geom("finger_inner_left_collision_00")
    assert tip.condim[0] == 6
    assert tip.priority[0] == 2
    np.testing.assert_allclose(tip.friction, (1.0, 5e-3, 5e-4))
    # Non-fingertip geoms are untouched (disable_other_geoms=False).
    other = model.geom("link6_left_collision_00")
    assert other.condim[0] == 3
    assert other.contype[0] == 1


def test_entity_builds_with_xml_actuators():
    from mjlab.entity import Entity

    robot = Entity(get_openarm_robot_cfg())
    model = robot.spec.compile()
    assert model.nu == 8
    assert robot.is_fixed_base
    assert robot.is_actuated
    # init_state keyframe: left joint4 at home, everything else 0.
    key = model.key(0)
    qpos = np.zeros(9)
    qpos[3] = LEFT_JOINT4_HOME
    np.testing.assert_allclose(key.qpos, qpos, atol=1e-6)
