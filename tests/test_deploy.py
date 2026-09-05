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

"""Tests for exporting a policy and its scene for use outside mjlab."""

import numpy as np
import pytest

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from openarm_mjlab.deploy.export import _merge_duplicate_default_classes


def test_merge_collapses_a_default_class_nested_inside_itself():
    """mjlab's scene export nests <default class="robot/main"> inside itself.

    MuJoCo rejects a repeated class name, so without this the exported scene
    does not load at all. Every child of the inner block must survive.
    """
    xml = (
        "<mujoco><default>"
        '<default class="a"><default class="x"/>'
        '<default class="a"><default class="y"/><default class="z"/></default>'
        "</default></default></mujoco>"
    )
    merged_xml, count = _merge_duplicate_default_classes(xml)
    assert count == 1
    import xml.etree.ElementTree as ET

    root = ET.fromstring(merged_xml)
    classes = [d.get("class") for d in root.iter("default")]
    assert classes.count("a") == 1, "the repeat should be gone"
    for kept in ("x", "y", "z"):
        assert kept in classes, f"child {kept} was dropped by the merge"


def test_merge_is_a_no_op_when_there_is_nothing_to_merge():
    xml = '<mujoco><default><default class="a"/><default class="b"/></default></mujoco>'
    merged_xml, count = _merge_duplicate_default_classes(xml)
    assert count == 0
    assert merged_xml == xml


@pytest.mark.parametrize("group", ["train", "interp"])
def test_exported_scene_matches_the_mjlab_model(tmp_path, group):
    """A rewritten XML is only useful if it compiles to the same physics."""
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from openarm_mjlab.deploy.export import export_scene_self_contained
    from openarm_mjlab.tasks.families import DOOR_INSTANCE_IDS

    task_id = DOOR_INSTANCE_IDS[group][0]
    out = tmp_path / group
    export_scene_self_contained(task_id, out)

    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    reference = env.sim.mj_model
    exported = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))

    for field in ("nq", "nv", "njnt", "ngeom", "nbody", "nu", "nsensor"):
        assert getattr(reference, field) == getattr(exported, field), field
    for field in (
        "geom_friction",
        "geom_priority",
        "geom_size",
        "body_mass",
        "jnt_range",
        "dof_damping",
        "geom_solref",
    ):
        assert np.allclose(
            np.asarray(getattr(reference, field)),
            np.asarray(getattr(exported, field)),
            atol=1e-9,
        ), field
    env.close()
