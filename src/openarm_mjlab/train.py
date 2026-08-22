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

"""Train OpenArm tasks with mjlab's RSL-RL CLI.

Registers openarm_mjlab tasks, then delegates to mjlab's train entry point.
Usage: uv run openarm-mjlab-train OpenArm-PickPlace --env.scene.num-envs 4096
"""

from . import tasks  # noqa: F401  # Registers OpenArm tasks.
from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
