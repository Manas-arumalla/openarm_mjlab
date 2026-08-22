# OpenArm mjlab

PPO manipulation tasks for the [OpenArm](https://github.com/enactic/openarm) bimanual robot,
built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp + RSL-RL).

## Tasks

- `OpenArm-PickPlace` — the left arm picks the orange cube from the table and
  places it in the black tray, in the full OpenArm Cell scene (right arm and lifter frozen at home).

## Setup

```bash
git clone git@github.com:enactic/openarm_mjlab.git
cd openarm_mjlab
uv sync
```

## Train

Real training needs a Linux machine with a CUDA GPU:

```bash
uv run openarm-mjlab-train OpenArm-PickPlace \
  --env.scene.num-envs 4096 \
  --env.viewer.height 480 \
  --env.viewer.max-extra-envs 0 \
  --env.viewer.width 640 \
  --video True
```

CPU smoke test (e.g. on macOS):

```bash
CUDA_VISIBLE_DEVICES= uv run openarm-mjlab-train OpenArm-PickPlace \
  --env.scene.num-envs 4 \
  --agent.max-iterations 2
```

## Play a checkpoint

```bash
uv run openarm-mjlab-play OpenArm-PickPlace --checkpoint-file <path/to/checkpoint>
```

## Tests

```bash
uv run pytest
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
