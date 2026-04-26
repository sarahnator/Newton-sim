# Newton-sim

`Newton-sim` is a tiny Newton-on-Warp demo repo with six small passive 3D variants: a rigid-body ramp/floor/ball/tower scene, a rigid-body ramp/floor/ball/cup scene, a rigid-body ramp/floor/ball/cup/water scene using Genesis DFSPH water, a deformable jelly single-wall scene, a deformable jelly domino scene with two walls, and a Newton Franka/Panda robot arm whose end-effector accelerates straight into a wall at high or low contact height.

## Install

```bash
python -m pip install -e .
```

## Interactive Run

```bash
python scripts/run_ramp_tower.py --variant jelly-single --viewer gl
```

Jelly domino variant:

```bash
python scripts/run_ramp_tower.py --variant jelly-domino --viewer gl
```

Rigid variant:

```bash
python scripts/run_ramp_tower.py --variant rigid --viewer gl
```

Rigid cup variant:

```bash
python scripts/run_ramp_tower.py --variant rigid-cup --viewer gl
```

Rigid cup water variant:

```bash
conda run -n genesis-sim python scripts/run_ramp_cup_water_genesis.py
```

Robot-arm wall variant. The high and low modes use the same horizontal
end-effector trajectory and only change the contact height:

```bash
python scripts/run_ramp_tower.py --variant roboarm-wall --viewer gl
python scripts/run_ramp_tower.py --variant roboarm-wall --roboarm-push-height low --viewer gl
python scripts/run_ramp_tower.py --variant roboarm-wall --roboarm-push-height center --roboarm-friction high --viewer gl
python scripts/run_ramp_tower.py --variant roboarm-wall --roboarm-push-height center --roboarm-friction low --viewer gl
```

## USD Export

```bash
python scripts/run_ramp_tower.py --variant rigid --viewer usd --output-path outputs/ramp_tower_rigid.usd
```

## MP4 Video Export

```bash
python scripts/render_ramp_tower_video.py --variant jelly-single --output-path outputs/ramp_tower_jelly_single.mp4
python scripts/render_ramp_tower_video.py --variant jelly-domino --output-path outputs/ramp_tower_jelly_domino.mp4
python scripts/render_ramp_tower_video.py --variant rigid --output-path outputs/ramp_tower_rigid.mp4
python scripts/render_ramp_tower_video.py --variant rigid-cup --output-path outputs/ramp_cup_rigid.mp4
python scripts/render_ramp_tower_video.py --variant roboarm-wall --roboarm-push-height high --output-path outputs/roboarm_wall_high.mp4
python scripts/render_ramp_tower_video.py --variant roboarm-wall --roboarm-push-height low --output-path outputs/roboarm_wall_low.mp4
python scripts/render_ramp_tower_video.py --variant roboarm-wall --roboarm-push-height center --roboarm-friction high --output-path outputs/roboarm_wall_center_high_friction.mp4
python scripts/render_ramp_tower_video.py --variant roboarm-wall --roboarm-push-height center --roboarm-friction low --output-path outputs/roboarm_wall_center_low_friction.mp4
conda run -n genesis-sim python scripts/run_ramp_cup_water_genesis.py --output-path outputs/ramp_cup_water.mp4
```

## Headless Run

```bash
python scripts/run_ramp_tower.py --viewer null --num-frames 180
```

The repo currently contains only these six small variants.
