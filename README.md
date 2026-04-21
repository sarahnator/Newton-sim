# Newton-sim

`Newton-sim` is a tiny Newton-on-Warp demo repo with five small passive 3D variants of the same setup: a rigid-body ramp/floor/ball/tower scene, a rigid-body ramp/floor/ball/cup scene, a rigid-body ramp/floor/ball/cup/water scene using MPM water, a deformable jelly single-wall scene, and a deformable jelly domino scene with two walls.

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
python scripts/run_ramp_tower.py --variant rigid-cup-water --viewer gl
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
python scripts/render_ramp_tower_video.py --variant rigid-cup-water --output-path outputs/ramp_cup_water.mp4
```

## Headless Run

```bash
python scripts/run_ramp_tower.py --viewer null --num-frames 180
```

The repo currently contains only these five small variants.
