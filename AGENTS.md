# AGENTS.md — ST3D repo notes

This repo is an OpenPCDet-based 3D object detection / domain-adaptation
codebase. Active work happens on branch `v20260727_UADA3D_migration`
(porting UADA3D's DANN/GRL domain-adaptation mechanism into this repo).
Full migration plan and deep-dive reports live in the separate
`experiments_md` repo (`/home/koyama/code/experiments_md`); see its
`AGENTS.md` for report-writing conventions.

## Repo structure

- `pcdet/` — the actual library (datasets, models, config loading, utils).
  - `pcdet/config.py` — `cfg_from_yaml_file`/`merge_new_config`: loads a yaml
    cfg and recursively merges in any `_BASE_CONFIG_` file it references.
    **Child values always take precedence over the base's** (a base only
    fills in keys the child doesn't already set — see "Config system" below;
    this was a real bug until 2026-08-23, fixed in commit `de9f9d7`).
  - `pcdet/models/discriminators/` — Discriminator2 + GRL (gradient reversal
    layer), ported from UADA3D for DANN-style conditional adaptation.
  - `pcdet/models/detectors/da_second_net.py`, `da_centerpoint.py` — DA
    detector wrappers; note `get_loss()` return arity differs by head type
    (`AnchorHeadSingle.get_loss()` returns 3 values incl. `domain_loss`,
    `CenterHead.get_loss()` returns 2 — don't assume one pattern for both).
- `tools/` — entry-point scripts and experiment configs.
  - `tools/train.py` — plain (non-DA) training entry point.
  - `tools/adaptive_train.py` — DA training entry point (DANN/GRL configs),
    uses `train_utils/train_utils_adaptive.py` + wandb-based logging (NOT
    tensorboard — this repo's own convention, unlike vanilla OpenPCDet).
    Both scripts have a `--batch_size` CLI arg that must default to `None`
    (so the yaml's `OPTIMIZATION.BATCH_SIZE_PER_GPU` is actually used) — 
    `train.py` had a `default=16` bug that silently ignored the yaml's batch
    size for months (fixed in commit `5456291`); `test.py` has the same bug,
    not yet fixed.
  - `tools/cfgs/` — experiment configs, organized by domain-adaptation
    pair/dataset, e.g. `kitti2nuscenes_models/`, `waymo2nuscenes_models/`,
    `nuscenes2kitti_models/`, `pandaset2nuscenes_models/`,
    `lyft2nuscenes_models/`, plus older `da-*_models/` groupings and
    single-dataset dirs (`kitti_models/`, `nuscenes_models/`, etc.).
  - `tools/cfgs/dataset_configs/` — shared `_BASE_CONFIG_` dataset configs
    (e.g. `da_waymo_dataset.yaml`, `da_kitti_dataset.yaml`) referenced by
    many experiment configs. Editing a default here affects every consumer —
    grep for existing per-config overrides before changing one.
  - `tools/scripts/run_experiment.sh` — the SLURM job script actually used
    to launch runs (see "Experiment launching" below). Untracked/local-only
    (not committed) — it's a scratch file with one active `singularity exec`
    command line at a time, others commented out with status notes.
  - `tools/logs/` — SLURM stdout/stderr per job:
    `output_<jobid>_a6000_ada.txt` / `error_<jobid>_a6000_ada.txt`.
- `tests/` — pytest suite (e.g. `test_config_merge.py`, a regression test
  for the `_BASE_CONFIG_` merge bug above). Run via
  `singularity exec ... python3 -m pytest tests/ -v` (see below).

## Config system: `_BASE_CONFIG_` inheritance

Any yaml block can set `_BASE_CONFIG_: <path>` to inherit defaults from
another yaml file. Semantics: **the block's own keys always win**; the base
file only supplies values for keys the block doesn't set itself (recursing
into nested dicts). This is implemented via `_fill_missing_from_base()` in
`pcdet/config.py`. If you add an override in a config that uses
`_BASE_CONFIG_`, it will be respected — this was NOT always true before the
2026-08-23 fix (see `experiments_md/20260823_01_merge_new_config_base_override_bug.md`
for the full bug writeup and `tests/test_config_merge.py` for the
regression test).

## Experiment launching process

Real GPU runs go through SLURM on the `a6000_ada` partition (this host has
no GPU itself). Standard flow:

1. Edit `tools/scripts/run_experiment.sh`: uncomment exactly one
   `singularity exec ...` command line (the active experiment), leave others
   commented with a status note (job ID + PASSED/FAILED + reason). Bump
   `--extra_tag`/`--run_name` when resubmitting after a fix so wandb runs
   don't collide.
2. Submit from `tools/`: `sbatch scripts/run_experiment.sh` → prints
   `Submitted batch job <jobid>`.
3. Monitor: `squeue -u koyama` for queue/running status; tail
   `tools/logs/output_<jobid>_a6000_ada.txt` and
   `error_<jobid>_a6000_ada.txt` for progress/tracebacks.
4. The launched command is always of the form:
   ```
   singularity exec --nv --bind /home/koyama/data/:/storage \
     /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
     python3 adaptive_train.py --cfg_file cfgs/<domain>_models/<config>.yaml \
     --epochs 2 --num_epochs_to_eval 1 --run_name "<name>" --extra_tag <tag>
   ```
   Use `python3 train.py ...` instead of `adaptive_train.py` for
   non-DA/sourceonly configs.
   - Pandaset-as-source configs additionally need
     `--bind /home/koyama/code/ST3D:/root/ST3D` — Pandaset's cached
     `*_infos_*.pkl` files bake in absolute paths under `/root/ST3D/...` from
     preprocessing time, and `/root` isn't otherwise readable inside the
     container for the `koyama` user.
5. W&B run dirs land at `/storage/wandb/run-<timestamp>-<run_id>/files`
   (i.e. `/home/koyama/data/wandb/...` outside the container). Get the
   run_id either from that path (printed near the start of every run,
   survives crashes) or from the `wandb: 🚀 View run ... /runs/<run_id>`
   line at the end of a clean run. See `experiments_md/AGENTS.md` for the
   W&B-link convention used when writing reports about a run.

## Running tests

```
singularity exec --nv --bind /home/koyama/data/:/storage \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python3 -m pytest tests/ -v
```
(pytest 9.0.2 confirmed available in this container.)

## Known dataset/config gotchas

- `INFO_WITH_FAKELIDAR` (Waymo-only key): legacy fakelidar box format
  `[x,y,z,w,l,h,r]` vs standard `[x,y,z,dx,dy,dz,heading]`. Every DA config
  using `EVAL_METRIC: kitti` needs this `False`; the base
  `dataset_configs/da_waymo_dataset.yaml` default is `False` (flipped from
  `True` on 2026-08-23 — see config-system bug above for why a per-config
  override alone wasn't a reliable fix at the time).
- `FOV_POINTS_ONLY`: `pcdet/datasets/kitti/kitti_dataset.py`'s `__getitem__`
  accesses `self.dataset_cfg.FOV_POINTS_ONLY` directly (no `.get()`
  fallback) → `AttributeError` if not set. `da_kitti_dataset.yaml` (the base
  used by all KITTI-source/target configs) does NOT define it — every
  config using KITTI must set it explicitly (`True` or `False`) in its own
  `DATA_CONFIG`/`DATA_CONFIG_TAR` block.
- `da_second_net.py`'s `get_training_loss()` must unpack `dense_head.get_loss()`
  as `loss_rpn, tb_dict, _` (3 values, discarding `domain_loss`) since
  `AnchorHeadTemplate.get_loss()` always returns 3 — `da_centerpoint.py`
  doesn't need this since `CenterHead.get_loss()` only returns 2.

For deeper history/rationale on any of the above, see the dated reports
under `/home/koyama/code/experiments_md/` (esp. the `20260727_UADA3D_*` and
`20260823_*` files) and repo memory notes.
