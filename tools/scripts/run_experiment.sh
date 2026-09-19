#!/bin/bash
#SBATCH --job-name=singlenode    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=96G                # total memory per node (bumped 32G->64G->96G: /dev/shm tmpfs
                                  # used by PyTorch DataLoader worker IPC counts against this
                                  # cgroup limit; 64G was enough for second-sourceonly.yaml but
                                  # not centerpoint-sourceonly.yaml, whose dense heatmap targets
                                  # are heavier per DataLoader worker - still segfaulted at
                                  # epoch2 worker respawn even at --mem=64G + batch_size=6)
#SBATCH --gres=gpu:1             # number of allocated gpus per node
#SBATCH --partition=a6000_ada    # GPU node partition
#SBATCH --output=logs/output_%j_a6000_ada.txt
#SBATCH --error=logs/error_%j_a6000_ada.txt
#SBATCH --time=99:00:00

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error
set -e

# singularity exec --nv --bind /home/koyama/data/:/storage st3d_cuda12.sif /bin/bash
# singularity exec --nv --bind /home/koyama/data/:/storage ./st3d_cuda12.sif python3 scripts/experiments_20250520_post_MIRU2025_test_MULTI_CLASSES_NMS_false.py 
# singularity exec --nv --bind /home/koyama/data/:/storage ./st3d_cuda12.sif python3 scripts/experiments_20250518_post_MIRU2025_multi_lkwp2nuscenes_point_label_calibrated_backward_together_off.py
# singularity exec --nv --bind /home/koyama/data/:/storage ./st3d_cuda12.sif python3 scripts/experiments_20250514_post_MIRU2025_multi_lnwp2kitti_point_label_calibrated_backward_together_off.py
# singularity exec --nv --bind /home/koyama/data/:/storage ./st3d_cuda12.sif python3 scripts/experiments_20250512_multi_st3d_dann_source_point_label_calibrated_lyft_nuscenes2kitti_equal_sampling.py
# singularity exec --nv --bind /home/koyama/data/:/storage ./st3d_cuda12.sif python3 scripts/experiments_20250512_multi_st3d_target_point_label_calibrated_lyft_nuscenes2kitti_equal_sampling.py
#singularity exec --nv --bind /home/koyama/data/:/storage ./st3d_cuda12.sif python3 scripts/experiments_20250920_multi_st3d_dann_source_target_point_label_calibrated_lyft_nuscenes2kitti_equal_sampling.py
# singularity exec --nv --bind /home/koyama/data/:/storage ./st3d_cuda12.sif bash scripts/dist_train.sh 2 --cfg_file cfgs/da-post-MIRU2025/second_old_anchor_st3d_basebev_multi_lyft_nuscenes2kitti_dann_source_target_car_ped_point_label_calibrated.yaml
# singularity exec --nv --bind /home/koyama/data/:/storage ./st3d_cuda12.sif python3 scripts/experiments_20251012_post_MIRU2025_single_st3d_dann_source_target_point_label_calibrated.py
# singularity exec --nv --bind /home/koyama/data/:/storage ./st3d_cuda12.sif python3 scripts/experiments_20251102_post_MIRU2025_single_lyft2kitti_point_label_calibrated.py

# UADA3D->ST3D migration smoke tests: quick 2-epoch runs of the migrated kitti2nuscenes_models
# configs. One line active at a time; uncomment the next one once the previous smoke test
# completes successfully.
# Status: centerpoint-rospm-C.yaml (job 21303) and second-rospm-C.yaml (job 21306) both
# confirmed training successfully on GPU as of 2026-08-01.

# centerpoint-rospm-C.yaml (Discriminator2+GRL conditional adaptation, KITTI->nuScenes) - job 21303, PASSED
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/kitti2nuscenes_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_migration" --extra_tag 20260727_migration_test

# second-rospm-C.yaml (Discriminator2+GRL conditional adaptation, KITTI->nuScenes) - job 21306, PASSED
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/kitti2nuscenes_models/second-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_second_rospm_C_migration" --extra_tag 20260727_migration_test

# second-sourceonly.yaml (no-DA floor baseline, plain SECONDNet, runs via regular train.py)
# job 21377 segfaulted (no traceback) at the epoch1->epoch2 boundary while running solo.
# job 21378 (--workers 0) ran clean (no segfault) but hit eval-time CUDA OOM - fixed via
# torch.cuda.empty_cache() in eval_utils.py (commit 7b1301f), retesting as job 21381.
# job 21381 (--workers 0 + empty_cache fix) PASSED - both epochs, both evals, no OOM.
# job 21382 (--mem=64G + default --workers 4 + empty_cache fix) PASSED - no segfault, no OOM.
# CONFIRMED: --mem=64G (cgroup /dev/shm limit) was the segfault root cause; empty_cache() fix
# resolves the eval OOM. Recommended settings going forward: --mem=64G, default --workers.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/kitti2nuscenes_models/second-sourceonly.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_second_sourceonly_migration_mem64G_workers4" --extra_tag 20260727_migration_test

# centerpoint-sourceonly.yaml (no-DA floor baseline, plain CenterPoint, runs via regular train.py)
# job 21366 (concurrent w/ 21365, --workers 4, --mem=32G) segfaulted mid epoch 1 - not yet
# retried solo. job 21395 (--mem=64G + default --workers 4, batch=12) STILL segfaulted, but much
# earlier (2:16 into epoch 0) - found BATCH_SIZE_PER_GPU:12 here vs 6 in centerpoint-rospm-C.yaml/
# second-rospm-C.yaml (the passing DA configs); fixed to 6. job 21396 (--mem=64G, batch=6) got
# past checkpoint_epoch_1 but STILL segfaulted partway into epoch 2 at the same point. job 21416
# (--mem=96G, batch=6) segfaulted at the SAME elapsed time/epoch2 boundary as 21396 - --mem bump
# had ZERO effect, ruling out the /dev/shm cgroup theory for CenterPoint specifically. Likely a
# fork()+CUDA DataLoader worker-respawn incompatibility instead. Reusing the proven --workers 0
# workaround (as used for second-sourceonly.yaml jobs 21378/21381) to unblock, pending a proper
# fix (persistent_workers=True or spawn context) investigated separately.
# job 21933 (--workers 0 workaround, running concurrently with the persistent_workers test below).
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/kitti2nuscenes_models/centerpoint-sourceonly.yaml --epochs 2 --num_epochs_to_eval 1 --workers 0 --run_name "test_centerpoint_sourceonly_migration_batch6_workers0" --extra_tag 20260727_migration_test

# Proper fix applied: added persistent_workers=(workers > 0) to build_dataloader() in
# pcdet/datasets/__init__.py, so DataLoader workers are no longer torn down and re-forked at
# every epoch boundary (avoids the fork()+CUDA-context collision). Tested with default
# --workers 4 restored (no --workers 0 workaround) to validate the real fix at full throughput.
# job 21933 (--workers 0 workaround, no DataLoader subprocess at all) STILL segfaulted, mid
# epoch 0 at 14:37 elapsed (not at an epoch boundary) - rules out the fork()+CUDA worker-respawn
# theory entirely. job 21934 (persistent_workers=True fix) segfaulted even earlier, at 1:48 into
# epoch 0. Crash timing is now inconsistent across runs (1:48/2:16/7:30/14:37), suggesting a
# data-dependent bug (e.g. a specific batch triggering an issue in a custom CUDA kernel like
# CenterPoint's box decoding or circle_nms) rather than a worker-lifecycle or --mem issue.
# No core dump accessible for post-mortem (ulimit -c 0, crashes routed through apport to
# /var/crash, not readable). Bisected with BATCH_SIZE_PER_GPU: 2 (job 21935) - STILL segfaulted
# at nearly identical timing, seemingly ruling out memory pressure too.
#
# ROOT CAUSE FOUND: train.py's --batch_size CLI arg defaulted to 16 (not None), so the
# 'if args.batch_size is None: args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU' branch
# was NEVER triggered - every single run above (21395/21396/21416/21933/21934/21935) silently
# trained at batch_size=16 regardless of the YAML, since run_experiment.sh never passed
# --batch_size explicitly. adaptive_train.py (used by the passing DA configs) already had the
# correct default=None. Fixed train.py's default to None (commit 5456291). Confirmed via an
# interactive session (job 21937, --batch_size 2 passed explicitly pre-fix) that training
# survived to 70% of epoch 0 (5:07 elapsed) with zero crashes - far past every prior crash
# point - strongly indicating batch_size=16 (not a data/kernel bug) was the real cause.
#
# Now resubmitting via normal sbatch with the CLI default fix in place, BATCH_SIZE_PER_GPU
# restored to 6 in the YAML (matching centerpoint-rospm-C.yaml/second-rospm-C.yaml), and no
# --batch_size CLI override needed since the fix makes the YAML value take effect correctly.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/kitti2nuscenes_models/centerpoint-sourceonly.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_sourceonly_migration_batch6_realfix" --extra_tag 20260727_migration_test

# 2026-08-17: quick 2-epoch smoke tests of the 4 NEW centerpoint-rospm-C.yaml configs
# (nuscenes2kitti_models, waymo2nuscenes_models, pandaset2nuscenes_models, lyft2nuscenes_models) -
# same adaptive_train.py + Discriminator2/GRL pattern as kitti2nuscenes_models/centerpoint-rospm-C.yaml
# (job 21303, PASSED). nuscenes2kitti_models is a direct UADA3D port; the other 3 are new
# (pattern-matched, no UADA3D reference) - see header comments in each yaml for assumptions.
# job 22359 (nuscenes2kitti) - PASSED, trained cleanly (loss 120->~10 within 1 epoch).
# job 22360/22363 (waymo2nuscenes) - FAILED twice: AssertionError (INFO_WITH_FAKELIDAR must be
#   False for kitti_eval). The per-config yaml override added for job 22363 did NOT fix it -
#   turned out to be a real bug in pcdet/config.py's merge_new_config, where _BASE_CONFIG_
#   values silently clobbered child overrides for shared scalar keys (see
#   experiments_md/20260823_01_merge_new_config_base_override_bug.md). FIXED at the source
#   (commit de9f9d7: merge_new_config now lets child values win; also flipped
#   da_waymo_dataset.yaml's INFO_WITH_FAKELIDAR default to False). Resubmitted as job 22817
#   to confirm the real fix.
# job 22361 (pandaset2nuscenes) - FAILED: PermissionError reading
#   /root/ST3D/data/pandaset/dataset/014/lidar/00.pkl.gz. Root cause: pandaset_infos_train.pkl
#   stores ABSOLUTE paths baked in at preprocessing time under /root/ST3D/..., not relative
#   paths - a pre-existing infra quirk unrelated to the new yaml. /root is unreadable by the
#   koyama user inside the container, so those reads fail unless /root/ST3D is itself bound to
#   the live repo. Fixed by adding `--bind /home/koyama/code/ST3D:/root/ST3D` to this job's
#   singularity exec call (verified interactively that this resolves the pkl.gz read).
#   Resubmitted as job 22364.
# job 22362 (lyft2nuscenes) - running, no errors yet as of report time.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/nuscenes2kitti_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_nuscenes2kitti" --extra_tag 20260817_migration_test
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/waymo2nuscenes_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_waymo2nuscenes" --extra_tag 20260823_migration_test_v3
#singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/pandaset2nuscenes_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_pandaset2nuscenes" --extra_tag 20260817_migration_test_v2

# 2026-08-23: re-verifying the other 4 domain-pair configs (kitti2nuscenes, nuscenes2kitti,
# lyft2nuscenes, pandaset2nuscenes) now that the merge_new_config bug fix (commit de9f9d7) makes
# DATA_AUGMENTOR/DATA_PROCESSOR overrides actually apply. Their prior "PASSED" jobs (21303, 22359,
# 22362, 22364) only proved the OLD (incorrectly-merged, base-defaults) variant worked - these are
# the first true runs of each config's actually-intended settings. New extra_tag per run to avoid
# wandb collision. One line active at a time, same as above.
# Submitted 2026-08-23: job 22818 (kitti2nuscenes), 22819 (nuscenes2kitti), 22820 (lyft2nuscenes),
# 22821 (pandaset2nuscenes) - all queued (PD/Priority), results not yet known.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/kitti2nuscenes_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_kitti2nuscenes_bugfix_verify" --extra_tag 20260823_bugfix_verify
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/nuscenes2kitti_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_nuscenes2kitti_bugfix_verify" --extra_tag 20260823_bugfix_verify
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/lyft2nuscenes_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_lyft2nuscenes_bugfix_verify" --extra_tag 20260823_bugfix_verify
#singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/pandaset2nuscenes_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_pandaset2nuscenes_bugfix_verify" --extra_tag 20260823_bugfix_verify
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/lyft2nuscenes_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_lyft2nuscenes" --extra_tag 20260817_migration_test

# 2026-08-23: full-length (40-epoch, config default) training runs of both kitti2nuscenes DA
# configs, post merge_new_config fix (commit de9f9d7) - prior runs (job 21303/21306, and the
# 2026-08-23 bugfix_verify smoke tests) were only 2-epoch smoke tests. No --num_epochs_to_eval
# override (default 100 evaluates every epoch, since num_epochs_to_eval > total epochs).
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/kitti2nuscenes_models/centerpoint-rospm-C.yaml --epochs 40 --run_name "train_centerpoint_rospm_C_kitti2nuscenes_full40ep" --extra_tag 20260823_full_train
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/kitti2nuscenes_models/second-rospm-C.yaml --epochs 40 --run_name "train_second_rospm_C_kitti2nuscenes_full40ep" --extra_tag 20260823_full_train

# 2026-08-23: smoke test of the head_per_dataset ontology branch-precedence bugfix
# (pcdet/datasets/dataset.py, see experiments_md/20260823_06_head_per_dataset_ontology_branch_precedence_bug.md)
# using a real multi-dataset (waymo+lyft+pandaset source -> kitti target) self-training config
# that legitimately sets SELF_TRAIN.MODEL_TEACHER.ONTOLOGY: 'head_per_dataset' with
# per-dataset-prefixed CLASS_NAMES. 1-epoch smoke test via train.py to confirm dataset
# construction for all 4 datasets succeeds and produces non-empty GT boxes with the fix in place.
# job 22842 CRASHED (FOV_POINTS_ONLY assert canary) - fixed, see
# experiments_md/20260829_01_job_result_confirmation_and_kitti_eval_bugfixes.md. Not resubmitted yet.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/da-waymo-lyft-pandaset-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_waymo_lyft_pandaset_to_kitti_st3d_car_basebev.yaml --epochs 1 --num_epochs_to_eval 1 --run_name "test_head_per_dataset_ontology_bugfix" --extra_tag 20260823_bugfix_verify

# 2026-08-29: genuine 40-epoch centerpoint-sourceonly.yaml (KITTI-only, no adaptation) baseline
# for kitti2nuscenes. Every prior run of this config (21366/21395/21396/21416/21933/21934/21935/
# 21938) used a leftover --epochs 2 smoke-test override, so no real full-length baseline has ever
# existed to compare against the 40-epoch DANN runs (22822/22823). See
# experiments_md/20260829_02_kitti2nuscenes_full40ep_low_ap_investigation.md (correction section).
# No --num_epochs_to_eval override, matching the 22822/22823 pattern (default evaluates every epoch).
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/kitti2nuscenes_models/centerpoint-sourceonly.yaml --epochs 40 --run_name "train_centerpoint_sourceonly_kitti2nuscenes_full40ep" --extra_tag 20260829_full_train

# 2026-08-29: kitti2kitti same-domain sanity check (no domain gap) - new cfgs/kitti2kitti_models/
# configs, exact copies of the kitti2nuscenes_models sourceonly configs with DATA_CONFIG_TAR
# redirected to KITTI val instead of nuScenes (see gt_sampling-absence finding discussion in
# session). 40-epoch full training, matching job 23360's settings for direct comparability.
# One line active at a time; job 23360 (kitti2nuscenes centerpoint-sourceonly) is commented above
# only to prevent this NEW submission from re-running it in the same job - it is already running
# independently as its own already-submitted SLURM job.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/kitti2kitti_models/second-sourceonly.yaml --epochs 40 --run_name "train_second_sourceonly_kitti2kitti_full40ep" --extra_tag 20260829_kitti2kitti
# job 23366 (second-sourceonly kitti2kitti) submitted 2026-08-29.
# job 23367 (centerpoint-sourceonly kitti2kitti) submitted 2026-08-29.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/kitti2kitti_models/centerpoint-sourceonly.yaml --epochs 40 --run_name "train_centerpoint_sourceonly_kitti2kitti_full40ep" --extra_tag 20260829_kitti2kitti

# 2026-08-29: gt_sampling-ENABLED variants of the above two kitti2kitti configs, to isolate
# whether the gt_sampling-absence finding is itself a significant contributor to low AP,
# independent of domain gap (see experiments_md/20260829_02..., "UPDATE" section).
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/kitti2kitti_models/second-sourceonly-gtsampling.yaml --epochs 40 --run_name "train_second_sourceonly_kitti2kitti_gtsampling_full40ep" --extra_tag 20260829_kitti2kitti_gtsampling
# job 23371 (second-sourceonly-gtsampling kitti2kitti) submitted 2026-08-29.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/kitti2kitti_models/centerpoint-sourceonly-gtsampling.yaml --epochs 40 --run_name "train_centerpoint_sourceonly_kitti2kitti_gtsampling_full40ep" --extra_tag 20260829_kitti2kitti_gtsampling
# job 23372 (centerpoint-sourceonly-gtsampling kitti2kitti) submitted 2026-08-29.

# 2026-08-30: first real run of the new PandaSet sensor-only domain-gap config
# (pandar64 spinning source -> PandarGT flash target, same vehicle/scene/labels - see
# experiments_md/20260829_03_pandaset_sensor_gap_config_and_da_ablation_taxonomy.md section 2).
# 2-epoch smoke test first (not yet run on GPU before), same DACenterPoint+Discriminator2
# adaptive_train.py pattern as the other *2nuscenes rospm-C configs. Needs the
# --bind /home/koyama/code/ST3D:/root/ST3D workaround like other Pandaset-as-source configs
# (pandaset_infos_*.pkl bake in absolute /root/ST3D/... paths from preprocessing time).
# job 23407 - PASSED: ran cleanly through both epochs + both evals, no crash, W&B run
# https://wandb.ai/nagisa/st3d/runs/t4z49f31. Near-zero BEV/3D AP after 2 epochs is expected for
# a smoke test, not a real result. This confirms end-to-end correctness (dataset construction for
# both LIDAR_DEVICE values, DACenterPoint+Discriminator2 training loop, kitti_eval dispatch) but
# is NOT the real experiment - no full-length run has been done yet.
#singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/pandaset-pandar64-to-pandargt_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_pandaset_pandar64_to_pandargt" --extra_tag 20260830_sensor_gap_smoke_test

# 2026-09-07: CLASS_NAMES ['Car','Pedestrian','Bicycle'] assumption re-verified against real GT
# box counts for PandarGT (device=1) - all three classes have ample boxes in both train/val splits
# (Car ~631k/183k, Pedestrian ~73k/29k, Bicycle ~6.8k/3.2k train/val) - see
# experiments_md/20260907_02_pandaset_pandar64_to_pandargt_classnames_verification.md and
# tools/verify_pandaset_pandargt_classes.py. No config change needed.
# job 24180 - submitted 2026-09-07, genuine 40-epoch full training run (config default NUM_EPOCHS).
#singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/pandaset-pandar64-to-pandargt_models/centerpoint-rospm-C.yaml --epochs 40 --run_name "train_centerpoint_rospm_C_pandaset_pandar64_to_pandargt_full40ep" --extra_tag 20260907_full_train

# 2026-09-07: REVERSE-direction sibling - PandarGT (device=1) as source, pandar64 (device=0) as
# unsupervised target (new cfgs/pandaset-pandargt-to-pandar64_models/centerpoint-rospm-C.yaml -
# see experiments_md/20260907_02_pandaset_pandar64_to_pandargt_classnames_verification.md for the
# CLASS_NAMES box-count evidence, reused unchanged from the forward direction; both devices already
# verified to have ample Car/Pedestrian/Bicycle boxes in both splits). Config-load assertions
# passed; pytest still green. Per explicit instruction, drafted directly as a full 40-epoch run
# (no 2-epoch smoke test first, unlike the forward direction's job 23407) - NOTE this means it has
# zero GPU runtime history, unlike the forward direction's config.
# job 24181 - submitted 2026-09-07, genuine 40-epoch full training run (config default NUM_EPOCHS).
#singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/pandaset-pandargt-to-pandar64_models/centerpoint-rospm-C.yaml --epochs 40 --run_name "train_centerpoint_rospm_C_pandaset_pandargt_to_pandar64_full40ep" --extra_tag 20260907_full_train

# 2026-08-30: retry of job 23371 (second-sourceonly-gtsampling, kitti2kitti), which segfaulted
# (no Python traceback, core dumped) at epoch 31/40 with zero preceding warnings/errors - see
# experiments_md/20260830_03_kitti2kitti_sanity_check_and_domain_gap_quantification.md section 5.
# Root cause not pinned down precisely (no core dump accessible), but isolated to gt_sampling's
# database_sampler.py CPU IoU collision-filter call, which had run cleanly ~9617 iterations
# before failing - profile consistent with a rare native-crash edge case, not a systematic
# config/resource bug. Plan: retry as-is first (cheapest option); only pursue defensive
# NaN/degenerate-box filtering in database_sampler.py if this retry also fails.
# job 23475 (retry1) - SEGFAULTED again (see experiments_md/active_context.md) - not retried a
# third time per the "don't blindly retry" guidance; next step is defensive NaN/degenerate-box
# filtering in database_sampler.py, not yet implemented.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/kitti2kitti_models/second-sourceonly-gtsampling.yaml --epochs 40 --run_name "train_second_sourceonly_kitti2kitti_gtsampling_full40ep_retry1" --extra_tag 20260830_kitti2kitti_gtsampling_retry1

# =====================================================================================
# 2026-09-19: PHASE 0 of the IEEE Access paper experiment programme (12 runs).
# Plan: experiments_md/20260919_03_ieee_access_paper_experiment_gap_analysis.md section 6.
# Audit motivating it: experiments_md/20260919_01_lidar_to_lidar_da_research_summary.md section 5.
#
# Purpose: fill the cheap cells that need no new training-loop code, and decide how large the
# remaining programme is. Cluster caps are 4 running / 8 queued, so run in waves of 4 and keep
# exactly ONE line uncommented at a time, per this file's existing convention.
#
# METRIC NOTE (verified 2026-09-19 by reading pcdet/datasets/kitti/kitti_object_eval_python/eval.py
# and kitti_utils.transform_annotations_to_kitti_format): the BEV AP math is standard (R40
# interpolation, rotated BEV IoU on the camera x-z plane), BUT non-KITTI eval targets (nuScenes,
# PandaSet) have their 2D bbox faked as [0,0,50,50] with truncation/occlusion 0, so every GT box
# passes all three difficulty levels and nothing is ignored - easy==moderate==hard, and the numbers
# are "all objects, no difficulty filter", NOT "moderate". KITTI-target runs (C4/C5/C6) DO use real
# KITTI annotations and are genuinely difficulty-stratified. Do not mix the two regimes in one
# table without saying so. Ignore the bbox/aos columns entirely for non-KITTI targets.
#
# ---------- WAVE 1: the two branch points ----------
# A1/A2 are a matched pair (PCGrad off vs on) and must be compared to each other - rows 10 and 11
# of the paper's matrix for dense->sparse, the first time those rows exist in one of the four
# claimed settings. Both REQUIRE the source-pretrained Lyft checkpoint: train.py guards
# --pretrained_model/--pretrained_model_teacher with plain `if not None`, so omitting them does not
# error, it silently trains a randomly-initialised frozen teacher and emits garbage pseudo-labels.
# Checkpoint verified present: 65MB, 2025-04-07.
# C1/C4 decide whether MIRU2025's published numbers reproduce under the bugfixed codebase
# (_BASE_CONFIG_ merge fix de9f9d7, --batch_size default fix 5456291), which determines whether the
# remaining programme is ~36 runs or ~44. Expected targets: C1 ~27.4, C4 ~14.1 BEV Car AP.

# A1 - S1 full method, PCGrad OFF
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/da-post-MIRU2025/second_old_anchor_st3d_basebev_multi_lyft2nuscenes_dann_source_target_car_ped_point_label_calibrated.yaml --batch_size 12 --pretrained_model /storage/wandb/run-20250303_153658-ggpm88cg/files/ckpt/checkpoint_epoch_50.pth --pretrained_model_teacher /storage/wandb/run-20250303_153658-ggpm88cg/files/ckpt/checkpoint_epoch_50.pth --set SELF_TRAIN.USE_TORCHJD False --run_name "phase0_A1_lyft2nuscenes_full_method_pcgrad_off" --extra_tag 20260919_phase0

# A2 - S1 full method, PCGrad ON
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/da-post-MIRU2025/second_old_anchor_st3d_basebev_multi_lyft2nuscenes_dann_source_target_car_ped_point_label_calibrated.yaml --batch_size 12 --pretrained_model /storage/wandb/run-20250303_153658-ggpm88cg/files/ckpt/checkpoint_epoch_50.pth --pretrained_model_teacher /storage/wandb/run-20250303_153658-ggpm88cg/files/ckpt/checkpoint_epoch_50.pth --set SELF_TRAIN.USE_TORCHJD True --run_name "phase0_A2_lyft2nuscenes_full_method_pcgrad_on" --extra_tag 20260919_phase0

# C1 - S1 naive (reproduction check vs MIRU2025 Table 2 = 27.4 BEV Car AP). Config default: batch 16, 50 epochs.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_default.yaml --run_name "phase0_C1_lyft2nuscenes_naive" --extra_tag 20260919_phase0

# C4 - S2 naive (reproduction check vs MIRU2025 Table 2 = 14.1 BEV Car AP). Config default: batch 16, 30 epochs.
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/da-MIRU2025/second_old_anchor_basebev_multi_nuscenes2kitti_car_ped_default.yaml --run_name "phase0_C4_nuscenes2kitti_naive" --extra_tag 20260919_phase0

# ---------- WAVE 2: floors + the reproduction verdict ----------
# B1/B2 are the unadapted floors for the PandaSet scan<->flash setting. Without them, jobs 24180
# (Car 3D 4.97 / BEV 7.49) and 24181 (15.67 / 24.28) cannot separate "adaptation worked" from
# "pandar64 is simply the easier target". PandaSet-as-source needs the /root/ST3D bind mount
# because pandaset_infos_*.pkl bake in absolute /root/ST3D/... paths.

# B1 - S3 floor: pandar64 -> PandarGT source-only
#singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/pandaset-pandar64-to-pandargt_models/centerpoint-sourceonly.yaml --run_name "phase0_B1_pandar64_to_pandargt_sourceonly_floor" --extra_tag 20260919_phase0

# B2 - S4 floor: PandarGT -> pandar64 source-only
#singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/pandaset-pandargt-to-pandar64_models/centerpoint-sourceonly.yaml --run_name "phase0_B2_pandargt_to_pandar64_sourceonly_floor" --extra_tag 20260919_phase0

# C3 - S1 point+label calibrated (vs MIRU2025 = 31.2 BEV Car AP)
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_point_label_calibrated.yaml --run_name "phase0_C3_lyft2nuscenes_point_label_calibrated" --extra_tag 20260919_phase0

# C6 - S2 point+label calibrated (vs MIRU2025 = 14.2 BEV Car AP; density correction is a KNOWN
# no-op in this direction since nuScenes is sparser than KITTI at every distance, so min(Pt/Ps,1)
# does nothing - reproducing the flat/slightly-down result IS the useful outcome, not a failure)
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/da-MIRU2025/second_old_anchor_basebev_multi_nuscenes2kitti_car_ped_point_label_calibrated.yaml --run_name "phase0_C6_nuscenes2kitti_point_label_calibrated" --extra_tag 20260919_phase0

# ---------- WAVE 3: ceilings + remaining table row ----------
# B3/B4 are supervised oracles (train AND eval on the same device) - deliberately NOT adaptation
# results. They are the ceilings that make a floor of ~5 AP and an adapted result of ~7 AP
# interpretable, and they also quantify how much of the ~3x directional asymmetry between jobs
# 24180/24181 is intrinsic target difficulty rather than anything about adaptation.

# B3 - S3 ceiling: oracle on PandarGT
#singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/pandaset-pandar64-to-pandargt_models/centerpoint-oracle.yaml --run_name "phase0_B3_pandargt_oracle_ceiling" --extra_tag 20260919_phase0

# B4 - S4 ceiling: oracle on pandar64
#singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/pandaset-pandargt-to-pandar64_models/centerpoint-oracle.yaml --run_name "phase0_B4_pandar64_oracle_ceiling" --extra_tag 20260919_phase0

# C2 - S1 point-only calibrated (vs MIRU2025 = 30.3 BEV Car AP)
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_point_calibrated.yaml --run_name "phase0_C2_lyft2nuscenes_point_calibrated" --extra_tag 20260919_phase0

# C5 - S2 point-only calibrated (vs MIRU2025 = 13.8 BEV Car AP)
#singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/da-MIRU2025/second_old_anchor_basebev_multi_nuscenes2kitti_car_ped_point_calibrated.yaml --run_name "phase0_C5_nuscenes2kitti_point_calibrated" --extra_tag 20260919_phase0
