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
# job 23407 - submitted 2026-08-30.
#singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 adaptive_train.py --cfg_file cfgs/pandaset-pandar64-to-pandargt_models/centerpoint-rospm-C.yaml --epochs 2 --num_epochs_to_eval 1 --run_name "test_centerpoint_rospm_C_pandaset_pandar64_to_pandargt" --extra_tag 20260830_sensor_gap_smoke_test

# 2026-08-30: retry of job 23371 (second-sourceonly-gtsampling, kitti2kitti), which segfaulted
# (no Python traceback, core dumped) at epoch 31/40 with zero preceding warnings/errors - see
# experiments_md/20260830_03_kitti2kitti_sanity_check_and_domain_gap_quantification.md section 5.
# Root cause not pinned down precisely (no core dump accessible), but isolated to gt_sampling's
# database_sampler.py CPU IoU collision-filter call, which had run cleanly ~9617 iterations
# before failing - profile consistent with a rare native-crash edge case, not a systematic
# config/resource bug. Plan: retry as-is first (cheapest option); only pursue defensive
# NaN/degenerate-box filtering in database_sampler.py if this retry also fails.
singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 train.py --cfg_file cfgs/kitti2kitti_models/second-sourceonly-gtsampling.yaml --epochs 40 --run_name "train_second_sourceonly_kitti2kitti_gtsampling_full40ep_retry1" --extra_tag 20260830_kitti2kitti_gtsampling_retry1
