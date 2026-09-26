# da-ieee-access-tier1: short-schedule proxies for screening methods

Each config here inherits one full `da-ieee-access` row through `_BASE_CONFIG_` and changes only
the schedule: `NUM_EPOCHS: 4` instead of 30. Everything else is inherited unchanged: model, data,
correction, ROS, LR, global batch 6. `adam_onecycle` derives its total steps from the runtime
epoch count, so a 4-epoch run gets a full warm-up and anneal of its own. It is NOT the same as
reading epoch 4 of a 30-epoch run, which would still be at high LR.

Budget: 4 x 3,150 = 12,600 optimizer steps (about 1/7.5 of the full row). About 2-2.5 h on
2 GPUs at recent node occupancy (the full rows ran 28-34 min per epoch). The global-correction
row adds about 5 min of calibration before training starts.

`NUM_EPOCHS_TO_EVAL: 1` scores epochs 3 AND 4 (the start_epoch filter is off by one). That is
deliberate: the two readings give a rough in-run noise estimate for free (+5 min).

## First use: validate the proxy before trusting it

The three Lyft -> nuScenes arms have full 30-epoch results against one control:

| arm | config | full-run Car BEV AP_R40 |
|---|---|---|
| MeanVFE baseline | centerpoint-sourceonly-lyft-4ep | 23.17 (job 25835) |
| global density correction | centerpoint-global-lyft2nuscenes-4ep | 28.22 (+5.05, job 25781) |
| GBlobs | centerpoint-gblobs-sourceonly-lyft-4ep | 31.43 (+8.26, job 25723) |

The proxy is adopted if it reproduces the ORDER baseline < global < GBlobs with gaps of the same
sign. Absolute AP will be lower and is not compared to full-run numbers.

Promotion rule once adopted: a gap of 3 AP or more over the matched 4-epoch control is promoted
to a full run. Below that, run a second seed first. A single full-length epoch already carries
about +-1.5 AP of noise, and the proxy's noise is expected to be larger.

Launch (one 2-GPU job each, via the frozen-code launcher):

    WANDB_NOTES="..." scripts/submit.sh scripts/run_sourceonly_2gpu.sh \
        cfgs/da-ieee-access-tier1/<config>.yaml tier1

## Self-training arms (foreground correction and its controls)

The three rows share one self-training block, one teacher and one starting checkpoint, and differ
only in the source-side correction. So each comparison is one variable:

| arm | proxy config | what differs from the foreground row |
|---|---|---|
| ST3D + foreground correction | centerpoint-foreground-lyft2nuscenes-4ep | (treatment) |
| ST3D + global correction | centerpoint-st3d-global-lyft2nuscenes-4ep | foreground channel off |
| ST3D, no correction | centerpoint-st3d-lyft2nuscenes-4ep | no correction at all |

The parents of the two controls are new full rows in `da-ieee-access/`; a test pins each to the
foreground row outside its correction keys. `PROG_AUG.UPDATE_AUG` is rescaled [8,15,23] -> [1,2,3]
with the schedule. Otherwise the curriculum would never fire in 4 epochs.

Run them on ONE GPU, like the full foreground row (job 26220). Self-training has never run under
DDP here. About 1.35 h per epoch there, so about 5.5-6 h per arm.

Teacher and student start: `iwg6l5v1` epoch 30 (the global-correction model, 28.22) for all three,
the same checkpoint job 26220 used. For textbook ST3D (the paper's baseline row), run the plain arm
a second time from the source-only model `ldb35c2o` epoch 30.

    T=/storage/wandb/run-20260923_093504-iwg6l5v1/files/ckpt/checkpoint_epoch_30.pth
    WANDB_NOTES="..." scripts/submit.sh --gres=gpu:1 scripts/run_sourceonly_2gpu.sh \
        cfgs/da-ieee-access-tier1/<config>.yaml tier1 <run_name> \
        --pretrained_model $T --pretrained_model_teacher $T

## UADA3D arm: BLOCKED at its matched batch

    WANDB_NOTES="..." ENTRY=adaptive_train.py BATCH=12 scripts/submit.sh --gres=gpu:1 \
        scripts/run_sourceonly_2gpu.sh cfgs/da-ieee-access-tier1/centerpoint-uada3d-lyft2nuscenes-4ep.yaml tier1

At `BATCH=12` (6 source + 6 target) the Lyft row segfaults deterministically near iteration 493 of
epoch 0 (jobs 25928, 25931). Job 26275's backtrace puts the crash in cuDNN
`convolution_backward`. The proxy would crash in its first ~20 minutes the same way. `BATCH=6`
(3+3) ran for hours in job 25922, but it halves the source batch and doubles the step count, so it
is NOT matched to the other arms. Only use it if you accept that.
