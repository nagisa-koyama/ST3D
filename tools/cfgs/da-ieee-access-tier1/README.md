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
