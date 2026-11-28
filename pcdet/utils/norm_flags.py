"""Which config features `--sync_bn` would silently break (experiments_md 20261003_14 §3; user 2026-11-28: 2-GPU rows use
SyncBatchNorm from now on).

`train.py` converts BatchNorm to SyncBatchNorm in an `if args.sync_bn: ... elif SELF_TRAIN.DSNORM: ...` branch, so a DSNorm row
given `--sync_bn` would lose DSNorm without a word. RANGE_BN keeps per-range statistics per process (single GPU only), and
NORM_SWAP leaves no BatchNorm to synchronize. `train.py` refuses `--sync_bn` for any of them.
"""


def sync_bn_conflicts(cfg):
    out = []
    st = cfg.get('SELF_TRAIN', None)
    if st and st.get('DSNORM', None):
        out.append('SELF_TRAIN.DSNORM: train.py converts to SyncBatchNorm or to DSNorm, never both, so DSNorm would be skipped')
    model = cfg.get('MODEL', None) or {}
    if model.get('RANGE_BN', None):
        out.append('MODEL.RANGE_BN: per-range statistics are per process; single GPU only')
    if model.get('NORM_SWAP', None):
        out.append('MODEL.NORM_SWAP: no BatchNorm is left to synchronize')
    return out
