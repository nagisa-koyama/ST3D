"""Config checks of experiments_md 20261003_05 §5 item 43 against what the jobs LOGGED at start (train.py / test.py print every
key), with the helpers of analysis/factorswap_pl_config_check.py. ANALYSIS tooling; reads configs and job logs only.

  43a: every accumulated checkpoint scored on nuScenes val builds the model the oracle config builds: MODEL / CLASS_NAMES /
       ONTOLOGY of cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml (today) equal those logged by 27468, 27804 (S2
       teacher, 15 / 10), 27183 / 27185 (Waymo depth, 30 / 20), 29354 (A4 teacher, 22 / 17), 25743 / its eval 25768 and 31818
       (the oracle seeds); each accumulated row is MeanVFE, without GBlobs and without pseudo-labels; 31818 logged 25743's
       config; the per-platform depths logged are those the own-input twins carry.
  43b: L'_K differs from 27468's and 27804's LOGGED configs only in the arm (per block: six HIST_DIST_* keys and the
       processor) and in DATA_CONFIG_TAR, which equals the oracle's (today and as 25743 / 31818 logged it).
Prints one line per check and "ALL ITEM-43 CONFIG CHECKS PASSED" or exits 1. Run inside the container from ST3D/tools:
    python analysis/item43_config_check.py
"""
import glob
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa: F401,E402
import factorswap_pl_config_check as C  # noqa: E402
import make_item43_configs as G  # noqa: E402

ACC = {27468: G.TEACHER, 27804: G.TEACHER, 27183: G.WAYMO_DEPTH, 27185: G.WAYMO_DEPTH, 29354: G.PROC_DEPTH}
DEPTH = {27468: (15, 10), 27804: (15, 10), 27183: (30, 20), 27185: (30, 20), 29354: (22, 17)}
ORACLE_JOBS = (25743, 25768, 31818)
MODEL_KEYS = ('MODEL', 'CLASS_NAMES', 'ONTOLOGY')


def log_text(job):
    path = (glob.glob('logs/error_%s_*.txt' % job) or [None])[0]
    assert path, job
    return open(path, errors='replace').read().replace('\r', '\n')


def main():
    oracle = C.flat(C.load(G.ORACLE))
    for job in sorted(ACC) + list(ORACLE_JOBS):
        C.expect('43a oracle config MODEL / CLASS_NAMES / ONTOLOGY vs job %d logged' % job,
                 C.diff(oracle, C.logged(job), MODEL_KEYS), [])
    for job, cfg in sorted(ACC.items()):
        lg, text = C.logged(job), log_text(job)
        C.check(lg['MODEL.VFE.NAME'] == 'MeanVFE' and not any('GBLOB' in k.upper() for k in lg)
                and 'cfg.SELF_TRAIN' not in text and 'USE_PSEUDO_LABEL: True' not in text,
                '43a job %d: MeanVFE, no GBlobs key, no SELF_TRAIN block, no pseudo-labels' % job)
        got = (lg['DATA_CONFIGS.NUSCENES_N008.MAX_SWEEPS'], lg['DATA_CONFIGS.NUSCENES_N015.MAX_SWEEPS'])
        own = dict(C.load(G.OWN[cfg]).DATA_CONFIG_TAR.MAX_SWEEPS_BY_VEHICLE)
        C.check(got == DEPTH[job] and (own['n008'], own['n015']) == got and
                all(lg['DATA_CONFIGS.%s.GT_BOXES_MOTION_COMPENSATION' % b] is True for b in G.BLOCK.values()),
                '43a job %d logged n008 / n015 %s sweeps with compensation; own-input twin %s' % (job, got, own))
        C.expect('43a job %d logged vs its config today (training blocks)' % job,
                 C.diff(C.flat(C.load(cfg)), lg, ('DATA_CONFIGS', 'MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'ONTOLOGY')), [])
    C.expect('43a 31818 logged vs 25743 logged (same config)', C.diff(C.logged(31818), C.logged(25743)), [])

    lp = C.flat(C.load(G.LPRIME))
    arm = ['DATA_CONFIGS.%s.%s' % (b, k) for b in G.BLOCK.values()
           for k in list(G.ARM) + ['HIST_DIST_REFERENCE_CFG', 'DATA_PROCESSOR']]
    for job in (27468, 27804):
        got = C.diff(lp, C.logged(job))
        C.expect("43b L'_K vs job %d logged, outside DATA_CONFIG_TAR" % job,
                 [p for p in got if not p.startswith('DATA_CONFIG_TAR.')], arm)
    C.expect("43b L'_K DATA_CONFIG_TAR vs the oracle config", C.diff(lp, oracle, ('DATA_CONFIG_TAR',)), [])
    for job in (25743, 31818):
        C.expect("43b L'_K DATA_CONFIG_TAR / MODEL vs job %d logged" % job,
                 C.diff(lp, C.logged(job), ('DATA_CONFIG_TAR',) + MODEL_KEYS), [])
    if C.FAILS:
        print('FAILED ITEM-43 CONFIG CHECKS (%d):' % len(C.FAILS))
        for f in C.FAILS:
            print('  ' + f)
        raise SystemExit(1)
    print('ALL ITEM-43 CONFIG CHECKS PASSED')


if __name__ == '__main__':
    main()
