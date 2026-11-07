"""Recursive-diff check of the configs of experiments_md 20261003_05 §5 item 37, with the helpers of
analysis/factorswap_pl_config_check.py (item 34). ANALYSIS tooling; reads configs and job logs only.
  37a: the L-on-spin eval twin differs from L's config (31539 / 31550) only in DATA_CONFIG_TAR, which equals the PandaSet
       oracle's; MODEL / CLASS_NAMES / ONTOLOGY equal what jobs 31539, 31550 and the oracle 26389 logged.
  37b: L_L40 differs from the C1-analog config (and 30313's logged config) only in HIST_DIST_REFERENCE_CFG /
       HIST_DIST_IN_GRID; its Lyft-40 val eval twin differs from it only in DATA_CONFIG_TAR, which equals O_L40's.
Prints one line per check and "ALL ITEM-37 CONFIG CHECKS PASSED" or exits 1. Run inside the container from ST3D/tools:
    python analysis/item37_config_check.py
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa: F401,E402
import factorswap_pl_config_check as C  # noqa: E402
import make_item37_configs as G  # noqa: E402


def check_37a():
    tw = C.flat(C.load(G.TWIN_37A))
    src = C.flat(C.load(G.L_PANDASET))
    spin = C.flat(C.load(C.D + 'centerpoint-pandaset-spin2spin.yaml'))
    got = C.diff(tw, src)
    C.check(len(got) > 0 and all(p.startswith('DATA_CONFIG_TAR.') for p in got),
            '37a L-on-spin vs L config: only DATA_CONFIG_TAR differs (%d paths)' % len(got))
    C.expect('37a L-on-spin DATA_CONFIG_TAR vs the PandaSet oracle\'s', C.diff(tw, spin, ('DATA_CONFIG_TAR',)), [])
    for j in (31539, 31550):
        C.expect('37a L-on-spin MODEL / CLASS_NAMES / ONTOLOGY vs job %d logged' % j,
                 C.diff(tw, C.logged(j), ('MODEL', 'CLASS_NAMES', 'ONTOLOGY')), [])
    C.expect('37a L-on-spin MODEL vs the PandaSet oracle 26389 logged', C.diff(tw, C.logged(26389), ('MODEL',)), [])
    p_tw = C.flat(C.load(C.A + 'factorswap-pandaset/centerpoint-factorswap-pandaset-evalonly-p-on-spin.yaml'))
    C.expect('37a L-on-spin vs the P-on-spin twin on MODEL / CLASS_NAMES / ONTOLOGY / DATA_CONFIG_TAR',
             C.diff(tw, p_tw, ('MODEL', 'CLASS_NAMES', 'ONTOLOGY', 'DATA_CONFIG_TAR')), [])


def check_37b():
    lc = C.flat(C.load(G.L_LYFT40))
    c1 = C.flat(C.load(C.D + 'centerpoint-global-recount-lyft40-lyft2nuscenes.yaml'))
    C.expect('37b L_L40 vs C1-analog (30313 config)', C.diff(lc, c1),
             ['DATA_CONFIGS.LYFT_40BEAM.HIST_DIST_REFERENCE_CFG', 'DATA_CONFIGS.LYFT_40BEAM.HIST_DIST_IN_GRID'])
    C.expect('37b L_L40 vs 30313 logged (all logged blocks)', C.diff(lc, C.logged(30313)),
             ['DATA_CONFIGS.LYFT_40BEAM.HIST_DIST_REFERENCE_CFG', 'DATA_CONFIGS.LYFT_40BEAM.HIST_DIST_IN_GRID'])
    tw = C.flat(C.load(G.TWIN_37B))
    got = C.diff(tw, lc)
    C.check(len(got) > 0 and all(p.startswith('DATA_CONFIG_TAR.') for p in got),
            '37b L-on-lyft40 vs L_L40 config: only DATA_CONFIG_TAR differs (%d paths)' % len(got))
    l40 = C.flat(C.load(C.A + 'factorswap-lyft40/centerpoint-factorswap-lyft40-oracle.yaml'))
    C.expect('37b L-on-lyft40 DATA_CONFIG_TAR vs O_L40\'s', C.diff(tw, l40, ('DATA_CONFIG_TAR',)), [])
    p_tw = C.flat(C.load(C.A + 'factorswap-lyft40/centerpoint-factorswap-lyft40-evalonly-p-on-lyft40.yaml'))
    C.expect('37b L-on-lyft40 vs the P-on-lyft40 twin on MODEL / CLASS_NAMES / ONTOLOGY / DATA_CONFIG_TAR',
             C.diff(tw, p_tw, ('MODEL', 'CLASS_NAMES', 'ONTOLOGY', 'DATA_CONFIG_TAR')), [])


def main():
    check_37a()
    check_37b()
    if C.FAILS:
        print('FAILED ITEM-37 CONFIG CHECKS (%d):' % len(C.FAILS))
        for f in C.FAILS:
            print('  ' + f)
        raise SystemExit(1)
    print('ALL ITEM-37 CONFIG CHECKS PASSED')


if __name__ == '__main__':
    main()
