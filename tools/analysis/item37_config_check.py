"""Recursive-diff check of the configs of experiments_md 20261003_05 §5 item 37, with the helpers of
analysis/factorswap_pl_config_check.py (item 34). ANALYSIS tooling; reads configs and job logs only.
  37a: the L-on-spin eval twin differs from L's config (31539 / 31550) only in DATA_CONFIG_TAR, which equals the PandaSet
       oracle's; MODEL / CLASS_NAMES / ONTOLOGY equal what jobs 31539, 31550 and the oracle 26389 logged.
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


def main():
    check_37a()
    if C.FAILS:
        print('FAILED ITEM-37 CONFIG CHECKS (%d):' % len(C.FAILS))
        for f in C.FAILS:
            print('  ' + f)
        raise SystemExit(1)
    print('ALL ITEM-37 CONFIG CHECKS PASSED')


if __name__ == '__main__':
    main()
