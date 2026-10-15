"""Stage driver for the legal LiT row (experiments_md 20261015_02). Runs INSIDE the LiT container with `python -I`; it
calls LiT's own stage functions (lit_02 `recon_bg`, lit_03 `recon_fg`, lit_04 `run_sim`) on an explicit list of scenes, with
every output under --root on NFS. LiT's stage scripts cannot be used as they are: lit_02 refuses to run unless ALL 1,000
scenes of its list are extracted, lit_03 drops into ipdb on any exception, and all of them write under the clone.

    python -I lit_port/lit_run_scenes.py --lit /home/koyama/code/lit --root <dir> --scenes a,b --stages bg,fg,sim \
        [--cpu] [--dst nuscenes] [--max_frames N]

Scene pickles come from `lit_port/lit01_waymo_from_processed.py` into <root>/scene. Defaults are the legal row's: ray-drop
off (LiT's simulator has none), `enable_scaling=False`, foreground from the packaged ShapeNet DeepSDF, no MVDeepSDF.
"""
import argparse
import importlib
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lit', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--stages', default='bg,fg,sim')
    ap.add_argument('--dst', default='nuscenes')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--skip_existing', action='store_true')
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here / 'shims'), a.lit]            # the shim pcdet must win over LiT's own fork

    from lit.path_utils import LitPaths
    import types  # noqa: E402
    _g = types.ModuleType('lit.raycast_engine_gpu')   # the OptiX engine needs cupy / pyoptix / pynvrtc: the legal row ray casts on CPU
    class _NoGPU:                                      # noqa: E301
        def __init__(self, *a, **k):
            raise RuntimeError('the OptiX ray caster is not installed; use the CPU engine (--cpu)')
    _g.RaycastEngineGPU = _NoGPU
    sys.modules['lit.raycast_engine_gpu'] = _g

    root = Path(a.root)
    names = a.scenes.split(',')
    paths = LitPaths(data_version='v0', data_domain='waymo', scene_names=names, lit_data_root=root,
                     scene_dir=root / 'scene', fg_dir=root / 'fg', bg_dir=root / 'bg',
                     sim_waymo_dir=root / 'sim_waymo', sim_nuscenes_dir=root / 'sim_nuscenes',
                     sim_kitti_dir=root / 'sim_kitti')
    for d in (paths.fg_dir, paths.bg_dir):
        d.mkdir(parents=True, exist_ok=True)
    stages = a.stages.split(',')
    t0 = time.time()
    if 'bg' in stages:
        m = importlib.import_module('lit_tools.lit_02_recon_bg')
        for n in names:
            t = time.time(); m.recon_bg(scene_name=n, lit_paths=paths, skip_existing=a.skip_existing)
            print(f'[stage bg] {n} {time.time() - t:.1f} s', flush=True)
    if 'fg' in stages:
        m = importlib.import_module('lit_tools.lit_03_recon_fg')
        sys.excepthook = sys.__excepthook__                    # lit_03 installs an ipdb hook: a batch job must not hang
        fgr = m.FGReconstructor(use_deepsdf=True)
        for n in names:
            t = time.time(); m.recon_fg(n, fgr=fgr, lit_paths=paths, skip_existing=a.skip_existing, verbose=False)
            print(f'[stage fg] {n} {time.time() - t:.1f} s', flush=True)
    if 'sim' in stages:
        m = importlib.import_module('lit_tools.lit_04_sim')
        # LiT's CPU engine returns float64 points / zero incident angles, and BaseContainer asserts float32 (the authors ray
        # cast on the GPU): cast at construction. Values are unchanged; float64 -> float32 only (job 28676).
        import numpy as np
        _SimFrame = m.SimFrame

        def _SimFrame32(**kw):
            return _SimFrame(**{k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v)
                                for k, v in kw.items()})
        m.SimFrame = _SimFrame32
        if a.cpu:
            from lit.raycast_engine_cpu import RaycastEngineCPU
            eng = RaycastEngineCPU()
        else:
            raise SystemExit('pass --cpu: the OptiX engine is not installed')
        for n in names:
            t = time.time()
            m.run_sim(data_version='v0', src_style='waymo', dst_style=a.dst, lit_paths=paths, scene_name=n,
                      enable_scaling=False, raycast_engine=eng, skip_existing=a.skip_existing)
            print(f'[stage sim] {n} {time.time() - t:.1f} s', flush=True)
    print(f'done in {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
