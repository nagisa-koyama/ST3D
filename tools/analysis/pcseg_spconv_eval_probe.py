"""Probe for the pcseg smoke 33127 crash (experiments_md episodes/20261116_01 §8.2): the first validation pass died in
spconv's implicit-GEMM tuner ("can't find suitable algorithm for 0") at the first down-sampling conv, after a clean
training epoch. Training runs spconv under autocast through its autograd Functions, which cast inputs to fp16; in eval
mode spconv calls ops.implicit_gemm directly, with no cast. This script runs one real validation batch through the
SpUNet in several modes and records, per mode, the feature / weight dtypes entering the stem and the first down conv,
and whether the forward succeeds.

    python analysis/pcseg_spconv_eval_probe.py --cfg cfgs/pcseg/spunet-pandaset64-smoke.yaml --out <json>
"""
import argparse
import copy
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _init_path  # noqa: F401,E402  (tools/: sys.path and the allocator guard, before torch touches CUDA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import torch
    import train_seg
    from pcseg import engine
    from pcseg.class_map import build_class_map
    from pcseg.pandaset_seg import collate

    cfg = train_seg.load_cfg(args.cfg)
    names = build_class_map(cfg.DATA.CLASS_MAP).names
    sets = train_seg.build_datasets(cfg)
    dev = torch.device('cuda')
    torch.manual_seed(0)
    model = train_seg.build_model(cfg, len(names)).to(dev)
    batch_all = collate([sets['val'][i] for i in range(args.frames)])
    batch_one = collate([sets['val'][0]])

    seen = []

    def hook(name):
        def f(mod, inp, out):
            w = getattr(mod, 'weight', None)
            seen.append({'module': name, 'in': str(inp[0].features.dtype), 'out': str(out.features.dtype),
                         'weight': None if w is None else str(w.dtype), 'n_in': int(inp[0].features.shape[0])})
        return f

    handles = [model.conv_input[0].register_forward_hook(hook('stem')),
               model.down[0][0].register_forward_hook(hook('down0'))]

    def run(tag, mdl, batch, train_mode, amp):
        seen.clear()
        mdl.train(train_mode)
        res = {'mode': tag, 'train_mode': train_mode, 'amp': amp, 'voxels': int(batch['feat'].shape[0]),
               'feat_dtype': str(batch['feat'].dtype)}
        try:
            with torch.no_grad(), torch.autocast('cuda', enabled=amp):
                out = mdl(engine.to_device(batch, dev))
            torch.cuda.synchronize()
            res.update(ok=True, out_dtype=str(out.dtype), out_shape=list(out.shape),
                       finite=bool(torch.isfinite(out.float()).all()))
        except Exception as e:  # noqa: BLE001  (a probe records every failure)
            res.update(ok=False, error=repr(e)[:400], tb=traceback.format_exc()[-1200:])
        res['hooks'] = list(seen)
        print(json.dumps({k: v for k, v in res.items() if k != 'tb'}), flush=True)
        return res

    results = [
        run('eval_fp32_%dframes' % args.frames, model, batch_all, False, False),
        run('eval_amp_1frame', model, batch_one, False, True),
        run('eval_amp_%dframes' % args.frames, model, batch_all, False, True),
        run('train_mode_amp_%dframes' % args.frames, copy.deepcopy(model), batch_all, True, True),
    ]
    for h in handles:
        h.remove()
    info = {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
            'capability': list(torch.cuda.get_device_capability(0))}
    try:
        import spconv
        info['spconv'] = spconv.__version__
    except Exception:  # noqa: BLE001
        pass
    with open(args.out, 'w') as f:
        json.dump({'info': info, 'results': results}, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
