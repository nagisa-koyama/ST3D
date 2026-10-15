"""Open a THIRD-PARTY checkpoint safely and re-save only its weights (experiments_md 20261015_01).

Released checkpoints are untrusted pickles, and `load_params_from_file` unpickles them fully. This script
  1. hashes the file (sha256);
  2. lists every global the pickle references, from its opcodes, without executing anything;
  3. loads it with `torch.load(weights_only=True)` and refuses (exit 3) rather than falling back to a full unpickle;
  4. writes {'model_state': tensors only, 'epoch', 'version', 'source_sha256'} to OUT, which our evaluator then reads.

Run it isolated, from outside the download directory, inside the container:
    singularity exec <sif> python3 -I analysis/sanitize_checkpoint.py <in.pth> <out.pth>
"""
import hashlib
import io
import json
import pickletools
import sys
import zipfile

import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pickle_globals(path):
    """Globals referenced by the pickle stream(s), read from opcodes only (pickletools never executes)."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            streams = [z.read(n) for n in z.namelist() if n.endswith('.pkl')]
    else:
        # legacy format: magic number, protocol, sys info and the data pickle are concatenated pickles
        raw = open(path, 'rb').read()
        streams, f = [], io.BytesIO(raw)
        while f.tell() < len(raw):
            start = f.tell()
            try:
                for _ in pickletools.genops(f):
                    pass
            except Exception:
                break
            streams.append(raw[start:f.tell()])
    found = set()
    for s in streams:
        strings = []
        for op, arg, _ in pickletools.genops(io.BytesIO(s)):
            if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'BINUNICODE8'):
                strings.append(arg)
            elif op.name == 'GLOBAL':
                found.add(arg.replace(' ', '.'))
            elif op.name == 'STACK_GLOBAL' and len(strings) >= 2:
                found.add(f'{strings[-2]}.{strings[-1]}')
    return sorted(found)


def main(src, out):
    info = {'source': src, 'source_sha256': sha256(src), 'pickle_globals': pickle_globals(src)}
    try:
        # Only numpy's scalar-reconstruction global and its dtype classes are allow-listed on top of torch's own list: the unpickler
        # stays closed to every other global (anything else still raises and exits 3). OpenPCDet-era checkpoints
        # store numpy scalars (e.g. the epoch) this way.
        import numpy as np
        try:
            from numpy._core.multiarray import scalar as np_scalar
        except ImportError:
            from numpy.core.multiarray import scalar as np_scalar
        dtype_classes = [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith('DType')] if hasattr(np, 'dtypes') else []
        with torch.serialization.safe_globals([np_scalar, np.dtype] + dtype_classes):
            ckpt = torch.load(src, map_location='cpu', weights_only=True)
    except Exception as e:  # never fall back to a full unpickle here
        info['weights_only_error'] = f'{type(e).__name__}: {e}'[:2000]
        print(json.dumps(info, indent=1))
        sys.exit(3)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state, meta = ckpt['model_state'], ckpt
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state, meta = ckpt['state_dict'], ckpt
    else:
        state, meta = ckpt, {}
    tensors = {k: v for k, v in state.items() if torch.is_tensor(v)}
    info.update({
        'top_level_keys': sorted(map(str, ckpt.keys())) if isinstance(ckpt, dict) else str(type(ckpt)),
        'n_tensors': len(tensors), 'n_non_tensor_entries': len(state) - len(tensors),
        'n_params': int(sum(v.numel() for v in tensors.values())),
        'epoch': meta.get('epoch') if isinstance(meta.get('epoch'), int) else None,
        'version': str(meta.get('version')) if meta.get('version') is not None else None,
        'key_prefixes': sorted({k.split('.')[0] for k in tensors}),
    })
    torch.save({'model_state': tensors, 'epoch': info['epoch'], 'version': info['version'],
                'source_sha256': info['source_sha256']}, out)
    info['out'], info['out_sha256'] = out, sha256(out)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
