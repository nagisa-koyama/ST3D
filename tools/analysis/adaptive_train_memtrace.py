"""Run adaptive_train.py unchanged, with crash and memory diagnostics.

Built to root-cause the da-ieee-access UADA3D segfault, which turned out to be a use-after-free in
torch 2.5.1's CUDA caching allocator (experiments_md/20260926_05). Usage, from tools/:

    [NODETRACE=<file>] [SEGV_BT=1] python analysis/adaptive_train_memtrace.py <adaptive_train.py args...>

Always on: around every forward, two lines on stdout -

    MEMTRACE pre it=<n> dom=<0|1> vox=<voxels> pts=<points> prev_peak=<GiB> ids=<frame ids>
    MEMTRACE it=<n> dom=<0|1> vox=... alloc=<GiB> peak=<GiB> reserved=<GiB> retries=<n> ooms=<n>

`prev_peak` on a source forward covers the previous iteration's backward, where the real
high-water mark lives. `retries`/`ooms` are the allocator's cumulative counters.

NODETRACE=<file>: <file> always holds the autograd node that started last, with its saved-tensor
shapes. After a native crash on autograd's device thread (no Python frame) it names the op.

SEGV_BT=1: native backtrace (library+offset per frame) on SIGSEGV/SIGBUS/SIGABRT/SIGILL. Needs
segv_bt/segv_bt.so, which is gitignored - build it in the container first (see segv_bt.c), then
name the frames with `python3 analysis/segv_bt/symbolize.py <file with the SEGV_BT block>`.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _init_path  # noqa: F401,E402  - live repo pcdet, not the image's stale copy

import torch  # noqa: E402
import pcdet.models as models  # noqa: E402

_orig_decorator = models.model_fn_decorator
_GiB = float(2 ** 30)
_calls = [0]


def _traced_decorator():
    model_func = _orig_decorator()

    def traced(model, batch_dict):
        vox = len(batch_dict['voxels']) if 'voxels' in batch_dict else -1
        pts = len(batch_dict['points']) if 'points' in batch_dict else -1
        ids = batch_dict.get('frame_id', [])
        dom = batch_dict.get('domain', -1)
        # Peak since the previous forward's end: for a source forward that window is the previous
        # iteration's backward + optimizer step, which is where the real high-water mark lives.
        print('MEMTRACE pre it=%d dom=%s vox=%d pts=%d prev_peak=%.2f ids=%s' % (
            _calls[0] // 2, dom, vox, pts, torch.cuda.max_memory_allocated() / _GiB, list(ids)),
            flush=True)
        torch.cuda.reset_peak_memory_stats()
        ret = model_func(model, batch_dict)
        torch.cuda.synchronize()
        st = torch.cuda.memory_stats()
        # num_alloc_retries counts cudaMalloc failures that sent the allocator into its
        # free-cached-blocks-and-retry path - the path release_available_cached_blocks lives on.
        print('MEMTRACE it=%d dom=%s vox=%d pts=%d alloc=%.2f peak=%.2f reserved=%.2f retries=%d ooms=%d' % (
            _calls[0] // 2, dom, vox, pts, torch.cuda.memory_allocated() / _GiB,
            torch.cuda.max_memory_allocated() / _GiB, torch.cuda.memory_reserved() / _GiB,
            st.get('num_alloc_retries', -1), st.get('num_ooms', -1)),
            flush=True)
        torch.cuda.reset_peak_memory_stats()
        _calls[0] += 1
        return ret

    return traced


models.model_fn_decorator = _traced_decorator

# SEGV_BT=1: replace faulthandler's signal handler with a native one that prints library+offset for
# every frame (segv_bt/segv_bt.c). Installed after torch/CUDA libs are loaded so dladdr can name them.
if os.environ.get('SEGV_BT'):
    import ctypes
    _so = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segv_bt', 'segv_bt.so')
    if not os.path.exists(_so):
        sys.exit('SEGV_BT=1 but %s is not built (it is gitignored). Build it inside the container:\n'
                 '  gcc -O0 -g -fPIC -shared -o %s %s -ldl' % (_so, _so, _so[:-3] + '.c'))
    ctypes.CDLL(_so).install()

# --- Last-backward-node recorder --------------------------------------------------------------
# A segfault on autograd's device thread has no Python frame, so faulthandler cannot say which op
# died. With NODETRACE=<path> set, every grad_fn reachable from the loss gets a pre-hook that
# overwrites <path> with "<iteration> <node index>/<node count> <node name>" just before the node
# runs; after a crash the file holds the node that was executing. One pwrite per node.
_nodetrace = os.environ.get('NODETRACE')
if _nodetrace:
    _fd = os.open(_nodetrace, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    _orig_backward = torch.Tensor.backward
    _bw_calls = [0]

    def _traced_backward(self, *args, **kwargs):
        nodes, seen, stack = [], set(), [self.grad_fn]
        while stack:
            n = stack.pop()
            if n is None or id(n) in seen:
                continue
            seen.add(id(n))
            nodes.append(n)
            stack.extend(nf for nf, _ in n.next_functions)
        total = len(nodes)
        it = _bw_calls[0]

        def make_hook(idx, name):
            rec = ('%d %d/%d %s' % (it, idx, total, name)).ljust(200).encode()

            def hook(grad_outputs):
                os.pwrite(_fd, rec, 0)
            return hook

        def describe(n):
            desc = n.name()
            for attr in ('_saved_input', '_saved_weight', '_saved_self'):
                t = getattr(n, attr, None)
                if isinstance(t, torch.Tensor):
                    desc += ' %s=%s' % (attr[7:], tuple(t.shape))
            return desc

        for i, n in enumerate(nodes):
            try:
                n.register_prehook(make_hook(i, describe(n)))
            except Exception:
                pass
        _bw_calls[0] += 1
        ret = _orig_backward(self, *args, **kwargs)
        os.pwrite(_fd, ('%d done' % it).ljust(200).encode(), 0)
        return ret

    torch.Tensor.backward = _traced_backward

if __name__ == '__main__':
    tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(tools_dir)
    sys.argv = [os.path.join(tools_dir, 'adaptive_train.py')] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')
