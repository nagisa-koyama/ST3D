"""Standalone probe of the conv whose backward crashed in the da-ieee-access UADA3D rows.

It does NOT reproduce that crash, and cannot. Read experiments_md/20260926_05 before reusing it.
The crash is a use-after-free in torch 2.5.1's CUDA caching allocator
(release_available_cached_blocks), reached while allocating this conv's cuDNN weight-grad
workspace. It faults only when the freed std::set node has been consolidated by glibc between the
free and the read, which needs other threads allocating in the same heap arena - a single-threaded
script never supplies that. What this script did establish: the conv itself (input (6, 256, 374,
374), weight (128, 256, 3, 3) - Discriminator2's second conv) is sound under memory pressure
(clean `CUDA out of memory` at every squeeze level) and over 6,000 repeated backward calls.

It runs that one conv, forward + backward, while a filler tensor leaves only `--free_gib` of device
memory free, optionally with `--cached` GiB of reserved-but-free blocks left in the caching
allocator. Each case runs in a child process: a segfault shows up as exit code -11, a clean CUDA OOM
as exit code 3.

    python analysis/cudnn_conv_backward_repro.py                 # sweep
    python analysis/cudnn_conv_backward_repro.py --case 2.0 1    # one case: free GiB, deterministic
"""
import argparse
import subprocess
import sys

SHAPE_IN = (6, 256, 374, 374)
SHAPE_W = (128, 256, 3, 3)


def run_case(free_gib, deterministic, iters, batch, cached_gib=0.0, chunk_mib=400):
    import torch
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False
    dev = torch.device('cuda')
    shape_in = (batch,) + SHAPE_IN[1:]
    x = torch.randn(shape_in, device=dev, requires_grad=True)
    conv = torch.nn.Conv2d(SHAPE_W[1], SHAPE_W[0], SHAPE_W[2], bias=False).to(dev)
    torch.cuda.synchronize()
    filler = None
    if cached_gib > 0:
        # Leave `cached_gib` of RESERVED-BUT-FREE memory in the caching allocator, in blocks too
        # small to hold the 820 MiB input grad, as the real run has (reserved 46.5, allocated 39-42).
        # A request that misses the cache then has to go through cudaMalloc -> fail ->
        # release cached blocks -> retry, the path the pure-filler cases never exercise.
        n = int(cached_gib * 1024 / chunk_mib)
        chunks = [torch.empty(chunk_mib * 2 ** 20, dtype=torch.uint8, device=dev) for _ in range(n)]
        del chunks
    if free_gib >= 0:
        free, total = torch.cuda.mem_get_info()
        # Leave the output and its grad (~0.43 GiB each at batch 6) out of the budget on purpose:
        # the point is to squeeze the WORKSPACE cuDNN asks for, as the real run does.
        take = free - int(free_gib * 2 ** 30)
        if take > 0:
            filler = torch.empty(take, dtype=torch.uint8, device=dev)
    free_now = torch.cuda.mem_get_info()[0] / 2 ** 30
    print('case free_target=%.2f free_now=%.2f cached=%.2f det=%d batch=%d cudnn=%s' % (
        free_gib, free_now, (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 2 ** 30,
        deterministic, batch, torch.backends.cudnn.version()), flush=True)
    for i in range(iters):
        y = conv(x)
        y.sum().backward()
        torch.cuda.synchronize()
        x.grad = None
        conv.weight.grad = None
    print('ok after %d iterations, peak %.2f GiB' % (iters, torch.cuda.max_memory_allocated() / 2 ** 30),
          flush=True)
    del filler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', nargs=2, type=float, default=None)
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--cached', type=float, default=0.0)
    ap.add_argument('--chunk_mib', type=int, default=400)
    ap.add_argument('--dets', type=int, nargs='*', default=[1, 0])
    ap.add_argument('--free', type=float, nargs='*', default=[-1, 8, 4, 2, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1])
    args = ap.parse_args()
    if args.case is not None:
        try:
            run_case(args.case[0], int(args.case[1]), args.iters, args.batch, args.cached, args.chunk_mib)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print('clean OOM: %s' % str(e).splitlines()[0], flush=True)
                sys.exit(3)
            raise
        return
    for det in args.dets:
        for free in args.free:
            r = subprocess.run([sys.executable, '-X', 'faulthandler', __file__, '--case', str(free), str(det),
                                '--iters', str(args.iters), '--batch', str(args.batch),
                                '--cached', str(args.cached), '--chunk_mib', str(args.chunk_mib)],
                               capture_output=True, text=True)
            tail = (r.stdout + r.stderr).strip().splitlines()
            msg = [l for l in tail if l.startswith(('case', 'ok', 'clean', 'Fatal', 'RuntimeError'))]
            print('RESULT det=%d free=%s cached=%s batch=%d exit=%d | %s' % (
                det, free, args.cached, args.batch, r.returncode, ' | '.join(msg[-3:])), flush=True)


if __name__ == '__main__':
    main()
