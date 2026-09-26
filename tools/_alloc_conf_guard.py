"""Drop `max_split_size_mb` from PYTORCH_CUDA_ALLOC_CONF on torch < 2.9, before CUDA starts.

Called from _init_path, the first import of every entry point. Why it exists
(experiments_md/20260926_05): torch 2.5.1's CUDA caching allocator has a use-after-free in
`release_available_cached_blocks` -

    release_block(*cur, context);      // erases the std::set node and deletes the Block
    totalReleased += (*cur)->size;     // then reads through the erased iterator

- fixed upstream in v2.9.0 by swapping the two lines. The function returns at its first line
unless `max_split_size_mb` is configured, and the container image exports
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" for every job, which arms it. Any job that then
has a cudaMalloc fail (i.e. runs near the memory wall) can die with a segfault that has no Python
frame, at a random iteration. It killed both da-ieee-access UADA3D rows.

Timing: torch reads the setting when the CUDA allocator is constructed (first CUDA use), not at
`import torch`, so editing the environment works any time before that. If CUDA is somehow already
initialised, the setting is re-parsed at runtime instead, which also takes effect: the allocator
consults max_split_size() live on every release.
"""
import os
import re
import sys

ENV = 'PYTORCH_CUDA_ALLOC_CONF'
KEY = 'max_split_size_mb'
FIXED_IN = (2, 9)


def _torch_version():
    """(major, minor) of the installed torch, read WITHOUT importing it, or None if unknown."""
    try:
        from importlib.metadata import version
        parts = version('torch').split('+')[0].split('.')
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def strip_max_split_size(conf):
    """`conf` with every `max_split_size_mb:<n>` option removed, other options kept in order.

    Options are split the way torch 2.5's own backend parser splits them (`[\\s,]+`), then each
    is a `key:value` pair.
    """
    kept = [opt for opt in re.split(r'[\s,]+', conf)
            if opt and opt.split(':', 1)[0] != KEY]
    return ','.join(kept)


def apply(environ=None, torch_version=None, stream=None):
    """Remove the option from `environ` (default os.environ). Returns the dropped setting or None."""
    environ = os.environ if environ is None else environ
    conf = environ.get(ENV)
    if not conf or KEY not in conf:
        return None
    torch_version = _torch_version() if torch_version is None else torch_version
    # Unknown version: strip anyway. Losing a fragmentation tweak costs far less than the crash.
    if torch_version is not None and torch_version >= FIXED_IN:
        return None
    new = strip_max_split_size(conf)
    if new:
        environ[ENV] = new
    else:
        del environ[ENV]
    if environ is os.environ and 'torch' in sys.modules:
        torch = sys.modules['torch']
        if torch.cuda.is_initialized():
            torch.cuda.memory._set_allocator_settings(new)
    print('[_alloc_conf_guard] %s: dropped %s=%r (torch %s allocator use-after-free, '
          'experiments_md/20260926_05); now %s' % (
              ENV, ENV, conf, '.'.join(map(str, torch_version)) if torch_version else '?',
              repr(new) if new else 'unset'),
          file=stream or sys.stderr, flush=True)
    return conf
