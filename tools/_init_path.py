import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

# Must run before anything initialises CUDA - which is why it lives in the module every entry point
# imports first. Drops max_split_size_mb from the image's PYTORCH_CUDA_ALLOC_CONF on torch < 2.9,
# where it arms an allocator use-after-free (see _alloc_conf_guard.py, experiments_md/20260926_05).
import _alloc_conf_guard  # noqa: E402
_alloc_conf_guard.apply()
