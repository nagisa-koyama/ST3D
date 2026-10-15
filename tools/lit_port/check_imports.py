"""Import check of the LiT container (run with the shim on the path), experiments_md 20261015_02."""
import sys
sys.path[:0] = ['/home/koyama/code/ST3D/tools/lit_port/shims', '/home/koyama/code/lit']
import types
_g = types.ModuleType('lit.raycast_engine_gpu'); _g.RaycastEngineGPU = object; sys.modules['lit.raycast_engine_gpu'] = _g
import torch, torch_scatter, nksr, open3d, camtools, pycg.vis  # noqa
print('torch', torch.__version__, torch.version.cuda, 'cuda available', torch.cuda.is_available())
print('open3d', open3d.__version__, 'nksr', nksr.__file__)
from lit.ext import lit_ext
print('lit_ext', lit_ext.split_mesh_by_cc)
from lit.recon_utils import split_mesh_by_cc, bbox_to_corners
from lit.raycast_engine_cpu import RaycastEngineCPU
from lit.fg_reconstructor import FGReconstructor
from lit.bg_reconstructor import BGReconstructor
import lit_tools.lit_02_recon_bg, lit_tools.lit_03_recon_fg, lit_tools.lit_04_sim
import numpy as np
from pcdet.utils.box_utils import box_to_corners_3d
print(np.round(box_to_corners_3d(np.array([1, 2, 3, 4, 2, 1.5, 0.5, 1.0])), 2)[:2])
print('IMPORTS OK')
