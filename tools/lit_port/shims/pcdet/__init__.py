"""Minimal stand-in for LiT's OpenPCDet fork, so the translation stage imports without its compiled CUDA ops
(experiments_md 20261015_02). LiT's translation code imports exactly two things from `pcdet`: `box_to_corners_3d`
(`lit/recon_utils.py`) and `furthest_point_sample` (`lit/mvfg_utils.py`, used only when `use_mvdeepsdf=True`, which the
legal row does not use). The first is re-implemented from LiT's `pcdet/utils/box_utils.py` + `common_utils.py`; the second
raises if called."""
