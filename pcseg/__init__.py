"""pcseg: LiDAR semantic segmentation for the next paper's factor-wise oracle analysis (experiments_md
episodes/20261116_01, integration plan B).

The boundary with ST3D's detection code, which every module here keeps:
  - pcseg CALLS reader-level and operator functions of pcdet (pandaset geometry, pandaset_rings, reflection_drop,
    osens2_utils, rerender_utils, loader_factor_swap) and never changes them;
  - pcseg NEVER calls DatasetTemplate.prepare_data, the augmentor, the data processor, the database sampler, a detector
    or the detection evaluation, and adds no key to any detection config;
  - every step that drops, reorders or creates points returns an INDEX (keep mask, index array, inverse map), and the
    per-point labels are moved by that index in the same function.

The ignore label is IGNORE = -1 at every stage (Pointcept's convention). It is never used as an index.
"""
IGNORE = -1
