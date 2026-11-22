"""pcocc: semantic occupancy GTs for the occupancy chapter of the next paper (experiments_md
episodes/20261121_02, design; prior art 20261116_02).

Three GTs per PandaSet frame, built from Pandar64 points only (G64), PandarGT points only (GGT) and both (GU), with
every other convention identical, so that a difference between them is the sensor's and not the convention's.

Boundaries (memory/repo/pcseg_shared_branch_rules.md):
  - pcocc imports pcseg READ-ONLY (reader, class map, cone) and never edits it; tests/test_occ_*.py pins what it uses;
  - pcocc reads two data files written by the semseg session and pins their sha256: the PandarGT repeated-scan census
    and the sensor fits (ray origins);
  - GT data is analysis / evaluation data: target labels are read here by design (uda_legality_rule.md, the analysis
    exception). No method may read these files.

Encoding of a GT voxel grid (uint8, shape Grid.shape):
  EMPTY = 0, class c (pcseg index 0..16) = c + 1, IGNORE_VOXEL = 255 (occupied, but only by ignore points).
  A separate boolean UNKNOWN mask marks voxels that no ray traversed and no point occupies; their label byte is EMPTY,
  so every consumer must apply the mask (use `grid.scoring_labels`, which folds UNKNOWN into IGNORE_VOXEL).
"""
EMPTY = 0
IGNORE_VOXEL = 255
