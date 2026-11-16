"""Raw dataset class id -> model class index, with IGNORE (-1) for classes a map leaves out.

A map is a lookup table that RAISES on a raw id it does not know (negative, above the table, or never declared), so a
label the dataset adds or a corrupted file cannot silently become a class (bug_prevention_checklist rule 2: numpy wraps
negative indices).
"""
import numpy as np

from . import IGNORE

_UNDECLARED = -2


class ClassMap:
    def __init__(self, names, raw_to_name):
        """names: the model classes, in index order. raw_to_name: {raw id: model class name, or None for IGNORE}; every
        raw id the dataset can produce must be declared."""
        self.names = list(names)
        assert len(set(self.names)) == len(self.names), 'duplicate model class names'
        index = {n: i for i, n in enumerate(self.names)}
        raw_ids = [int(r) for r in raw_to_name]
        assert min(raw_ids) >= 0, 'raw ids must be non-negative'
        self.lut = np.full(max(raw_ids) + 1, _UNDECLARED, dtype=np.int64)
        for raw, name in raw_to_name.items():
            if name is None:
                self.lut[int(raw)] = IGNORE
            else:
                assert name in index, 'raw id %s maps to %r, which is not a model class' % (raw, name)
                self.lut[int(raw)] = index[name]
        unused = set(self.names) - {n for n in raw_to_name.values() if n is not None}
        assert not unused, 'model classes no raw id maps to: %s' % sorted(unused)

    @property
    def num_classes(self):
        return len(self.names)

    def __call__(self, raw):
        raw = np.asarray(raw)
        if raw.size == 0:
            return np.zeros(raw.shape, dtype=np.int64)
        if not np.issubdtype(raw.dtype, np.integer):
            assert np.all(raw == np.round(raw)), 'non-integer raw class ids'
            raw = raw.astype(np.int64)
        lo, hi = int(raw.min()), int(raw.max())
        if lo < 0 or hi >= len(self.lut):
            raise ValueError('raw class id outside the declared range [0, %d]: min %d, max %d'
                             % (len(self.lut) - 1, lo, hi))
        out = self.lut[raw]
        if np.any(out == _UNDECLARED):
            raise ValueError('undeclared raw class ids: %s' % np.unique(raw[out == _UNDECLARED]).tolist())
        return out


# --- PandaSet maps taken from published work ------------------------------------------------------------------------
# O2 of episodes/20261116_01 (the user, 2026-11-16: 「先行研究のマッピングを参考とする」). Raw ids are PandaSet's
# (dataset/<seq>/annotations/semseg/classes.json 1..42, plus the stored 0, which classes.json omits: 19 of 22.06 M points
# in 5 of 130 sampled frames, 2026-11-16). IGNORE replaces each source's own ignore value. Class names are the source's,
# lower-cased with '_' for spaces and punctuation; the index order is the source's. One map serves both sensors.
#
# ScaLR 17 (Puy et al., CVPR 2024; github.com/valeoai/ScaLR datasets/pandaset_for_scalr.py MAPPING_CLASS and CLASS_NAME,
# file at 0b310a8 2024-02-17, read at main 53c474d): the only published nuScenes-style PandaSet map, applied by its
# authors to Pandar64 and PandarGT alike. Their train id is MAPPING_CLASS - 1; their 0 ('noise') becomes 255 = ignore.
# The P1 training map.
PANDASET_SCALR17_NAMES = [
    'road', 'traffic_sign', 'barrier', 'pedestrian', 'vegetation', 'road_marking', 'sidewalk', 'manmade', 'traffic_cone',
    'car', 'motorcycle', 'truck', 'bus', 'bicycle', 'other_vehicle', 'ground', 'driveway',
]
PANDASET_SCALR17_RAW_TO_NAME = {
    0: None,                                                  # stored 0 ('Noise' in ScaLR)
    1: None, 2: None, 3: None, 4: None,                       # Smoke, Exhaust, Spray or rain, Reflection
    5: 'vegetation',                                          # Vegetation
    6: 'ground',                                              # Ground
    7: 'road',                                                # Road
    8: 'road_marking', 9: 'road_marking', 10: 'road_marking',  # Lane Line / Stop Line / Other Road Marking
    11: 'sidewalk',                                           # Sidewalk
    12: 'driveway',                                           # Driveway
    13: 'car',                                                # Car
    14: 'truck', 15: 'truck', 16: 'truck',                    # Pickup Truck, Medium-sized Truck, Semi-truck
    17: 'other_vehicle',                                      # Towed Object
    18: 'motorcycle',                                         # Motorcycle
    19: 'other_vehicle', 20: 'other_vehicle', 21: 'other_vehicle',  # Construction Vehicle, Uncommon, Pedicab
    22: None,                                                 # Emergency Vehicle
    23: 'bus',                                                # Bus
    24: None, 25: None,                                       # Personal Mobility Device, Motorized Scooter
    26: 'bicycle',                                            # Bicycle
    27: 'other_vehicle', 28: 'other_vehicle', 29: 'other_vehicle',  # Train, Trolley, Tram / Subway
    30: 'pedestrian', 31: 'pedestrian',                       # Pedestrian, Pedestrian with Object
    32: None, 33: None,                                       # Animals - Bird, Animals - Other
    34: 'traffic_cone',                                       # Pylons
    35: 'barrier',                                            # Road Barriers
    36: 'traffic_sign',                                       # Signs
    37: 'traffic_cone',                                       # Cones
    38: 'traffic_sign',                                       # Construction Signs
    39: 'barrier',                                            # Temporary Construction Barriers
    40: None,                                                 # Rolling Containers
    41: 'manmade', 42: 'manmade',                             # Building, Other Static Object
}

# 3DLabelProp 8 (Sanchez et al., ICCV 2023; github.com/JulesSanchez/3DLabelProp mapping/sk.py "pandaset"
# target_to_common, identical in mapping/ns.py, files at 25d4636 2023-07-11, read at 202127f; applied by its authors to
# PandaFF and Panda64 alike, and reused by TTYD, ECCV 2024): the coarse common set of the cross-dataset generalisation
# literature. For cross-dataset rows; a coarse map removes most of the convention term (20261027_04).
PANDASET_3DLP8_NAMES = [
    'two_wheeled', 'pedestrian', 'driveable_ground', 'sidewalk', 'other_ground', 'manmade', 'vegetation', 'four_wheeled',
]
PANDASET_3DLP8_RAW_TO_NAME = {
    **{r: None for r in (0, 1, 2, 3, 4, 27, 28, 29, 32, 33)},   # noise / weather, rail vehicles, animals
    **{5: 'vegetation', 6: 'other_ground', 11: 'sidewalk', 30: 'pedestrian', 31: 'pedestrian'},
    **{r: 'driveable_ground' for r in (7, 8, 9, 10, 12)},     # road, the three markings, driveway
    **{r: 'four_wheeled' for r in (13, 14, 15, 16, 17, 19, 20, 22, 23)},
    **{r: 'two_wheeled' for r in (18, 21, 24, 25, 26)},        # motorcycle, pedicab, PMD, scooter, bicycle
    **{r: 'manmade' for r in range(34, 43)},                   # pylons .. other static object
}

# --- nuScenes-lidarseg 16 (the reproduction gate G0, episodes/20261116_01 §5) -----------------------------------------
# The official lidarseg challenge classes: nuscenes-devkit python-sdk/nuscenes/eval/lidarseg/utils.py LidarsegClassMapper
# (get_fine2coarse, get_coarse2idx; file at c44366d 2021-08-03), in Pointcept's index order (pointcept/datasets/nuscenes.py
# get_learning_map, Pointcept at 77f87ab), which is the devkit's coarse index - 1. Raw ids are the `index` of the
# lidarseg tree's v1.0-trainval/category.json (0..31). tests/test_pcseg_nuscenes.py checks both sources.
NUSCENES_LIDARSEG16_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer',
    'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
]
NUSCENES_LIDARSEG16_RAW_TO_NAME = {
    0: None, 1: None,                                         # noise, animal
    2: 'pedestrian', 3: 'pedestrian', 4: 'pedestrian',        # human.pedestrian.adult / child / construction_worker
    5: None,                                                  # human.pedestrian.personal_mobility
    6: 'pedestrian',                                          # human.pedestrian.police_officer
    7: None, 8: None,                                         # human.pedestrian.stroller / wheelchair
    9: 'barrier',                                             # movable_object.barrier
    10: None, 11: None,                                       # movable_object.debris / pushable_pullable
    12: 'traffic_cone',                                       # movable_object.trafficcone
    13: None,                                                 # static_object.bicycle_rack
    14: 'bicycle', 15: 'bus', 16: 'bus', 17: 'car',           # vehicle.bicycle, bus.bendy, bus.rigid, car
    18: 'construction_vehicle',                               # vehicle.construction
    19: None, 20: None,                                       # vehicle.emergency.ambulance / police
    21: 'motorcycle', 22: 'trailer', 23: 'truck',             # vehicle.motorcycle, trailer, truck
    24: 'driveable_surface', 25: 'other_flat', 26: 'sidewalk', 27: 'terrain',   # flat.*
    28: 'manmade',                                            # static.manmade
    29: None,                                                 # static.other
    30: 'vegetation',                                         # static.vegetation
    31: None,                                                 # vehicle.ego
}

MAPS = {
    'pandaset_scalr17': lambda: ClassMap(PANDASET_SCALR17_NAMES, PANDASET_SCALR17_RAW_TO_NAME),
    'pandaset_3dlp8': lambda: ClassMap(PANDASET_3DLP8_NAMES, PANDASET_3DLP8_RAW_TO_NAME),
    'nuscenes_lidarseg16': lambda: ClassMap(NUSCENES_LIDARSEG16_NAMES, NUSCENES_LIDARSEG16_RAW_TO_NAME),
}


def build_class_map(name):
    if name not in MAPS:
        raise KeyError('unknown class map %r; known: %s' % (name, sorted(MAPS)))
    return MAPS[name]()
