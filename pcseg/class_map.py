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


# --- PandaSet (42 classes, dataset/<seq>/annotations/semseg/classes.json) -> a nuScenes-16-like set ---------------------
# DRAFT, open item O2 of episodes/20261116_01: to be approved by the user before the first training. The rule: follow the
# nuScenes lidarseg definitions where they exist (pick-ups are trucks, scooters are motorcycles, emergency vehicles,
# personal mobility, pushable objects and animals are ignored), so that a later nuScenes pair reuses the map; weather
# and sensor artefacts (smoke, exhaust, spray, reflection) are ignored. One map serves both PandaSet sensors.
PANDASET_NAMES_DRAFT = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer',
    'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
]
PANDASET_RAW_TO_NAME_DRAFT = {
    0: None,                        # not in classes.json, but present: 19 of 22.06 M points in 5 of 130 sampled frames
                                    # (every 3rd semseg sequence x frames 0/20/40/60/79, 2026-11-16); treated as unlabelled
    1: None,                        # Smoke
    2: None,                        # Exhaust
    3: None,                        # Spray or rain
    4: None,                        # Reflection
    5: 'vegetation',                # Vegetation
    6: 'terrain',                   # Ground
    7: 'driveable_surface',         # Road
    8: 'driveable_surface',         # Lane Line Marking
    9: 'driveable_surface',         # Stop Line Marking
    10: 'driveable_surface',        # Other Road Marking
    11: 'sidewalk',                 # Sidewalk
    12: 'other_flat',               # Driveway
    13: 'car',                      # Car
    14: 'truck',                    # Pickup Truck (nuScenes: pick-ups are vehicle.truck)
    15: 'truck',                    # Medium-sized Truck
    16: 'truck',                    # Semi-truck
    17: 'trailer',                  # Towed Object
    18: 'motorcycle',               # Motorcycle
    19: 'construction_vehicle',     # Other Vehicle - Construction Vehicle
    20: None,                       # Other Vehicle - Uncommon
    21: None,                       # Other Vehicle - Pedicab
    22: None,                       # Emergency Vehicle (nuScenes lidarseg-16 ignores vehicle.emergency.*)
    23: 'bus',                      # Bus
    24: None,                       # Personal Mobility Device
    25: 'motorcycle',               # Motorized Scooter
    26: 'bicycle',                  # Bicycle
    27: None,                       # Train
    28: None,                       # Trolley
    29: None,                       # Tram / Subway
    30: 'pedestrian',               # Pedestrian
    31: 'pedestrian',               # Pedestrian with Object
    32: None,                       # Animals - Bird
    33: None,                       # Animals - Other
    34: 'barrier',                  # Pylons
    35: 'barrier',                  # Road Barriers
    36: 'manmade',                  # Signs
    37: 'traffic_cone',             # Cones
    38: 'manmade',                  # Construction Signs
    39: 'barrier',                  # Temporary Construction Barriers
    40: None,                       # Rolling Containers (nuScenes: movable_object.pushable_pullable, ignored)
    41: 'manmade',                  # Building
    42: 'manmade',                  # Other Static Object
}

MAPS = {
    'pandaset_nus16_draft': lambda: ClassMap(PANDASET_NAMES_DRAFT, PANDASET_RAW_TO_NAME_DRAFT),
}


def build_class_map(name):
    if name not in MAPS:
        raise KeyError('unknown class map %r; known: %s' % (name, sorted(MAPS)))
    return MAPS[name]()
