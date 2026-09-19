"""
Ad-hoc verification script (not a permanent tool): computes per-class GT box counts for
PandaSet's pandar64 (device=0) vs PandarGT (device=1) lidars, split by train/val, to check the
CLASS_NAMES: ['Car', 'Pedestrian', 'Bicycle'] assumption in
cfgs/pandaset-pandar64-to-pandargt_models/centerpoint-rospm-C.yaml against real GT label
frequency. See experiments_md/20260829_03_pandaset_sensor_gap_config_and_da_ablation_taxonomy.md
for the background.

Run from ST3D/tools/ inside the singularity container (needs pandas + pandaset devkit, and the
--bind /home/koyama/code/ST3D:/root/ST3D mount since pandaset_infos_*.pkl bake in absolute
/root/ST3D/... paths):

    cd /home/koyama/code/ST3D/tools
    singularity exec --nv \
      --bind /home/koyama/data/:/storage \
      --bind /home/koyama/code/ST3D:/root/ST3D \
      /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
      python3 verify_pandaset_pandargt_classes.py
"""
import _init_path  # noqa: F401 -- must precede any pcdet import, see repo AGENTS.md gotcha
from collections import Counter

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset
from pcdet.utils import common_utils

CFG_FILE = 'cfgs/pandaset-pandar64-to-pandargt_models/centerpoint-rospm-C.yaml'
LOGGER = common_utils.create_logger()


def count_labels(dataset_cfg, class_names, training):
    ds = PandasetDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=training, logger=LOGGER)
    counter = Counter()
    sensor_ids_seen = set()
    for info in ds.pandaset_infos:
        pose = ds._get_pose(info)
        _, labels, _ = ds._get_annotations(info, pose)
        counter.update(labels.tolist())
    return len(ds.pandaset_infos), counter


def main():
    cfg_from_yaml_file(CFG_FILE, cfg)
    print(f'CLASS_NAMES under test: {cfg.CLASS_NAMES}\n')

    variants = [
        ('pandar64 (source, device=0)', cfg.DATA_CONFIG),
        ('PandarGT (target, device=1)', cfg.DATA_CONFIG_TAR),
    ]
    for device_label, dataset_cfg in variants:
        for training, split_name in [(True, 'train'), (False, 'val')]:
            n_frames, counter = count_labels(dataset_cfg, cfg.CLASS_NAMES, training)
            print(f'--- {device_label} / split={split_name} (n_frames={n_frames}) ---')
            for name, count in counter.most_common():
                flag = '  <-- in CLASS_NAMES' if name in cfg.CLASS_NAMES else ''
                print(f'    {name!r}: {count}{flag}')
            for target_class in cfg.CLASS_NAMES:
                if target_class not in counter:
                    print(f'    WARNING: {target_class!r} has ZERO boxes in this split/device')
            print()


if __name__ == '__main__':
    main()
