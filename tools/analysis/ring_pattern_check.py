"""Pre-launch checks for RING_PATTERN on Waymo -> nuScenes (experiments_md 20261005_01). CPU.

    python analysis/ring_pattern_check.py <ringpattern cfg> <control cfg> <frames> <nuscenes_prof.npy or ->

A. density of the thinned TOP cloud (horizontal range about the vehicle origin) and its radial profile against
   nuScenes TRAIN (label-free reference: 1.5 m bins, pts/frame), plus the raw Waymo cloud;
B. laser lines per car: distinct kept rows in Waymo TRAIN Vehicle boxes (SOURCE labels) at 20 / 40 / 60 m, raw and
   thinned, against what nuScenes' 1.33 deg lattice gives on the same box from a 1.84 m sensor (geometry, no target
   label);
C. Car training boxes with zero points in the cloud the detector is trained on (real __getitem__, training mode),
   for this row and the control;
D. loader cost per sample, this row against the control (single process).
"""
import sys, time
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.waymo.waymo_rings import top_beam_ids, ring_pattern_mask
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

cfg_rp, cfg_ctl, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
nus_prof = np.load(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != '-' else None
logger = common_utils.create_logger()

def build(path):
    c = EasyDict(); cfg_from_yaml_file(path, c)
    ds, _, _ = build_dataloader(dataset_cfg=c.DATA_CONFIG, class_names=c.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=logger, training=True, model_ontology=c.get('ONTOLOGY'))
    return c, ds

c_rp, ds = build(cfg_rp)
rp = c_rp.DATA_CONFIG.RING_PATTERN
import pickle
calib = pickle.load(open(rp.TOP_CALIB, 'rb'))
step = max(1, len(ds.infos) // n)
idx = list(range(0, len(ds.infos), step))[:n]
rng = np.random.default_rng(0)

# ---- A + B
prof_t, prof_r, tot_t, tot_r, near_t, near_r = np.zeros(50), np.zeros(50), [], [], [], []
lines = {20: [], 40: [], 60: []}
for i in idx:
    info = ds.infos[i]; seq = info['point_cloud']['lidar_sequence']
    raw = np.load(ds.data_path / seq / ('%04d.npy' % info['point_cloud']['sample_idx']))
    cnt = np.asarray(info['num_points_of_each_lidar']).astype(int)
    top = raw[:cnt[0]]
    beam, col, el, inc, _ = top_beam_ids(top[:, :3], calib[seq]['extrinsic'], calib[seq]['inclinations'])
    keep, kept = ring_pattern_mask(beam, col, inc, rp.SPACING_DEG, rp.AZ_RES_DEG, rng.random(), rng.random())
    nlz_top = top[:, 5] == -1
    thin = top[keep & nlz_top, :3]
    allr = raw[raw[:, 5] == -1, :3]
    for pts, prof, tot, near in ((thin, prof_t, tot_t, near_t), (allr, prof_r, tot_r, near_r)):
        r = np.hypot(pts[:, 0], pts[:, 1])
        prof += np.histogram(r, bins=50, range=(0, 75))[0]; tot.append((r < 75).sum()); near.append((r < 20).sum())
    an = info['annos']; m = an['name'] == 'Vehicle'
    boxes = np.asarray(an['gt_boxes_lidar'])[m][:, :7]
    if len(boxes):
        rb = np.hypot(boxes[:, 0], boxes[:, 1])
        inb = roiaware_pool3d_utils.points_in_boxes_cpu(top[:, :3].astype(np.float32), boxes.astype(np.float32)) > 0
        for R in lines:
            for j in np.nonzero(np.abs(rb - R) <= 2)[0]:
                pin = inb[j] & nlz_top
                if pin.sum() < 3:
                    continue
                zb, zt = boxes[j, 2] - boxes[j, 5] / 2, boxes[j, 2] + boxes[j, 5] / 2
                exp_nus = (np.arctan2(1.84 - zb, rb[j]) - np.arctan2(1.84 - zt, rb[j])) / np.radians(1.33)
                exp_way = (np.arctan2(2.184 - zb, rb[j]) - np.arctan2(2.184 - zt, rb[j])) / np.radians(1.33)
                lines[R].append((len(np.unique(beam[pin])), len(np.unique(beam[pin & keep])), exp_nus, exp_way,
                                 pin.sum(), (pin & keep).sum()))
k = len(idx)
print(f'A. {k} Waymo TRAIN frames: in range (<75 m) raw {np.mean(tot_r):.0f} -> thinned {np.mean(tot_t):.0f} pts/frame '
      f'(nuScenes TRAIN 25,068); 0-20 m raw {np.mean(near_r):.0f} -> thinned {np.mean(near_t):.0f} (nuScenes 20,497)')
if nus_prof is not None:
    pt = prof_t / k
    print('   radial profile thinned / nuScenes TRAIN per 7.5 m (pts/frame):')
    for a in range(0, 50, 5):
        print(f'     {a*1.5:4.1f}-{(a+5)*1.5:4.1f} m: {pt[a:a+5].sum():7.0f} / {nus_prof[a:a+5].sum():7.0f} = {pt[a:a+5].sum()/max(nus_prof[a:a+5].sum(),1):.2f}')
print('B. laser lines per Vehicle box (median), raw Waymo -> thinned; geometric expectation of a 1.33 deg lattice on the same box: nuScenes (1.84 m) / Waymo (2.184 m); points raw -> thinned')
for R, v in lines.items():
    v = np.array(v)
    if len(v):
        print(f'   {R} m (n={len(v)}): {np.median(v[:,0]):.0f} -> {np.median(v[:,1]):.0f} lines; expected {np.median(v[:,2]):.2f} / {np.median(v[:,3]):.2f}; points {np.median(v[:,4]):.0f} -> {np.median(v[:,5]):.0f}')

# ---- C + D
def emptied_and_time(dset, frames):
    tot = emp = 0; t = time.time()
    for i in frames:
        d = dset[i]
        b = d['gt_boxes']; b = b[b[:, 7] == 1] if len(b) else b
        if not len(b): continue
        cnt = roiaware_pool3d_utils.points_in_boxes_cpu(d['points'][:, :3].astype(np.float32), b[:, :7].astype(np.float32)).sum(1)
        tot += len(b); emp += int((cnt == 0).sum())
    return emp, tot, (time.time() - t) / len(frames)
fr = idx[:min(len(idx), 150)]
np.random.seed(0)
e1, t1, s1 = emptied_and_time(ds, fr)
_, ds_c = build(cfg_ctl)
np.random.seed(0)
e0, t0, s0 = emptied_and_time(ds_c, fr)
print(f'C. Car training boxes with 0 points in the training cloud (real __getitem__, {len(fr)} frames): '
      f'ring pattern {e1}/{t1} = {e1/max(t1,1):.3f}; control {e0}/{t0} = {e0/max(t0,1):.3f}')
print(f'D. loader seconds per sample (single process, master node): ring pattern {s1:.3f}, control {s0:.3f}')
