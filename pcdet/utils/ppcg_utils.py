"""PPCG geometry: reference-object frames, ray-constrained resampling, reference libraries.

DALI's PPCG (IEEE T-RO 2024) attacks pseudo-label noise at the instance level. Where ST3D refines
a box to fit the points inside it, PPCG regenerates the POINTS to fit the box: for a pseudo box
with too few returns, it finds a clean reference object of the same class, scales it to the box's
l/w/h, and replaces each observed return with the reference surface point that lies along the same
ray. The rays are unchanged - only what they hit is. That is what makes it ray-CONSTRAINED, and
what keeps the result consistent with the sensor's own sampling pattern.

This module is Tier D2.1/D2.2 of the plan in experiments_md/20260922_04 section 2.4: the reference
library and the ray tracer. It does NOT contain the generation pass (D2.3) or the dataset that
serves the result (D2.4), and it deliberately predates the Tier D1 stopping rule being applied -
these two pieces are the ones CF-PPCG (D3) would reuse as well, so they are the cheapest part of
D2 to own early.

Ported from upstream's `simulation_utils/mesh_ray_intersection.py::mesh_ray_tracing_point1` and
`tools/ppcg_points.py::generate_source_db`. The functions we need are numpy/scipy only - upstream's
module additionally imports trimesh, vedo, sklearn and a custom CUDA op at module scope, none of
which is in this container, which is why the FUNCTIONS are ported rather than the file.

FOUR upstream defects are deliberately not reproduced; each is called out at its site:
  1. mesh_ray_tracing_point1 mutates its caller's reference array in place.
  2. it argsorts an entire (N x M) distance matrix to read the k=10 smallest of each row.
  3. generate_source_db computes an object's `size` from the box's CENTRE COORDINATES rather than
     its extent, which is meaningless and can go NaN.
  4. the reference-matching step measures angular difference as abs(diff % 2*pi), which maps a
     near-perfect match at -epsilon onto ~2*pi, the worst possible score. `angular_difference`
     here is the correct circular distance; see its docstring.
"""
import numpy as np
from scipy.spatial.distance import cdist


def rotation_z(angle):
    """Rotation about +z by `angle`, as a (3, 3) matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def angular_difference(a, b):
    """Signed-magnitude circular distance between two angles, in [0, pi].

    Upstream writes `abs((a - b) % (2*pi))`. numpy's `%` returns a result in [0, 2*pi), so a
    difference of -0.01 rad - a near-perfect match - comes out as 6.273 and is then ranked as the
    WORST possible candidate. Half of all near-matches are scored as maximally distant, which
    systematically corrupts reference selection. Wrapping to [-pi, pi] first is the fix.
    """
    return np.abs((np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi)


def canonicalize_object_points(points, box):
    """World-frame points inside `box` -> the object's own frame (centred, heading-aligned).

    This is the frame a reference library is stored in, so that one object can later be placed at
    any pose. `box` is [x, y, z, dx, dy, dz, heading]; only columns 0:3 and 6 are read, and
    columns beyond xyz in `points` (intensity, ...) are carried through untouched.
    """
    points = np.asarray(points)
    if len(points) == 0:
        return points.copy()
    out = points.copy().astype(np.float64)
    out[:, 0:3] -= np.asarray(box)[0:3]
    out[:, 0:3] = out[:, 0:3] @ rotation_z(-float(box[6])).T
    return out


def place_object_points(points, box):
    """The exact inverse of `canonicalize_object_points`: object frame -> world frame at `box`."""
    points = np.asarray(points)
    if len(points) == 0:
        return points.copy()
    out = points.copy().astype(np.float64)
    out[:, 0:3] = out[:, 0:3] @ rotation_z(float(box[6])).T
    out[:, 0:3] += np.asarray(box)[0:3]
    return out


def scale_object_points(points, reference_lwh, target_lwh):
    """Stretch a canonical object to another box's extent, per axis.

    Per-axis rather than isotropic, matching upstream - a reference car and a target box can differ
    in aspect ratio, and the point of PPCG is that the regenerated points fill THIS box.
    """
    points = np.asarray(points)
    if len(points) == 0:
        return points.copy()
    ratio = np.asarray(target_lwh, dtype=np.float64) / np.asarray(reference_lwh, dtype=np.float64)
    out = points.copy().astype(np.float64)
    out[:, 0:3] = out[:, 0:3] * ratio[None, :]
    return out


def ray_constrained_resample(reference_points, observed_points, ray_origin, box, k=10):
    """RC-PPCG's core: replace each observed return with a reference point on the same ray.

    Port of upstream `mesh_ray_tracing_point1(pts_ref, ray_origin, pts, bbox)`.

    Args:
        reference_points: (M, 3+C) canonical object points, already scaled to this box's extent.
        observed_points: (N, 3+C) the real returns inside the box, in world frame.
        ray_origin: (3,) sensor position in the same frame - NOT hardcoded per dataset here; see
            `sensor_origin` for why.
        box: [x, y, z, dx, dy, dz, heading] the reference object is placed at.
        k: how many best-aligned reference points to consider per ray before taking the nearest.

    Returns:
        (N, 3+C) regenerated points - exactly one per observed return, so the ray pattern, the
        point count and hence the density of the box are all preserved. Only the surface they
        land on changes.

    For each observed ray the k most closely aligned reference rays are taken (cosine distance on
    the direction vectors), and among those the SHALLOWEST is kept - the first surface the ray
    would strike, which is what makes the result self-occluding rather than a mixture of front
    and back faces.

    Two departures from upstream. It does not mutate `reference_points`; upstream rotates and
    translates the caller's array in place, which is safe there only because the one call site
    deep-copies first. And it selects the k smallest with `argpartition` (O(N*M)) instead of
    argsorting every row in full (O(N*M log M)) to read ten of them - the ordering within the k
    is never used, only the depth argmin over them. Exact ties at the k-th distance may resolve
    to a different member than upstream's argsort would pick.
    """
    reference_points = np.asarray(reference_points)
    observed_points = np.asarray(observed_points)
    if len(observed_points) == 0 or len(reference_points) == 0:
        return observed_points.copy()

    ray_origin = np.asarray(ray_origin, dtype=np.float64).reshape(3)
    reference_world = place_object_points(reference_points, box)

    observed_rays = observed_points[:, 0:3].astype(np.float64) - ray_origin
    reference_rays = reference_world[:, 0:3] - ray_origin
    observed_depth = np.linalg.norm(observed_rays, axis=1)
    reference_depth = np.linalg.norm(reference_rays, axis=1)

    # A return exactly at the sensor origin has no direction; cosine distance would be NaN and
    # would then win every argmin. Drop such reference points and pass such observations through.
    usable = reference_depth > 0
    if not usable.any():
        return observed_points.copy()
    reference_world, reference_rays = reference_world[usable], reference_rays[usable]
    reference_depth = reference_depth[usable]

    observed_rays = observed_rays / np.where(observed_depth > 0, observed_depth, 1.0)[:, None]
    reference_rays = reference_rays / reference_depth[:, None]

    distance = cdist(observed_rays, reference_rays, 'cosine')
    k = int(min(k, reference_rays.shape[0]))
    nearest_in_angle = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
    depth_of_candidates = reference_depth[nearest_in_angle]
    chosen = nearest_in_angle[np.arange(len(observed_points)),
                              np.argmin(depth_of_candidates, axis=1)]

    out = reference_world[chosen]
    if observed_points.shape[1] > 3:
        # Carry the observed returns' own extra channels; the reference object's intensity belongs
        # to a different sensor and a different surface. Moot while DA runs xyz-only, but the
        # regenerated cloud has to have the same width as the one it replaces.
        out = np.concatenate([out[:, 0:3], observed_points[:, 3:]], axis=1)
    out[observed_depth == 0] = observed_points[observed_depth == 0]
    return out


def sensor_origin(dataset_cfg):
    """Where the sensor sits in the frame the points are expressed in.

    Upstream hardcodes [0, 0, 1.6] for KITTI and [0, 0, 1.8] otherwise. In this repo the frame is
    decided by SHIFT_COOR, which lifts the cloud so the ground plane sits at z = 0 - so the sensor
    is at z = SHIFT_COOR[2]. It is set only in the *_point_calibrated* configs; the *_default
    family runs unshifted, where the sensor IS the origin. Deriving it means a ray origin that is
    right for both families instead of right for one.
    """
    shift = dataset_cfg.get('SHIFT_COOR', None) if hasattr(dataset_cfg, 'get') else None
    return np.array([0.0, 0.0, float(shift[2]) if shift is not None else 0.0])


def save_reference_objects(path, objects, metadata=None):
    """Write a reference library: one packed .npz rather than a file per object.

    Upstream writes `objects/<idx>.npy` per object plus a pickled index - for a KITTI Car library
    that is ~4,000 small files on an NFS home the lab manual explicitly flags as a round-trip
    bottleneck, and the whole library is loaded into RAM at once anyway. Packing the points into
    one array with offsets is the same data, one read.

    Args:
        path: destination .npz.
        objects: list of dicts with keys points (P, 3+C), box (7,), direction, orientation,
            numpts, size, name.
        metadata: optional dict of plain scalars/strings recorded alongside.
    """
    points = np.concatenate([np.asarray(o['points'], dtype=np.float32) for o in objects], axis=0) \
        if objects else np.zeros((0, 3), dtype=np.float32)
    counts = np.array([len(o['points']) for o in objects], dtype=np.int64)
    np.savez_compressed(
        path,
        points=points,
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        boxes=np.array([o['box'] for o in objects], dtype=np.float64).reshape(-1, 7),
        direction=np.array([o['direction'] for o in objects], dtype=np.float64),
        orientation=np.array([o['orientation'] for o in objects], dtype=np.float64),
        numpts=counts,
        size=np.array([o['size'] for o in objects], dtype=np.float64),
        name=np.array([str(o['name']) for o in objects]),
        metadata=np.array([repr(metadata or {})]),
    )


class ReferenceObjects(object):
    """A loaded reference library. `lwh`, `direction` and `orientation` are what matching reads."""

    def __init__(self, blob):
        self._points = blob['points']
        self._offsets = blob['offsets']
        self.boxes = blob['boxes']
        self.direction = blob['direction']
        self.orientation = blob['orientation']
        self.numpts = blob['numpts']
        self.size = blob['size']
        self.name = blob['name']
        self.metadata = str(blob['metadata'][0]) if 'metadata' in blob else '{}'

    def __len__(self):
        return len(self._offsets) - 1

    def points(self, index):
        """Canonical (centred, heading-aligned) points of object `index`."""
        return self._points[self._offsets[index]:self._offsets[index + 1]]

    @property
    def lwh(self):
        """(K, 3) extents, the quantity a pseudo box is scaled against."""
        return self.boxes[:, 3:6]


def load_reference_objects(path):
    """Read a library written by `save_reference_objects`."""
    return ReferenceObjects(np.load(path, allow_pickle=False))
