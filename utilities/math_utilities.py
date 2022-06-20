import numpy as np
import warnings
import os

from itertools import permutations

from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from skimage.measure import regionprops

try:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax
    from jax.scipy import ndimage as jndimage

    @jax.jit
    def jit_map_coordinates(image, coords):
        return jndimage.map_coordinates(image, coords, order=1)
except ModuleNotFoundError:
    warnings.warn("Could not import jax. Please install to use jax backend for speedup.")


def matlab_mod(a, m):
    return a-m*np.floor(a/m)


def robust_linear_regression(x, y, p):
    # Robust linear regression
    n_points = np.round(len(x)*p/100).astype(np.uint32)
    coeff = np.array(list(permutations(range(1, n_points), 2)))
    indices = np.random.permutation(n_points)
    indices = indices[:n_points]
    x = np.array(x)[indices]
    y = np.array(y)[indices]

    slope = np.median((y[coeff[:, 1]]-y[coeff[:, 0]])/(x[coeff[:, 1]]-x[coeff[:, 0]]))
    intercept = np.median(y-slope*x)
    return slope, intercept


def transform_points(transform, points, dtype=np.float32):
    points = np.append(points, np.ones((*points.shape[:-1], 1)), axis=-1)
    updated_points = points @ transform
    return updated_points[..., :3].astype(dtype=dtype)


def oblique_slice(volume, point, normal, return_grids=False, backend=None):
    """
    Function to perform 2D oblique slicing of volume.

    Parameters
    ----------
    volume: np.ndarray
            Volume to perform oblique slicing
    point:  Sequence or np.ndarray
            Point on plane
    normal: Sequence or np.ndarray
            Normal vector perpendicular to the plane
    return_grids: bool
            Whether to return the coordinates of the extracted slice in the volume
    backend: str
            Which backend to use to map coordinates. Options are "jax" and "scipy". Default: Choose jax if available.

    Returns
    -------
    np.ndarray
        2D oblique slice
    """

    width, height, depth = volume.shape
    unit_normal = normal/np.linalg.norm(normal)
    init_normal = [0, 0, 1]

    if np.allclose(unit_normal, init_normal):
        rot_axis = unit_normal
    else:
        rot_axis = np.cross(init_normal, unit_normal)
        rot_axis = rot_axis/np.linalg.norm(rot_axis)
        rot_axis[np.isnan(rot_axis)] = np.finfo(float).eps

    angle = np.arccos(np.dot(init_normal, unit_normal))
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(-angle*rot_axis).as_matrix()

    plane_size = np.max(volume.shape)*3
    n_rows = plane_size
    n_cols = plane_size

    # Grids of x, y, z of a plane with origin (0, 0, 0) as the center
    xp, yp, zp = np.meshgrid(np.arange(-n_cols//2, n_cols//2+1, dtype=np.int32),
                             np.arange(-n_rows//2, n_rows//2+1, dtype=np.int32),
                             np.array(0, dtype=np.int32))

    # Rotated coordinates of the plane wrt. transform
    xr, yr, zr = transform_points(transform, np.concatenate((xp, yp, zp), axis=-1), dtype=np.float32).T

    # Zero center input point
    point_offset = point-np.array([np.round(width/2), np.round(height/2), np.round(depth/2)])

    # Shortest distance between plane passing through origin and input point
    shortest_distance = -np.dot(unit_normal, point_offset)

    # Translate plane passing from origin to input point
    offset = shortest_distance*unit_normal - np.array([np.round(width/2), np.round(height/2), np.round(depth/2)])
    xq = xr - offset[0]
    yq = yr - offset[1]
    zq = zr - offset[2]

    if backend == "scipy":
        interp = map_coordinates(volume, coordinates=(xq, yq, zq))
    elif backend == "jax" or backend is None:
        try:
             interp = np.array(jit_map_coordinates(volume, (xq, yq, zq)))
        except NameError: #ModuleNotFoundError:
            interp = map_coordinates(volume, coordinates=(xq, yq, zq))

    slice_mask_limit = np.all(np.stack((np.logical_and(xq >= 1, xq <= width), np.logical_and(yq >= 1, yq <= height),
                                        np.logical_and(zq >= 1, zq <= depth))), axis=0)
    xmin, ymin, xmax, ymax = regionprops(slice_mask_limit.astype(np.uint8))[0].bbox

    if return_grids:
        return interp[xmin:xmax, ymin:ymax], \
               (xq[xmin:xmax, ymin:ymax], yq[xmin:xmax, ymin:ymax], zq[xmin:xmax, ymin:ymax])
    return interp[xmin:xmax, ymin:ymax]


def map_point_to_plane(point, plane_grids):
    norm_diff = np.linalg.norm(np.stack(plane_grids, axis=-1)-point, axis=-1)
    min_index_raveled = np.argmin(norm_diff)
    return np.unravel_index(min_index_raveled, shape=plane_grids[0].shape)


def rotation_matrix_from_vectors(vec_a, vec_b):
    """ Find the rotation matrix that aligns vec_1 with vec_2

    Parameters
    ----------
        vec_a: list or np.ndarray
            Source vector
        vec_b: list or np.ndarray
            Destination vector

    Returns
    -------
    Rotation matrix (3x3) that aligns vec_a with vec_b.

    """
    a, b = (vec_a / np.linalg.norm(vec_a)).reshape(3), (vec_b / np.linalg.norm(vec_b)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def find_centroid(arr):
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])
