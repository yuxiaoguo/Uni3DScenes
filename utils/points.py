"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""

import numpy as np


def points2voxel(points: np.ndarray, res=0.005):
    """
    Convert points into voxel coordinates
    """
    p_points = points

    vd_points = np.floor(p_points / res).astype(np.int64)
    vd_max = np.max(vd_points, axis=0)
    vd_min = np.min(vd_points, axis=0)
    vd_box = np.cumprod([1, *(vd_max - vd_min)[:2]])

    vb_points = vd_points - vd_min[np.newaxis, ...]

    vd_indices = np.sum(vb_points * vd_box[np.newaxis, ...], axis=-1)
    _, vd_uni = np.unique(vd_indices, return_index=True)
    return vd_uni, vb_points[vd_uni]
