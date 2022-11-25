# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

class S3DUtilize(object):
    """
    Structured3D utilize functions
    """
    @staticmethod
    def get_fov_normal(image_size, cam_focal, norm=True):
        """
        Get the normal FoV directions
        """
        u2x, v2y = [(np.arange(1, image_size[a_i] + 1) - image_size[a_i] / 2) / cam_focal[a_i]\
            for a_i in [0, 1]]
        cam_m_u2x = np.tile([u2x], (image_size[1], 1))
        cam_m_v2y = np.tile(v2y[:, np.newaxis], (1, image_size[0]))
        cam_m_depth = np.ones(image_size).T
        fov_normal = np.stack((cam_m_depth, -1 * cam_m_v2y, cam_m_u2x), axis=-1)
        if norm:
            fov_normal = fov_normal / np.sqrt(np.sum(np.square(fov_normal), axis=-1, keepdims=True))
        return fov_normal

    @staticmethod
    def cast_perspective_to_local_coord(depth_img: np.ndarray, fov_normal):
        """
        Cast the perspective image into 3D coordinate system
        """
        return depth_img * fov_normal

    @staticmethod
    def cast_points_to_voxel(points, labels, room_size=(6.4, 3.2, 6.4), room_stride=0.2):
        """
        Voxelize the points
        """
        vol_resolution = (np.asarray(room_size) / room_stride).astype(np.int32)
        vol_index = np.floor(points / room_stride).astype(np.int32)
        in_vol = np.logical_and(np.all(vol_index < vol_resolution, axis=1), \
            np.all(vol_index >= 0, axis=1))
        v_x, v_y, v_z = [d_[..., 0] for d_ in np.split(vol_index[in_vol], 3, axis=-1)]
        vol_label = labels[in_vol]
        vol_data = np.zeros(vol_resolution, dtype=np.uint8)
        vol_data[v_x, v_y, v_z] = vol_label
        return vol_data

    @staticmethod
    def get_rotation_matrix_from_tu(cam_front, cam_up):
        """
        Get the rotation matrix from TU-coords
        """
        cam_n = np.cross(cam_front, cam_up)
        cam_m = np.stack((cam_front, cam_up, cam_n), axis=1).astype(np.float32)
        return cam_m
