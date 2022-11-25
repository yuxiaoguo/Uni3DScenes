"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
# pylint: disable=no-member
import os
import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image as pil_image 

from graphics_utils import g_io, g_math, g_perf

from .s3d_utils import S3DUtilize
from ..utils.base import DatasetBase, ProcessUnit


class Structured3DDataGen(DatasetBase):
    """
    Dataset generation for Structured3D
    """
    IMAGE_PREFIX = '/2D_rendering'

    PERSPECTIVE_PREFIX = '/perspective/full'
    PRSP_CAM_FILE = 'camera_pose.txt'

    PANORAMIC_PREFIX = '/panorama/full'
    PANO_CAM_PREFIX = '/panorama'
    PANO_CAM_FILE = 'camera_xyz.txt'

    SEMANTIC_FILE = 'semantic.png'
    RGB_FILE = 'rgb_rawlight.png'
    DEPTH_FILE = 'depth.png'

    def __init__(self, root_dir: str, proc_units: List[ProcessUnit]) -> None:
        super().__init__(root_dir, proc_units)

        self.scenes_dict = dict()

    def _load_zips(self, filter_regex='Structured3D') -> g_io.GroupZipIO:
        ctx_files = [f for f in os.listdir(self.root_dir) if filter_regex in f and \
            f.endswith('zip')]
        zip_reader = g_io.GroupZipIO([os.path.join(self.root_dir, f) for f in ctx_files])
        return zip_reader

    def _get_rooms_list_by_types(self, room_types: List[str]) -> List[str]:
        assert len(room_types) == 1 and 'all' in room_types
        zip_reader = self._load_zips()
        scenes_list = [_c.split('/')[1] for _c in zip_reader.namelist() if \
            _c.find('annotation_3d.json') != -1]
        rooms_list = list()
        for scene_path in scenes_list:
            rooms_name = zip_reader.listdir(f'{scene_path}{__class__.IMAGE_PREFIX}')
            rooms_list.extend([f'{scene_path}{__class__.IMAGE_PREFIX}/{_r}' for _r in rooms_name])
        return rooms_list

    @staticmethod
    def _read_camera_and_image(zip_reader: g_io.GroupZipIO, cam_path: str, info_flags: int, \
        info_root: str) -> Tuple[List, List[np.ndarray]]:
        if info_root is None:
            info_root = cam_path[:cam_path.rfind('/')]

        out_cams = list()
        if info_flags & 1:
            # Load camera poses
            z2y_top_m = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
            cam_extr = np.fromstring(io.BytesIO(zip_reader.read(cam_path)).read(), \
                dtype=np.float32, sep=' ')
            cam_t = np.matmul(z2y_top_m, cam_extr[:3] / 1000)
            if cam_extr.shape[0] > 3:
                cam_r = S3DUtilize.get_rotation_matrix_from_tu(cam_extr[3:6], cam_extr[6:9])
                cam_r = np.matmul(z2y_top_m, cam_r)
                cam_hf = cam_extr[9:11]
            else:
                cam_r = np.eye(3, dtype=np.float32)
                cam_hf = None
            out_cams.extend([cam_r, cam_t, cam_hf])
        out_images = list()
        if info_flags & 2:
            # Load depth image
            depth_image = zip_reader.read_image(f'{info_root}/{__class__.DEPTH_FILE}')[..., np.newaxis]
            depth_image[depth_image == 0] = 65535
            out_images.append(depth_image)
        if info_flags & 4:
            # Load RGB image
            color_image = cv2.imdecode(np.frombuffer(io.BytesIO(zip_reader.read(\
                f'{info_root}/{__class__.RGB_FILE}')).read(), np.uint8), \
                cv2.IMREAD_UNCHANGED)[...,:3][..., ::-1]
            out_images.append(color_image)
        if info_flags & 8:
            # Load semantic image
            smnt_image = np.array(pil_image.open(io.BytesIO(zip_reader.read(\
                f'{info_root}/{__class__.SEMANTIC_FILE}'))))[..., np.newaxis]
            out_images.append(smnt_image)
        return out_cams, out_images

    @staticmethod
    def _view2points_prsp(cam_paras: List[np.ndarray], attr_images: List[np.ndarray], cos_thrsh=0.15):
        depth_img, color_img, smnt_img = attr_images
        cam_r, cam_t, cam_hf = cam_paras
        img_size = np.asarray(depth_img.shape[:2])[::-1]
        cam_focal = img_size / 2 / np.tan(cam_hf)
        cam_fov_d = S3DUtilize.get_fov_normal(img_size, cam_focal).astype(np.float32)
        v_points = S3DUtilize.cast_perspective_to_local_coord(depth_img, cam_fov_d)
        v_normal = g_math.normal_from_cross_product(v_points)

        # Filtering invalid points
        view_dist = np.maximum(np.linalg.norm(v_points, axis=-1, keepdims=True), float(10e-5))
        cosine_dist = np.sum((v_points * v_normal / view_dist), axis=-1, keepdims=True)
        cosine_dist = np.abs(cosine_dist)
        point_valid = cosine_dist > cos_thrsh
        depth_valid = depth_img < 65535
        smnt_valid = smnt_img > 0
        all_valid = (point_valid & depth_valid & smnt_valid)[..., 0]

        v_points = np.matmul(v_points / 1000, cam_r.T) + cam_t

        return v_points[all_valid], color_img[all_valid], smnt_img[all_valid]

    @staticmethod
    def _view2points_pano(cam_paras: List[np.ndarray], attr_images: List[np.ndarray]):
        depth_img, color_img, smnt_img = attr_images
        _, cam_t, _ = cam_paras
        p_h, p_w = attr_images[0].shape[:2]
        p_a = np.arange(p_w, dtype=np.float32) / p_w * 2 * np.pi - np.pi
        p_b = np.arange(p_h, dtype=np.float32) / p_h * np.pi * -1 + np.pi/2
        p_a = np.tile(p_a[None], [p_h, 1])[..., np.newaxis]
        p_b = np.tile(p_b[:, None], [1, p_w])[..., np.newaxis]
        p_a_sin, p_a_cos, p_b_sin, p_b_cos = np.sin(p_a), np.cos(p_a), np.sin(p_b), np.cos(p_b)
        point_x = depth_img * p_a_cos * p_b_cos
        point_y = depth_img * p_b_sin
        point_z = depth_img * p_a_sin * p_b_cos
        points = np.concatenate([point_x, point_y, point_z], axis=-1)
        points = points / 1000 + cam_t

        # Filtering invalid points
        all_valid = np.logical_and(depth_img < 65535, smnt_img > 0)[..., 0]

        return points[all_valid], color_img[all_valid], smnt_img[all_valid]

    @staticmethod
    def _points2voxel(attr_points: List[np.ndarray], res=0.005):
        p_points, p_colors, p_labels = attr_points

        if not p_points.shape:
            return list(), list(), list()
        
        vd_points = np.floor(p_points / res).astype(np.int64)
        vd_max = np.max(vd_points, axis=0)
        vd_min = np.min(vd_points, axis=0)
        vd_box = np.cumprod([1, *(vd_max - vd_min)[:2]])

        vd_indices = np.sum((vd_points - vd_min[np.newaxis, ...]) * vd_box[np.newaxis, ...], axis=-1)
        _, vd_uni = np.unique(vd_indices, return_index=True)

        return p_points[vd_uni], p_colors[vd_uni], p_labels[vd_uni]

    def _mp_view2pointcloud(self, rooms_list: List[str], proc_unit: ProcessUnit,\
        start_index=0, worker_id=0):
        del start_index, worker_id
        zip_reader = self._load_zips()

        dump_folder = os.path.join(self.root_dir, 'point_cloud')
        os.makedirs(dump_folder, exist_ok=True)

        for r_idx, room_path in enumerate(rooms_list):
            scene_id, _, room_id = room_path.split('/')
            dump_name = f'{scene_id}_{room_id}'
            dump_path = os.path.join(dump_folder, f'{dump_name}_v001_points.npz')
            if os.path.exists(dump_path):
                continue

            prsp_root = f'{room_path}{__class__.PERSPECTIVE_PREFIX}'
            cam_paths = [_c for _c in zip_reader.namelist() if _c.find(prsp_root) != -1 and \
                _c.find(__class__.PRSP_CAM_FILE) != -1]
            all_infos = list()
            for cam_path in cam_paths:
                cam_paras, attr_images = self._read_camera_and_image(zip_reader, cam_path, 15, None)
                r_points, r_colors, r_labels = self._view2points_prsp(cam_paras, attr_images)
                all_infos.append((r_points, r_colors, r_labels))

            pano_cam_root = f'{room_path}{__class__.PANO_CAM_PREFIX}'
            cam_paths = [_c for _c in zip_reader.namelist() if _c.find(pano_cam_root) != -1 and \
                _c.find(__class__.PANO_CAM_FILE) != -1]
            for cam_path in cam_paths:
                pano_root = cam_path[:cam_path.rfind('/')]
                pano_root = pano_root[:pano_root.rfind('/')]
                pano_root = f'{pano_root}{__class__.PANORAMIC_PREFIX}'
                cam_paras, attr_images = self._read_camera_and_image(zip_reader, cam_path, 15, \
                    pano_root)
                r_points, r_colors, r_labels = self._view2points_pano(cam_paras, attr_images)
                all_infos.append((r_points, r_colors, r_labels))

            a_points = np.concatenate([_i[0] for _i in all_infos], axis=0)
            a_colors = np.concatenate([_i[1] for _i in all_infos], axis=0)
            a_labels = np.concatenate([_i[2] for _i in all_infos], axis=0)
            # np.savez(f'{dump_name}_raw_points.npz', points=a_points, colors=a_colors, \
            #     labels=a_labels)

            v_points, v_colors, v_labels = self._points2voxel((a_points, a_colors, a_labels), 0.01)
            np.savez(dump_path, points=v_points, colors=v_colors, labels=v_labels)

    def view2pointcloud(self, proc_unit: ProcessUnit):
        rooms_list = self._get_rooms_list_by_types(proc_unit.room_types)

        g_perf.multiple_processor(self._mp_view2pointcloud, rooms_list, 8, \
            (proc_unit, ))
        # self._mp_view2pointcloud(rooms_list, proc_unit)
