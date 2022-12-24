"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
# pylint: disable=no-member
import os
import io
import json
import pickle
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image as pil_image

from graphics_utils import g_io, g_math, g_perf

from utils.config import ProcessUnit, EnvsConfig
from utils.labels.nyu_40 import NYU40
from .s3d_utils import S3DUtilize
from ..base_dataset import DatasetBase
from ..protocol.mmdet3d_scannet import Annotations


class Structured3DDataGen(DatasetBase):
    """
    Dataset generation for Structured3D.

    Two separated folders will be created in target folder -- points and semantic_mask.
        Points will be saved a .bin file with raw shape [N, 6] (3 for XYZ, 3 for RGB)
        and data type np.float32. Semantic mask will be saved a .bin file with raw shape
        [N] and data type np.int64.
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

    ANNO_FILE = 'bbox_3d.json'

    def __init__(self, proc_units: List[ProcessUnit], envs: EnvsConfig) -> None:
        super().__init__(proc_units, envs)
        self._zip_folder = None

    def _load_zips(self, filter_regex='Structured3D') -> g_io.GroupZipIO:
        ctx_files = [f for f in os.listdir(self._zip_folder) if filter_regex in f and \
            f.endswith('zip')]
        zip_reader = g_io.GroupZipIO([os.path.join(self._zip_folder, f) for f in ctx_files])
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
    def read_camera_and_image(zip_reader: g_io.GroupZipIO, cam_path: str, info_flags: int, \
        info_root: str) -> Tuple[List, List[np.ndarray]]:
        """
        Read camera poses and images from GroupZipIO

        Args:
            zip_reader (g_io.GroupZipIO): GroupZipIO instance
            cam_path (str): the relative path of camera
            info_flags (int): the flag of the type of images to be read

        Returns:
            Tuple[List, List[np.ndarray]]: Camera information and a list of images
        """
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
            depth_image = zip_reader.read_image(f'{info_root}/{__class__.DEPTH_FILE}')\
                [..., np.newaxis]
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
    def view2points_prsp(cam_paras: List[np.ndarray], attr_images: List[np.ndarray],
        cos_thrsh=0.15):
        """
        View to 3D points casting of a single perspective image

        Args:
            cam_paras (List[np.ndarray]): camera parameters
            attr_images (List[np.ndarray]): a list of images to be casted
            cos_thrsh (float, optional): the cosine threshold to filtering
                interpolated depth. Defaults to 0.15.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
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
    def view2points_pano(cam_paras: List[np.ndarray], attr_images: List[np.ndarray]):
        """
        View to 3D points casting of a single panoramic image

        Args:
            cam_paras (List[np.ndarray]): camera parameters
            attr_images (List[np.ndarray]): a list of images to be casted

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
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
    def _points2voxel(attr_points: List[np.ndarray], res=0.005) ->\
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_points, p_colors, p_labels = attr_points

        try:
            vd_points = np.floor(p_points / res).astype(np.int64)
            vd_max = np.max(vd_points, axis=0)
            vd_min = np.min(vd_points, axis=0)
            vd_box = np.cumprod([1, *(vd_max - vd_min)[:2]])

            vd_indices = np.sum((vd_points - vd_min[np.newaxis, ...]) * \
                vd_box[np.newaxis, ...], axis=-1)
            _, vd_uni = np.unique(vd_indices, return_index=True)
        except ValueError:
            return None, None, None

        return p_points[vd_uni], p_colors[vd_uni], p_labels[vd_uni]

    @staticmethod
    def _view2points(zip_reader: g_io.GroupZipIO, room_path) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
        prsp_root = f'{room_path}{__class__.PERSPECTIVE_PREFIX}'
        cam_paths = [_c for _c in zip_reader.namelist() if _c.find(prsp_root) != -1 and \
            _c.find(__class__.PRSP_CAM_FILE) != -1]
        all_infos = list()
        for cam_path in cam_paths:
            cam_paras, attr_images = __class__.read_camera_and_image(zip_reader, cam_path, 15, None)
            r_points, r_colors, r_labels = __class__.view2points_prsp(cam_paras, attr_images)
            all_infos.append((r_points, r_colors, r_labels))

        pano_cam_root = f'{room_path}{__class__.PANO_CAM_PREFIX}'
        cam_paths = [_c for _c in zip_reader.namelist() if _c.find(pano_cam_root) != -1 and \
            _c.find(__class__.PANO_CAM_FILE) != -1]
        for cam_path in cam_paths:
            pano_root = cam_path[:cam_path.rfind('/')]
            pano_root = pano_root[:pano_root.rfind('/')]
            pano_root = f'{pano_root}{__class__.PANORAMIC_PREFIX}'
            cam_paras, attr_images = __class__.read_camera_and_image(zip_reader, cam_path, 15, \
                pano_root)
            r_points, r_colors, r_labels = __class__.view2points_pano(cam_paras, attr_images)
            all_infos.append((r_points, r_colors, r_labels))

        a_points = np.concatenate([_i[0] for _i in all_infos], axis=0)
        a_colors = np.concatenate([_i[1] for _i in all_infos], axis=0)
        a_labels = np.concatenate([_i[2] for _i in all_infos], axis=0)

        a_points = a_points[..., [2, 0, 1]]  # Convert Y-top to Z-top
        return a_points, a_colors, a_labels

    @staticmethod
    def _read_instance_infos(zip_reader: g_io.GroupZipIO, room_path: str, \
        points: np.ndarray, labels: np.ndarray, min_pts=50) -> Dict:
        scene_id, _, _  = room_path.split('/')
        try:
            anno_file, = [_f for _f in zip_reader.namelist() if \
                _f.find(f'{scene_id}/{__class__.ANNO_FILE}') != -1]
        except ValueError:
            return None
        boxes_info: List[Dict] = json.loads(zip_reader.read(anno_file))

        anno_infos = Annotations()
        rb_idx = 0  # room bounding box ID
        for box_info in boxes_info:
            # b_id = int(box_info['ID'])
            centroid = np.asarray(box_info['centroid'], dtype=np.float32) / 1000
            coeffs = np.asarray(box_info['coeffs'], dtype=np.float32) / 1000
            basis = np.asarray(box_info['basis'], dtype=np.float32)
            obb_8pts = S3DUtilize.get_8points_bounding_box(basis, coeffs, centroid)

            box_min = np.min(obb_8pts, axis=0, keepdims=True)
            box_max = np.max(obb_8pts, axis=0, keepdims=True)

            point_max_mask = np.all(points < box_max, axis=1)
            point_min_mask = np.all(points > box_min, axis=1)
            point_mask = np.logical_and(point_max_mask, point_min_mask)
            box_points: np.ndarray = points[point_mask]
            if box_points.size < min_pts:
                continue

            box_instances = labels[point_mask][..., 0]
            instance_id, instance_count = np.unique(box_instances, return_counts=True)
            instance_id = instance_id[np.argmax(instance_count)]

            instance_points = box_points[box_instances == instance_id]
            ip_box_min = np.min(instance_points, axis=0)
            ip_box_max = np.max(instance_points, axis=0)
            dimension = np.maximum(centroid - ip_box_min, ip_box_max - centroid)

            ur_depth = np.concatenate([centroid, dimension * 2], axis=0)

            anno_infos.index.append(rb_idx)
            anno_infos.classes.append(instance_id)
            anno_infos.name.append(NYU40.index_to_label(instance_id))
            anno_infos.location.append(centroid)
            anno_infos.dimensions.append(dimension)
            anno_infos.gt_boxes_upright_depth.append(ur_depth)
            anno_infos.unaligned_location.append(centroid)
            anno_infos.unaligned_dimensions.append(dimension)
            anno_infos.unaligned_gt_boxes_upright_depth.append(ur_depth)

            rb_idx += 1
        anno_infos.gt_num = rb_idx
        anno_infos.axis_align_matrix = np.eye(4, dtype=np.float64)
        return anno_infos.dump()

    def _mp_format_dataset(self, rooms_list: List[str], proc_unit: ProcessUnit,\
        start_index=0, worker_id=0):
        del start_index, worker_id
        zip_reader = self._load_zips()

        points_folder = self.envs.get_env_path(proc_unit.out_paths[0])
        os.makedirs(points_folder, exist_ok=True)
        semantics_folder = self.envs.get_env_path(proc_unit.out_paths[1])
        os.makedirs(semantics_folder, exist_ok=True)
        instance_folder = self.envs.get_env_path(proc_unit.out_paths[2])
        os.makedirs(instance_folder, exist_ok=True)
        annotation_folder = self.envs.get_env_path(proc_unit.out_paths[3])
        os.makedirs(annotation_folder, exist_ok=True)

        for _, room_path in enumerate(rooms_list):
            scene_id, _, room_id = room_path.split('/')
            dump_name = f'{scene_id}_{room_id}_1cm.bin'
            points_path = os.path.join(points_folder, dump_name)
            semantics_path = os.path.join(semantics_folder, dump_name)
            instance_path = os.path.join(instance_folder, dump_name)
            annotation_path = os.path.join(annotation_folder, dump_name)
            if np.all([os.path.exists(_path) for _path in \
                [points_path, semantics_path, annotation_path]]):
                continue

            # Step 1: Read images and make point clouds
            a_points, a_colors, a_labels = self._view2points(zip_reader, room_path)
            v_points, v_colors, v_labels = self._points2voxel((a_points, a_colors, \
                a_labels), 0.01)
            if v_points is None:
                print(f'Ignore {room_path} with invalid points')
                continue
            # Step 2: Read bounding box information
            anno_infos = self._read_instance_infos(zip_reader, room_path, \
                v_points, v_labels)
            if anno_infos is None:
                print(f'Ignore {room_path} with invalid annotations')
                continue

            np.concatenate([v_points.astype(np.float32), v_colors.astype(np.float32)],\
                 axis=-1).tofile(points_path)
            v_labels.astype(np.int64).tofile(semantics_path)
            with open(annotation_path, 'wb') as a_fp:
                pickle.dump(anno_infos, a_fp)

    def format_dataset(self, proc_unit: ProcessUnit):
        attrs = proc_unit.attrs

        desc_dir = os.path.join(self.envs.out_data_root, 'desc')
        os.makedirs(desc_dir, exist_ok=True)
        with open(os.path.join(desc_dir, proc_unit.out_paths[0]), 'wb') as b_fp:
            pickle.dump(np.zeros([0, 6], np.float32), b_fp)
        with open(os.path.join(desc_dir, proc_unit.out_paths[1]), 'wb') as b_fp:
            pickle.dump(np.zeros([0], np.int64), b_fp)

        self._zip_folder = self.envs.get_env_path(proc_unit.in_paths[0])

        rooms_list = self._get_rooms_list_by_types(attrs['room_types'])

        g_perf.multiple_processor(self._mp_format_dataset, rooms_list, 8, \
            (proc_unit, ))
