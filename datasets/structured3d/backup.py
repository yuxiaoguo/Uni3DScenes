"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
import io
import cv2
import json
import logging
import numpy as np
from typing import List
from copy import deepcopy

from .. import data as g_data
from ..utils import BaseDataGen, ProcessPipeline, AxisAlignBoundingBox
from graphics_utils import g_io, g_cfg, g_math, g_str, g_perf


class SemanticsPlane(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.ID: int = int(0)
        self.planeID: List[int] = list([int(0)])
        self.type: str = str()


class Junction(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.ID: int = int(0)
        self.coordinate: List[float] = list([float(0)])


class Line(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.ID: int = int(0)
        self.point: List[float] = list([float(0)])
        self.direction: List[float] = list([float(0)])


class Plane(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.offset: float = float(0)
        self.type: str = str()
        self.ID: int = int(0)
        self.normal: List[float] = list([float(0)])


class Annotation3D(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.junctions: List[Junction] = list([Junction()])
        self.lines: List[Line] = list([Line()])
        self.planes: List[Plane] = list([Plane()])
        self.planeLineMatrix: List[List[int]] = list([list([0])])
        self.lineJunctionMatrix: List[List[int]] = list([list([0])])
        self.semantics: List[SemanticsPlane] = list([SemanticsPlane()])

    def get_semantics_by_room_id(self, room_id):
        for k_id, k_ in enumerate(self.semantics):
            if k_.ID == room_id:
                return k_
        return None

    def get_rooms_by_type(self, r_type) -> List[int]:
        room_list = list()
        for k_id, k_ in enumerate(self.semantics):
            if k_.type != r_type:
                continue
            room_list.append(k_.ID)
        return room_list

    def get_semantic_bounding_box(self, room_id) -> AxisAlignBoundingBox:
        planes_id = self.get_semantics_by_room_id(int(room_id)).planeID
        planes = [p_ for p_ in self.planes if p_.ID in planes_id]
        plane_lines_matrix = np.asarray(self.planeLineMatrix)
        lines_id = [np.argwhere(plane_lines_matrix[p_.ID])[..., 0] for p_ in planes]
        lines_id = np.unique(np.concatenate(lines_id))
        line_junctions_matrix = np.asarray(self.lineJunctionMatrix)
        junctions_id = [np.argwhere(line_junctions_matrix[l_])[..., 0] for l_ in lines_id]
        junctions_id = np.unique(np.concatenate(junctions_id))
        junctions = [j_ for j_ in self.junctions if j_.ID in junctions_id]
        points = [p_.coordinate for p_ in junctions]
        semantic_box = AxisAlignBoundingBox()
        semantic_box.assign_box_size(np.max(points, axis=0).tolist(), np.min(points, axis=0).tolist())
        return semantic_box


class S3DUtilize(object):
    @staticmethod
    def get_fov_normal(image_size, cam_focal, norm=True):
        u2x, v2y = [(np.arange(1, image_size[a_i] + 1) - image_size[a_i] / 2) / cam_focal[a_i] for a_i in [0, 1]]
        cam_m_u2x = np.tile([u2x], (image_size[1], 1))
        cam_m_v2y = np.tile(v2y[:, np.newaxis], (1, image_size[0]))
        cam_m_depth = np.ones(image_size).T
        fov_normal = np.stack((cam_m_depth, -1 * cam_m_v2y, cam_m_u2x), axis=-1)
        if norm:
            fov_normal = fov_normal / np.sqrt(np.sum(np.square(fov_normal), axis=-1, keepdims=True))
        return fov_normal

    @staticmethod
    def cast_perspective_to_local_coord(depth_img: np.ndarray, fov_normal):
        return np.expand_dims(depth_img, axis=-1) * fov_normal

    @staticmethod
    def cast_points_to_voxel(points, labels, room_size=(6.4, 3.2, 6.4), room_stride=0.2):
        vol_resolution = (np.asarray(room_size) / room_stride).astype(np.int32)
        vol_index = np.floor(points / room_stride).astype(np.int32)
        in_vol = np.logical_and(np.all(vol_index < vol_resolution, axis=1), np.all(vol_index >= 0, axis=1))
        x, y, z = [d_[..., 0] for d_ in np.split(vol_index[in_vol], 3, axis=-1)]
        vol_label = labels[in_vol]
        vol_data = np.zeros(vol_resolution, dtype=np.uint8)
        vol_data[x, y, z] = vol_label
        return vol_data

    @staticmethod
    def get_rotation_matrix_from_tu(cam_front, cam_up):
        cam_n = np.cross(cam_front, cam_up)
        cam_m = np.stack((cam_front, cam_up, cam_n), axis=1).astype(np.float32)
        return cam_m


class Structured3DDataGen(BaseDataGen):
    def __init__(self, data_dir, out_dir, process_pipelines, cfg=None, **kargs):
        super().__init__(data_dir, out_dir, process_pipelines, **kargs)
        self.cfg = cfg if cfg is not None else process_pipelines[0]
        room_size = np.insert(self.cfg.room_size, 1, self.cfg.room_height)
        self.room_size, self.room_stride = np.array(room_size), self.cfg.room_stride
        self.room_center = self.room_size * [0.5, 0, 0.5]
        self.vox_size = (self.room_size / self.room_stride).astype(np.int32)
        self.depth_scale = np.linalg.norm(self.room_size * 0.75)
        logging.info(f'Depth scale: [0, {self.depth_scale}] for a camera located in 0.75 room space')

        self.label_type, self.data_label, self.label_list, self.color_map = self.get_label_info(self.cfg)

        self.fov_n = None
        self.nyu40, self.nyu40_label, self.nyu40_color_map = None, None, None
        self.select_nyu_label_id, self.label_mapping, self.category_mapping = None, None, None
        self.init_config()

    def init_config(self):
        self.nyu40 = g_data.NYU40()
        self.nyu40_label = self.nyu40.label_id_map_arr()
        select_nyu_label = self.label_list + ['desk'] if 'living' in self.label_type else self.label_list
        self.select_nyu_label_id = [self.nyu40_label.index(s_l) for s_l in select_nyu_label]
        self.nyu40_color_map = self.nyu40.color_map_arr()
        self.category_mapping = np.zeros(len(self.nyu40_label), dtype=np.uint8)
        for s_i, s_l in enumerate(self.select_nyu_label_id):
            self.category_mapping[s_l] = s_i
        if 'living' in self.label_type:
            self.category_mapping[self.nyu40_label.index('desk')] = self.label_list.index('table')

        image_size = np.array([1280, 720], np.int32)
        cam_half_fov = np.array([0.698132, 0.440992])
        self.fov_n = S3DUtilize.get_fov_normal(image_size, image_size / 2 / np.tan(cam_half_fov))

    def load_zips(self, filter_regex='Structured3D') -> g_io.GroupZipIO:
        ctx_files = [f for f in os.listdir(self.data_dir) if filter_regex in f and f.endswith('zip')]
        zip_reader = g_io.GroupZipIO([os.path.join(self.data_dir, f) for f in ctx_files])
        return zip_reader

    @staticmethod
    def read_file_from_zip(zip_reader, scene_id, file_, filter_regex='Structured3D'):
        ctx = zip_reader.read('/'.join((filter_regex, scene_id, file_)))
        return io.BytesIO(ctx)

    def load_scene_anno_from_zip(self, zip_reader, scene_id: str):
        anno_3d = Annotation3D()
        anno_3d.load(json.load(self.read_file_from_zip(zip_reader, scene_id, 'annotation_3d.json')))
        return anno_3d

    def get_room_box_from_zip(self, zip_reader, scene_id: str, room_id: str, src_room=False):
        scene_anno = self.load_scene_anno_from_zip(zip_reader, scene_id)
        room_box_src = scene_anno.get_semantic_bounding_box(room_id)
        room_box = deepcopy(room_box_src)
        room_box.scale(1 / 1000)
        z2y_top_m = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        room_box.rotation(z2y_top_m)
        return (room_box_src, room_box) if src_room else room_box

    def assemble_semantic_points_from_img(self, depth_img, semantic_pano, cos_threshold=0.15):
        points = S3DUtilize.cast_perspective_to_local_coord(depth_img, self.fov_n)
        points_normal = g_math.normal_from_cross_product(points)
        view_dist = np.maximum(np.linalg.norm(points, axis=-1, keepdims=True), float(10e-5))
        cosine_dist = np.sum((points * points_normal / view_dist), axis=-1)
        cosine_dist = np.abs(cosine_dist)
        point_valid = np.logical_and(cosine_dist > cos_threshold, depth_img < 65535)
        label_valid = semantic_pano > 0
        all_valid = np.logical_and(point_valid, label_valid)
        return points[all_valid], semantic_pano[all_valid]

    @staticmethod
    def assemble_semantic_points_from_panorama(depth_pano, semantic_pano):
        pano_h, pano_w = depth_pano.shape[:2]
        p_a, p_b = np.arange(pano_w) / pano_w * 2*np.pi - np.pi, np.arange(pano_h) / pano_h * np.pi * -1 + np.pi/2
        p_a, p_b = np.tile(p_a[None], [pano_h, 1]), np.tile(p_b[:, None], [1, pano_w])
        p_a_sin, p_a_cos, p_b_sin, p_b_cos = np.sin(p_a), np.cos(p_a), np.sin(p_b), np.cos(p_b)
        point_x = depth_pano * p_a_cos * p_b_cos
        point_y = depth_pano * p_b_sin
        point_z = depth_pano * p_a_sin * p_b_cos
        points = np.stack([point_x, point_y, point_z], axis=-1)
        all_valid = np.logical_and(depth_pano < 65535, semantic_pano > 0)
        return points[all_valid], semantic_pano[all_valid]

    def get_all_rooms_by_type(self, room_type):
        room_list_path = os.path.join(self.out_assemble_dir, f'{room_type}_list')
        if os.path.exists(room_list_path):
            with open(room_list_path, 'r') as fp:
                room_list = [f.rstrip() for f in fp.readlines()]
        else:
            invalid_scene_list = [
                'scene_01155', 'scene_01714', 'scene_01816', 'scene_03398', 'scene_01192', 'scene_01852'
            ]
            wrong_scene_room = [
                'scene_01778_room_858455', 'scene_00010_room_846619', 'scene_00043_room_1518', 'scene_00043_room_3128',
                'scene_00043_room_474', 'scene_00043_room_732', 'scene_00043_room_856', 'scene_00173_room_4722',
                'scene_00240_room_384', 'scene_00325_room_970753', 'scene_00335_room_686', 'scene_00339_room_2193',
                'scene_00501_room_1840', 'scene_00515_room_277475', 'scene_00543_room_176', 'scene_00587_room_9914',
                'scene_00703_room_762455', 'scene_00703_room_771712', 'scene_00728_room_5662',
                'scene_00828_room_607228',
                'scene_00865_room_1026', 'scene_00865_room_1402', 'scene_00875_room_739214', 'scene_00917_room_188',
                'scene_00917_room_501284', 'scene_00926_room_2290', 'scene_00936_room_311', 'scene_00937_room_1955',
                'scene_00986_room_141', 'scene_01009_room_3234', 'scene_01009_room_3571', 'scene_01021_room_689126',
                'scene_01034_room_222021', 'scene_01036_room_301', 'scene_01043_room_2193', 'scene_01104_room_875',
                'scene_01151_room_563', 'scene_01165_room_204', 'scene_01221_room_26619', 'scene_01222_room_273364',
                'scene_01282_room_1917', 'scene_01282_room_24057', 'scene_01282_room_2631', 'scene_01400_room_10576',
                'scene_01445_room_3495', 'scene_01470_room_1413', 'scene_01530_room_577', 'scene_01670_room_291',
                'scene_01745_room_342', 'scene_01759_room_3584', 'scene_01759_room_3588', 'scene_01772_room_897997',
                'scene_01774_room_143', 'scene_01781_room_335', 'scene_01781_room_878137', 'scene_01786_room_5837',
                'scene_01916_room_2648', 'scene_01993_room_849', 'scene_01998_room_54762', 'scene_02034_room_921879',
                'scene_02040_room_311', 'scene_02046_room_1014', 'scene_02046_room_834', 'scene_02047_room_934954',
                'scene_02101_room_255228', 'scene_02172_room_335', 'scene_02235_room_799012', 'scene_02274_room_4093',
                'scene_02326_room_836436', 'scene_02334_room_869673', 'scene_02357_room_118319',
                'scene_02484_room_43003',
                'scene_02499_room_1607', 'scene_02499_room_977359', 'scene_02509_room_687231',
                'scene_02542_room_671853',
                'scene_02564_room_702502', 'scene_02580_room_724891', 'scene_02650_room_877946',
                'scene_02659_room_577142',
                'scene_02690_room_586296', 'scene_02706_room_823368', 'scene_02788_room_815473',
                'scene_02889_room_848271',
                'scene_03035_room_631066', 'scene_03120_room_830640', 'scene_03327_room_315045',
                'scene_03376_room_800900',
                'scene_03399_room_337', 'scene_03478_room_2193'
            ]

            room_list = list()
            data_zip_meta = self.load_zips()
            scene_list = [c.split('/')[1] for c in data_zip_meta.namelist() if 'annotation_3d.json' in c]
            for scene_id in scene_list:
                if scene_id in invalid_scene_list:
                    continue
                scene_anno = self.load_scene_anno_from_zip(data_zip_meta, scene_id)
                room_ids = scene_anno.get_rooms_by_type(room_type)
                room_ids_filter = [r_i for r_i in room_ids if f'{scene_id}_room_{r_i}' not in wrong_scene_room]
                room_list.extend([f'{scene_id}/2D_rendering/{r_i}' for r_i in room_ids_filter])
            with open(room_list_path, 'w') as fp:
                fp.writelines('\n'.join(room_list))
            data_zip_meta.close()
        return room_list


class Structured3DPointCloudDataGen(Structured3DDataGen):
    def __init__(self, data_dir, out_dir, process_pipelines, cfg=None, **kargs):
        super().__init__(data_dir, out_dir, process_pipelines, cfg, **kargs)

    def load_camera_and_image(self, zip_meta, cam_path, info_type=0, image_size=None, depth_map=True,
                              pano_cam_path=None):
        # 1-camera, 2-color, 4-depth, 8-semantic, 16-instance
        cam_r, cam_t, color_img, depth_img, label_img, inst_img, img_shape = None, None, None, None, None, None, None
        # load camera_pose
        if info_type & 2**0:
            z2y_top_m = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            if pano_cam_path:
                camera_para = io.BytesIO(zip_meta.read(pano_cam_path)).read().decode('utf-8').split(' ')
                camera_para = np.asarray([float(i_) for i_ in camera_para], np.float32)
            else:
                camera_para = io.BytesIO(zip_meta.read(cam_path)).read().decode('utf-8').split(' ')
                assert np.all(np.array(camera_para[-3:-1]) == np.array(['0.698132', '0.440992']))
                camera_para = np.asarray([float(i_) for i_ in camera_para], np.float32)
                cam_r = S3DUtilize.get_rotation_matrix_from_tu(camera_para[3:6], camera_para[6:9])
                cam_r = np.matmul(z2y_top_m, cam_r)
            cam_t = np.matmul(z2y_top_m, camera_para[:3] / 1000)

        # load color image
        if info_type & 2**1:
            color_path = cam_path.replace('camera_pose.txt', 'rgb_rawlight.png')
            color_img_data = np.frombuffer(io.BytesIO(zip_meta.read(color_path)).read(), np.uint8)
            color_img = cv2.imdecode(color_img_data, cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1]

        # load depth image
        if info_type & 2**2:
            depth_path = cam_path.replace('camera_pose.txt', 'depth.png')
            depth_img_data = np.frombuffer(io.BytesIO(zip_meta.read(depth_path)).read(), np.uint8)
            depth_img = cv2.imdecode(depth_img_data, cv2.IMREAD_UNCHANGED)
            if depth_map:
                depth_img[depth_img == 0] = 65535

        # load semantic image
        if info_type & 2**3:
            semantic_path = cam_path.replace('camera_pose.txt', 'semantic.png')
            semantic_img_data = np.frombuffer(io.BytesIO(zip_meta.read(semantic_path)).read(), np.uint8)
            semantic_img = cv2.imdecode(semantic_img_data, cv2.IMREAD_UNCHANGED)[..., ::-1]
            label_img = np.zeros(semantic_img.shape[:2], dtype=np.uint8)
            for l_id in range(len(self.nyu40_label)):
                color = np.asarray(g_data.NYU40().color_map(l_id), dtype=np.uint8)
                label_img[np.all(semantic_img == color, axis=-1)] = l_id

        # load instance image
        if info_type & 2**4:
            instance_path = cam_path.replace('camera_pose.txt', 'instance.png')
            inst_img_data = np.frombuffer(io.BytesIO(zip_meta.read(instance_path)).read(), np.uint8)
            inst_img = cv2.imdecode(inst_img_data, cv2.IMREAD_UNCHANGED)

        return cam_r, cam_t, color_img, depth_img, label_img, inst_img

    def generate_point_cloud(self, process_pipeline: ProcessPipeline):
        for room_type in process_pipeline.room_types:
            room_list = self.get_all_rooms_by_type(room_type)

            zip_meta = self.load_zips()
            # out_dir = g_str.mkdir_automated(os.path.join(self.out_assemble_dir, f'{room_type}_scene_point_cloud'))
            vis_dir = g_str.mkdir_automated(os.path.join(self.out_assemble_dir, f'{room_type}_scene_point_cloud_vis'))

            for r_i, room_path in enumerate(room_list):
                if r_i % 100 == 0:
                    logging.info(f'{r_i}th/{len(room_list)}')
                cam_path_list = [c for c in zip_meta.namelist() if room_path + '/perspective/full' in c and 'camera_pose.txt' in c]
                if len(cam_path_list) == 0:
                    continue

                scene_id, _, room_id = room_path.split('/')

                view_list, point_list, label_list, cam_t_list, cam_r_list = list(), list(), list(), list(), list()
                for cam_path in cam_path_list:
                    _, scene_id, _, room_id, _, _, view_id, _ = cam_path.split('/')
                    room_view_id = '%s-room_%s-view_%03d' % (scene_id, room_id, int(view_id))

                    cam_r, cam_t, color_img, depth_img, label_img, _ = \
                        self.load_camera_and_image(zip_meta, cam_path, info_type=15)
                    r_points, r_labels = self.assemble_semantic_points_from_img(depth_img, label_img)
                    if len(r_points) == 0:
                        continue
                    r_points = np.matmul(r_points / 1000, cam_r.T).astype(np.float32) + cam_t

                    point_list.append(r_points)
                    label_list.append(r_labels)
                    if r_i < 16:
                        vis_path = os.path.join(vis_dir, f'{room_view_id}')
                        self.visualize_multi_views_images(vis_path, color_img, depth_img, label_img, self.nyu40_color_map)

                # read panorama
                semantic_pano_path_list = [c for c in zip_meta.namelist() if f'{room_path}/panorama/full/semantic.png' in c]
                if len(semantic_pano_path_list) == 0:
                    logging.warning(f'{room_path}: no view')
                    continue
                for v_i, semantic_pano_path in enumerate(semantic_pano_path_list):
                    cam_path = semantic_pano_path.replace('semantic.png', 'camera_pose.txt')
                    pano_cam_path = cam_path.replace('full/camera_pose.txt', 'camera_xyz.txt')

                    # 1-camera, 2-color, 4-depth, 8-semantic
                    pano_cam_r, pano_cam_t, color_pano, depth_pano, nyu_label_pano, _ = \
                        self.load_camera_and_image(zip_meta, cam_path, info_type=15, pano_cam_path=pano_cam_path)
                    label_pano = nyu_label_pano
                    if r_i < 16:
                        vis_path = os.path.join(vis_dir, f'{scene_id}-room_{room_id}-panorama{v_i}')
                        self.visualize_multi_views_images(vis_path + '_panorama', color_pano, depth_pano, label_pano,
                                                          self.nyu40_color_map)

                    pano_points, pano_labels = self.assemble_semantic_points_from_panorama(depth_pano, label_pano)
                    pano_points = pano_points.astype(np.float32) / 1000 + pano_cam_t
                    point_list.append(pano_points)
                    label_list.append(pano_labels)

                if len(point_list) == 0:
                    continue

                point_list, label_list = np.concatenate(point_list), np.concatenate(label_list)
                if r_i < 16 and vis_dir is not None:
                    vis_path = os.path.join(vis_dir, '%s-room_%s.ply' % (scene_id, room_id))
                    g_io.PlyIO().dump_points(vis_path, point_list, label_list, colors_map=self.nyu40_color_map)

    def mediate_process(self):
        data_dir, out_dir = self.data_dir, self.out_dir
        process_pipelines = self.process_pipelines
        for p_p in process_pipelines:
            self.__init__(data_dir, os.path.join(out_dir, p_p.label_type), self.process_pipelines, p_p)
            self.generate_point_cloud(p_p)
