"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
import io
import re
import cv2
import json
import copy
import imageio
import logging
import trimesh
import numpy as np
import pandas as pd
from typing import Tuple
from scipy import ndimage
from abc import abstractmethod

from .. import data as g_data
from examples.scenegen.tools.analyzer import visualization_utilize
from graphics_utils import g_cfg, g_io, g_str, g_perf, g_plt


class Perspective2Equirectangular(object):
    @staticmethod
    def get_equirectangular(in_image, in_theta, in_phi, in_fov=(90, 90), out_resolution=(64, 256), out_fov=(90, 360),
                            interpolation=cv2.INTER_CUBIC, vis_path=None):
        # in_fov=(50, 80)
        # out_resolution=(128, 256), out_fov=(180, 360)
        # THETA is left/right angle, PHI is up/down angle, both in degree

        in_fov_h, in_fov_w = in_fov
        in_height, in_width = in_image.shape[:2]
        w_len = np.tan(np.radians(in_fov_w / 2.0))
        h_len = np.tan(np.radians(in_fov_h / 2.0))

        # x, y = np.meshgrid(np.linspace(-180, 180, out_width), np.linspace(90, -90, out_height))
        out_height, out_width = out_resolution
        out_fov_h, out_fov_w = out_fov
        x, y = np.meshgrid(np.linspace(-out_fov_w//2, out_fov_w//2, out_width),
                           np.linspace(out_fov_h//2, -out_fov_h//2, out_height))

        # z top
        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map, y_map, z_map), axis=2).reshape([out_height * out_width, 3])
        theta_sin, theta_cos = np.sin(np.radians(in_theta)), np.cos(np.radians(in_theta))
        phi_sin, phi_cos = np.sin(np.radians(-in_phi)), np.cos(np.radians(-in_phi))
        R1 = np.array([[theta_cos, -theta_sin, 0], [theta_sin, theta_cos, 0], [0, 0, 1]], dtype=np.float32)
        R1 = np.round(R1, decimals=15)
        R2 = np.array([[phi_cos, 0, phi_sin], [0, 1, 0], [-phi_sin, 0, phi_cos]], dtype=np.float32)
        R2 = np.round(R2, decimals=15)
        R3 = np.matmul(R1, R2)
        R3 = np.round(R3, decimals=15)
        xyz = np.matmul(np.linalg.inv(R3), xyz.T).T
        # # g_io.PlyIO().dump_points(vis_path + '_pano_points_R3.ply', xyz)

        xyz = xyz.reshape([out_height, out_width, 3])
        inverse_mask = np.where(xyz[:, :, 0] > 0, 1, 0)

        xyz[:, :] = xyz[:, :] / np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)
        xyz_valid = (-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) & (-h_len < xyz[:, :, 2]) & (xyz[:, :, 2] < h_len)
        lon_map = np.where(xyz_valid, xyz[:, :, 1] * in_width / 2 / w_len + in_width / 2, 0)
        lat_map = np.where(xyz_valid, -xyz[:, :, 2] * in_height / 2 / h_len + in_height / 2 , 0)
        mask = np.where(xyz_valid, 1, 0)

        persp = cv2.remap(in_image, lon_map.astype(np.float32), lat_map.astype(np.float32), interpolation,
                          borderMode=cv2.BORDER_WRAP)
        persp = persp[..., None] if persp.ndim == 2 else persp

        mask = mask * inverse_mask
        mask = np.broadcast_to(mask[:, :, np.newaxis], persp.shape)
        persp = persp * mask

        return persp, mask

    @staticmethod
    def get_equirectangular_from_images(in_image_list, in_fov=(90, 90), out_resolution=(64, 256), out_fov=(90, 360),
                                        interpolation=cv2.INTER_CUBIC, vis_path=None):
        in_image_list = [img[..., None] if img.ndim == 2 else img for img in in_image_list]
        assert np.all([img.ndim == 3 for img in in_image_list]), f'in images: {[img.shape for img in in_image_list]}'

        num_image, image_channel = len(in_image_list), in_image_list[0].shape[-1]
        out_height, out_width = out_resolution
        pano_image = np.zeros((out_height, out_width, image_channel))
        pano_mask = np.zeros((out_height, out_width, image_channel))

        # THETA is left/right angle, PHI is up/down angle, both in degree
        if num_image == 4:
            in_theta_list, in_phi_list = np.arange(num_image)/num_image * 360, [0] * num_image
        elif num_image == 6:
            in_theta_list, in_phi_list = [0, 90, 180, 270, 0, 0], [0, 0, 0, 0, -90, 90]
        else:
            raise NotImplementedError

        c_args = dict(in_fov=in_fov, out_resolution=out_resolution, out_fov=out_fov, interpolation=interpolation)
        for img_i, in_image, in_theta, in_phi in zip(list(range(num_image)), in_image_list, in_theta_list, in_phi_list):
            c_args['vis_path'] = vis_path[:-4] + f'_p{img_i}' if vis_path is not None else vis_path
            out_image, out_mask = Perspective2Equirectangular.get_equirectangular(in_image, in_theta, in_phi, **c_args)
            pano_image += out_image
            pano_mask += out_mask

        pano_mask = np.where(pano_mask == 0, 1, pano_mask)
        pano_image = (np.divide(pano_image, pano_mask))
        return pano_image


class Equirectangular2Perspective(object):
    @staticmethod
    def get_perspective_image_from_panorama(in_panorama, out_theta, out_phi, out_fov=(90, 90), out_resolution=(64, 64),
                                            interpolation=cv2.INTER_CUBIC):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        height, width = out_resolution
        equ_h, equ_w = in_panorama.shape[:2]
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        hFOV, wFOV = out_fov
        # wFOV = FOV
        # hFOV = float(height) / width * wFOV

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))

        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])
        z_map = -np.tile(np.linspace(-h_len, h_len, height), [width, 1]).T

        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(D[:, :, np.newaxis], 3, axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(out_theta))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-out_phi))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2])
        lon = np.arctan2(xyz[:, 1], xyz[:, 0])

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180

        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        persp = cv2.remap(in_panorama, lon.astype(np.float32), lat.astype(np.float32), interpolation,
                          borderMode=cv2.BORDER_WRAP)
        return persp

    @staticmethod
    def get_rotated_view_from_panorama(in_panorama, image_size=(64, 64), num_view=36, interpolation=cv2.INTER_AREA):
        view_hq_list, view_list = list(), list()
        persp_args = dict(out_phi=0, out_fov=(90, 90), out_resolution=[256, 256], interpolation=interpolation)
        for v_i in range(num_view):
            view_hq = Equirectangular2Perspective.get_perspective_image_from_panorama(in_panorama, 360/num_view * v_i, **persp_args)
            view_lq = cv2.resize(view_hq, image_size, interpolation=interpolation)
            view_hq_list.append(view_hq)
            view_list.append(view_lq)
        return np.stack(view_hq_list, axis=0), np.stack(view_list, axis=0)

    @staticmethod
    def get_cube_images_from_panorama(in_panorama, out_theta=0, out_phi=0, out_resolution=(64, 64), out_fov=(90, 90),
                                      interpolation=cv2.INTER_CUBIC, top_down=False):
        cube_images = list()
        persp_args = dict(out_fov=out_fov, out_resolution=out_resolution, interpolation=interpolation)

        for c_i in range(4):
            persp_img = Equirectangular2Perspective.get_perspective_image_from_panorama(
                in_panorama, out_theta + c_i * 90, out_phi, **persp_args)
            cube_images.append(persp_img)

        if top_down:
            persp_img = Equirectangular2Perspective.get_perspective_image_from_panorama(
                in_panorama, out_theta, 90, **persp_args)
            cube_images.append(persp_img)
            persp_img = Equirectangular2Perspective.get_perspective_image_from_panorama(
                in_panorama, out_theta, -90, **persp_args)
            cube_images.append(persp_img)

        return np.stack(cube_images, axis=0)


class ViewImageGenUtil(object):
    @staticmethod
    def xy_rotation(phi, vec):
        rotate_matrix = np.array([[np.cos(phi), -np.sin(phi), 0],
                                  [np.sin(phi), np.cos(phi), 0],
                                  [0, 0, 1]], dtype=np.float32)
        rotate_matrix = np.round(rotate_matrix, decimals=15)
        vec_rotate = np.matmul(rotate_matrix, vec)
        return vec_rotate

    @staticmethod
    def zx_rotation(phi, vec):
        rotate_matrix = np.array([[np.cos(phi), 0, np.sin(phi)],
                                  [0, 1, 0],
                                  [-np.sin(phi), 0, np.cos(phi)]], dtype=np.float32)
        rotate_matrix = np.round(rotate_matrix, decimals=15)
        vec_rotate = np.matmul(rotate_matrix, vec)
        return vec_rotate

    @staticmethod
    def yz_rotation(phi, vec):
        rotate_matrix = np.array([[1, 0, 0],
                                  [0, np.cos(phi), -np.sin(phi)],
                                  [0, np.sin(phi), np.cos(phi)]], dtype=np.float32)
        rotate_matrix = np.round(rotate_matrix, decimals=15)
        vec_rotate = np.matmul(rotate_matrix, vec)
        return vec_rotate

    @staticmethod
    def rotation_matrix_from_angle_x(angle_y):
        angle_cos, angle_sin = np.cos(angle_y), np.sin(angle_y)
        cam_r_zeros, cam_r_ones = np.zeros_like(angle_cos), np.ones_like(angle_cos)
        cam_r_f = np.stack([cam_r_ones, cam_r_zeros, cam_r_zeros], 1)
        cam_r_s = np.stack([cam_r_zeros, angle_cos, angle_sin * -1], 1)
        cam_r_t = np.stack([cam_r_zeros, angle_sin, angle_cos], 1)
        cam_r = np.stack([cam_r_f, cam_r_s, cam_r_t], 1)
        cam_r = np.round(cam_r, decimals=15)
        return cam_r

    @staticmethod
    def rotation_matrix_from_angle_y(angle_y):
        angle_cos, angle_sin = np.cos(angle_y), np.sin(angle_y)
        cam_r_zeros, cam_r_ones = np.zeros_like(angle_cos), np.ones_like(angle_cos)
        cam_r_f = np.stack([angle_cos, cam_r_zeros, angle_sin], 1)
        cam_r_s = np.stack([cam_r_zeros, cam_r_ones, cam_r_zeros], 1)
        cam_r_t = np.stack([angle_sin * -1, cam_r_zeros, angle_cos], 1)
        cam_r = np.stack([cam_r_f, cam_r_s, cam_r_t], 1)
        cam_r = np.round(cam_r, decimals=15)
        return cam_r

    @staticmethod
    def rotation_matrix_from_angle_z(angle_z):
        angle_cos, angle_sin = np.cos(angle_z), np.sin(angle_z)
        cam_r_zeros, cam_r_ones = np.zeros_like(angle_cos), np.ones_like(angle_cos)
        cam_r_f = np.stack([angle_cos, angle_sin * -1, cam_r_zeros], 1)
        cam_r_s = np.stack([angle_sin, angle_cos, cam_r_zeros], 1)
        cam_r_t = np.stack([cam_r_zeros, cam_r_zeros, cam_r_ones], 1)
        cam_r = np.stack([cam_r_f, cam_r_s, cam_r_t], 1)
        cam_r = np.round(cam_r, decimals=15)
        return cam_r

    @staticmethod
    def to_transformation_matrix(rot_m, loc):
        trans_m = np.eye(4)
        trans_m[:3, :3] = rot_m
        trans_m[:3, 3] = loc
        trans_m = trans_m.T
        return trans_m

    @staticmethod
    def exr2numpy(exr_file, channel_name):
        file = OpenEXR.InputFile(exr_file)
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_str = file.channel(channel_name, Float_Type)
        channel = np.fromstring(channel_str, dtype=np.float32).reshape(size[1], -1)
        return channel

    @staticmethod
    def encode2srgb(v):
        return np.clip((np.where(v <= 0.0031308, v * 12.92, 1.055 * (v ** (1.0 / 2.4)) - 0.055)), 0, 1)

    @staticmethod
    def exr2color(exr_file):
        color_img = np.dstack([ViewImageGenUtil.encode2srgb(ViewImageGenUtil.exr2numpy(exr_file, channel_name))
                               for channel_name in ['R', 'G', 'B']])
        return color_img


class ProcessPipeline(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.label_type = str()
        self.assemble_name = str()
        self.room_types = list([str()])
        self.ndims = int(3)
        # 1-label, 2-ins, 4-center offset, 8-bbox size, 16-rotation
        self.info_type = int(1)
        self.room_size = list([float(0), float(0)])
        self.room_height = float(0)
        self.room_stride = float(0)
        self.render_cfg = str()


class DataGenConfig(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.data_dir = str()
        self.out_dir = str()
        self.data_type = str()
        self.process_pipelines = list([ProcessPipeline()])


class GroupDataGenConfig(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.dataset_list = list([DataGenConfig()])


def get_cfg(cfg_type, cfg_path, shared_scope=''):
    cfg = cfg_type()
    cfg.load_from_yaml(cfg_path, shared_scope=shared_scope)
    return cfg


def get_cfg_from_pipeline(cfg_type, cfg_path: str, assemble_name: str = None) -> Tuple[DataGenConfig, ProcessPipeline]:
    data_cfg = [p_p for p_p in get_cfg(cfg_type, cfg_path).dataset_list]
    assert len(data_cfg) == 1, f'cfg_path: {cfg_path}, assemble_name: {assemble_name}, {len(data_cfg)} dataset'
    eval_cfg = [p_p for p_p in data_cfg[0].process_pipelines if assemble_name in [p_p.assemble_name, None]]
    assert len(eval_cfg) == 1, f'cfg_path: {cfg_path}, assemble_name: {assemble_name}, {len(eval_cfg)} cfg'
    return data_cfg[0], eval_cfg[0]


def get_map_dict_from_csv(map_file, sep_str, id_key, map_key):
    map_dict = dict()
    map_csv = pd.read_csv(map_file, sep=sep_str)
    for k, v in map_csv[[id_key, map_key]].values:
        map_dict[k] = str(v)
    return map_dict


class BinvoxIO(object):
    def __init__(self, bin_path, align_vox=False, object_scale=None, vox_stride=None):
        self.bin_path = bin_path
        self.align_vox = align_vox
        self.object_scale = object_scale
        self.vox_stride = vox_stride

        self._scale = None
        self._dimension = None
        self._translation = None
        self._volumetric = None
        self.points_xyz = None
        self.read_bin_vox()
        self.post_process()

    def read_bin_vox(self):
        fp = self.bin_path
        if not isinstance(self.bin_path, io.BytesIO):
            fp = open(self.bin_path, 'rb')
        _ = fp.readline().rstrip().decode('utf-8')
        dim_str = g_str.reorganize_by_symbol(fp.readline().rstrip().decode('utf-8'), ' ', slice(1, 4))
        self._dimension = np.fromstring(dim_str, sep=' ', dtype=np.int32)
        translate_str = g_str.reorganize_by_symbol(fp.readline().rstrip().decode('utf-8'), ' ', slice(1, 4))
        self._translation = np.fromstring(translate_str, sep=' ', dtype=np.float32)
        self._scale = float(g_str.reorganize_by_symbol(fp.readline().rstrip().decode('utf-8'), ' ', slice(1, 2)))
        scale_norm = self._scale / self._dimension[0]
        _ = fp.readline().rstrip().decode('utf-8')
        data_pair = np.transpose(np.reshape(np.frombuffer(fp.read(), dtype=np.uint8), [-1, 2]))
        data_stop = np.cumsum(data_pair[1])
        data_start = np.concatenate([np.array([0], dtype=np.uint32), data_stop[:-1]])
        data_stop = data_stop[data_pair[0] != 0]
        data_start = data_start[data_pair[0] != 0]

        points_indices = np.concatenate([np.arange(stt, stp) for stt, stp in zip(data_start, data_stop)])
        points_xy = (points_indices / self._dimension[2]).astype(np.uint32)
        points_x = (points_xy / self._dimension[1]).astype(np.uint32)
        points_y = points_xy % self._dimension[1]
        points_z = points_indices % self._dimension[2]
        points_xyz = np.stack([points_x, points_y, points_z], axis=-1).astype(np.int32)
        self.points_xyz = points_xyz.astype(np.float32) * scale_norm + self._translation
        self._volumetric = np.zeros(self._dimension, np.uint8)
        self._volumetric[tuple(np.split(points_xyz, 3, axis=-1))] = 1

    def post_process(self):
        if self.align_vox:
            align_indices = [0, 2, 1]
            target_position = np.min(self.points_xyz, axis=0)
            target_position[1] = 0
            self.points_xyz = self.points_xyz[:, align_indices]
            points_xyz_position = np.min(self.points_xyz, axis=0)
            points_translation = target_position - points_xyz_position
            self.points_xyz = self.points_xyz + points_translation

            self._translation = self._translation[align_indices] + points_translation
            self._volumetric = np.transpose(self._volumetric, align_indices)
            self._dimension = self._dimension[align_indices]

        if self.object_scale is not None:
            vox_stride = np.max(self.object_scale * self._scale / self._dimension)
            if vox_stride >= self.vox_stride:
                vox_stride = vox_stride if vox_stride > self.vox_stride else vox_stride * 1.1
                zoom_factor = int(np.ceil(vox_stride / self.vox_stride))
                self._dimension = self._dimension * zoom_factor
                self._volumetric = ndimage.zoom(self._volumetric, zoom_factor, np.uint8, mode='nearest')
                object_vox_indices = np.argwhere(self._volumetric > 0)
                scale_norm = self._scale / self._dimension[0]
                self.points_xyz = object_vox_indices.astype(np.float32) * scale_norm + self._translation


class AxisAlignBoundingBox(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.max = list([float(0.), float(0.), float(0.)])
        self.min = list([float(0.), float(0.), float(0.)])

    def assign_box_size(self, maximum, minimum):
        self.max = np.asarray(maximum, dtype=np.float32)
        self.min = np.asarray(minimum, dtype=np.float32)

    def scale(self, scale_num):
        self.max = np.asarray(self.max) * scale_num
        self.min = np.asarray(self.min) * scale_num

    def translation(self, translation_offset):
        self.max = np.asarray(self.max) + translation_offset
        self.min = np.asarray(self.min) + translation_offset

    def rotation(self, rotation_m):
        min_rot = np.matmul(rotation_m, np.asarray(self.min))
        max_rot = np.matmul(rotation_m, np.asarray(self.max))
        self.max = np.maximum(min_rot, max_rot)
        self.min = np.minimum(min_rot, max_rot)

    def center(self):
        return (np.array(self.min) + self.max) / 2

    def center_floor(self):
        c_floor = np.asarray(self.center())
        c_floor[1] = self.min[1]
        return c_floor

    def box_size(self):
        return np.array(self.max) - self.min

    def box_area(self):
        return np.prod(self.box_size())

    def is_in_box(self, points, eps=0):
        points = np.asarray(points)
        lower = np.all(points + eps > self.min, axis=-1)
        upper = np.all(points - eps < self.max, axis=-1)
        return np.logical_and(lower, upper)


class UniformObject(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.label = str()
        self.label_id = int(0)
        self.model_id = str()
        self.scale = int(1)
        self.bbox = AxisAlignBoundingBox()
        self.world_pose = list([[float(0.)] * 4] * 4)
        self.interpolation = int(1)
        self.rotation = list([float(0.)] * 3)

    def mesh_center_floor(self, model_dir):
        mesh_dir = os.path.join(model_dir, '../../object')
        mesh_path = os.path.join(mesh_dir, self.model_id, f'{self.model_id}_0.obj')
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(mesh_dir, self.model_id, f'{self.model_id}.obj')
            if not os.path.exists(mesh_path):
                raise NotImplementedError
        with g_io.OpenText(mesh_path, 'r') as obj_io:
            obj_local = np.array([list(map(float, obj_line.split()[1:4])) for obj_line in obj_io.read_lines()
                                  if obj_line.strip() != '' and obj_line.split()[0] == 'v'])

        obj_world = np.dot(np.insert(obj_local, 3, values=1, axis=1), self.world_pose)[:, :3]
        obj_max, obj_min = np.max(obj_world, axis=0), np.min(obj_world, axis=0)
        obj_center_floor = (obj_max + obj_min) / 2
        obj_center_floor[1] = obj_min[1]
        return obj_center_floor

    def read_points(self, objs_zip_meta):
        npz_meta = io.BytesIO(objs_zip_meta.read(f'{self.model_id}.npz'))
        obj_points = np.load(npz_meta)['arr_0'].astype(np.float32)
        if self.interpolation > 1:
            obj_min = np.min(obj_points, axis=0)
            obj_max = np.max(obj_points, axis=0)
            obj_center = (obj_max + obj_min) / 2
            obj_vox_scale = 126 / np.max(obj_max - obj_min)
            obj_vox_indices = ((obj_points - obj_center) * obj_vox_scale).astype(np.int32) + 64

            obj_voxel = np.zeros((128, 128, 128), dtype=np.uint8)
            obj_voxel[tuple(np.split(obj_vox_indices, 3, axis=-1))] = 1

            obj_vox_scaled = ndimage.zoom(obj_voxel, self.interpolation, np.uint8, mode='nearest')
            obj_points_scaled = np.argwhere(obj_vox_scaled > 0)
            obj_points = (obj_points_scaled / self.interpolation - 64) / obj_vox_scale + obj_center

        obj_points_world = np.concatenate([obj_points / self.scale, np.ones([obj_points.shape[0], 1])], axis=-1)
        obj_points_world = np.matmul(obj_points_world, self.world_pose)[:, :-1]
        return obj_points_world

    def read_vox_points(self, model_dir, object_scale=None, room_stride=None, align_vox=False):
        binvox = BinvoxIO(io.BytesIO(model_dir.read(self.model_id + '.binvox')), align_vox=align_vox,
                          object_scale=object_scale, vox_stride=room_stride)
        obj_points = binvox.points_xyz

        obj_points_world = np.concatenate([obj_points, np.ones([obj_points.shape[0], 1])], axis=-1)
        obj_points_world = np.matmul(obj_points_world, self.world_pose)[:, :-1]

        object_center_floor = (np.max(obj_points_world, axis=0) + np.min(obj_points_world, axis=0)) / 2
        object_center_floor[1] = np.min(obj_points_world, axis=0)[1]
        global_translate = self.bbox.center_floor() - object_center_floor
        obj_points_align = obj_points_world + global_translate
        return obj_points_align


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


class ZipResolver(trimesh.resolvers.Resolver):
    def __init__(self, zip_reader, model_id):
        self.zip_reader = zip_reader
        self.model_dir = '/'.join(model_id.split('/')[:-1])

    def get(self, name):
        file_path = os.path.normpath(os.path.join(self.model_dir, name)).replace('\\', '/')
        return io.BytesIO(self.zip_reader.read(file_path)).read()


class UniformScene(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.scene_id = str()
        self.room_type = str()
        self.label_type = str()
        self.bbox = AxisAlignBoundingBox()
        self.objects = list([UniformObject()])
        self.world_pose = list([[float(0.)] * 4] * 4)

    def get_object_by_model_id(self, model_id):
        model_list = [obj.model_id for obj in self.objects]
        assert len(set(model_list)) == len(model_list) and model_id in model_list, f'{model_id}, {model_list}'
        return self.objects[model_id.index(model_id)]

    def dump_to_json(self):
        return json.dumps(self.save(), cls=JsonEncoder, indent=4)

    def dump_scene_mesh(self, objs_zip_meta, vis_path, vis_type=2):
        scene_mesh = None
        for obj_i, obj_info in enumerate(self.objects):
            if 'front3d' in self.label_type.lower():
                model_path = os.path.join('3D-FUTURE-model', obj_info.model_id, 'raw_model.obj').replace('\\', '/')
            else:
                model_path = obj_info.model_id
            obj_resolver = ZipResolver(objs_zip_meta, model_path)
            obj_file = io.BytesIO(objs_zip_meta.read(model_path))
            obj_mesh = trimesh.load(obj_file, file_type='obj', resolver=obj_resolver)
            if isinstance(obj_mesh, trimesh.Trimesh):
                obj_mesh = trimesh.Scene(geometry=obj_mesh)
            obj_mesh.apply_transform(np.asarray(obj_info.world_pose).T)
            if scene_mesh is not None:
                scene_mesh.add_geometry(obj_mesh.geometry)
            else:
                scene_mesh = obj_mesh

        if scene_mesh is not None:
            if vis_type & 2**0:
                scene_mesh.export(vis_path + '_mesh.glb')
            if vis_type & 2**1:
                scene_mesh.export(vis_path + '_mesh.obj')

    @staticmethod
    def bbox2points(obj_max, obj_min, obj_label):
        bbox_edge_x_range = np.linspace(obj_min[0], obj_max[0], max((obj_max[0] - obj_min[0]) // 0.025, 2))
        bbox_edge_x_zeros = np.zeros_like(bbox_edge_x_range)
        bbox_edge_x_min = np.stack([bbox_edge_x_range, bbox_edge_x_zeros, bbox_edge_x_zeros + obj_min[2]], 1)
        bbox_edge_x_max = np.stack([bbox_edge_x_range, bbox_edge_x_zeros, bbox_edge_x_zeros + obj_max[2]], 1)

        bbox_edge_z_range = np.linspace(obj_min[2], obj_max[2], max((obj_max[2] - obj_min[2]) // 0.025, 2))
        bbox_edge_z_zeros = np.zeros_like(bbox_edge_z_range)
        bbox_edge_z_min = np.stack([bbox_edge_z_zeros + obj_min[0], bbox_edge_z_zeros, bbox_edge_z_range], 1)
        bbox_edge_z_max = np.stack([bbox_edge_z_zeros + obj_max[0], bbox_edge_z_zeros, bbox_edge_z_range], 1)

        bbox_edge_points = np.concatenate([bbox_edge_x_min, bbox_edge_x_max, bbox_edge_z_min, bbox_edge_z_max])
        bbox_edge_points_label = np.zeros([bbox_edge_points.shape[0]]) + obj_label
        return bbox_edge_points, bbox_edge_points_label

    def parse_scene(self, objs_zip_meta, bbox=False, vox_stride=0.2):
        assert objs_zip_meta is not None or bbox
        objs_points = list()
        objs_points_label = list()
        for obj in self.objects:
            if objs_zip_meta is not None:
                obj_points = obj.read_points(objs_zip_meta)
                obj_points_label = np.array([obj.label_id] * obj_points.shape[0])
                objs_points.append(obj_points)
                objs_points_label.append(obj_points_label)
            if bbox:
                bbox_max = np.array(obj.bbox.max) - vox_stride
                bbox_edge_points, bbox_edge_points_label = self.bbox2points(bbox_max, obj.bbox.min, obj.label_id)
                objs_points.append(bbox_edge_points)
                objs_points_label.append(bbox_edge_points_label)
        return objs_points, objs_points_label

    def get_scene_voxel(self, objs_zip_meta, room_size, room_stride, pad=0.1):
        if len(self.objects) == 0:
            return None

        objs_points, points_label = self.parse_scene(objs_zip_meta)

        objs_points = np.concatenate(objs_points, axis=0)
        points_label = np.concatenate(points_label, axis=0)
        obj_vox_indices = (objs_points / room_stride + pad).astype(np.int32)
        point_xyz_max = (np.array(room_size) / room_stride).astype(np.int32)
        valid_indices = np.logical_and(np.all(obj_vox_indices >= 0, axis=1),
                                       np.all(obj_vox_indices < point_xyz_max, axis=1))
        obj_vox_indices, points_label = obj_vox_indices[valid_indices], points_label[valid_indices]

        voxel_size = (np.array(room_size) / room_stride).astype(np.int32)
        voxel_room = np.zeros(voxel_size, dtype=np.uint8)
        voxel_room[tuple(np.split(obj_vox_indices, 3, axis=-1))] = np.expand_dims(points_label, axis=-1)

        return voxel_room

    def vis_scene_voxel(self, objs_zip_meta, room_size, room_stride, vis_path):
        voxel_room = self.get_scene_voxel(objs_zip_meta, room_size, room_stride)
        if voxel_room is None:
            return None

        colors_map = getattr(g_data, self.label_type.upper())().color_map_arr()
        g_io.PlyIO().dump_vox(vis_path, voxel_room, vox_scale=room_stride, colors_map=colors_map)

    def dump_scene_bbox(self, vis_path: str, color_map=g_plt.d3c20_rgb(), room_vis=True, num_cls=-1,
                        order_vis=False, transpose=False, room_size=6.4, image_size=128):
        obj_label = np.asarray([obj.label_id for obj in self.objects])
        obj_center = np.asarray([obj.bbox.center() for obj in self.objects]) / room_size
        obj_size = np.asarray([obj.bbox.box_size() for obj in self.objects]) / room_size
        obj_rot = np.asarray([o.rotation[1] for o in self.objects])
        if room_vis:
            num_cls = num_cls if num_cls > 0 else (np.max(obj_label) + 1)
            obj_label = np.concatenate([obj_label, [num_cls]], axis=0)
            obj_center = np.concatenate([obj_center, [self.bbox.center_floor() / room_size]], axis=0)
            room_bbox_size = self.bbox.box_size()
            room_bbox_size[1] = 0.05
            obj_size = np.concatenate([obj_size, [room_bbox_size / room_size]], axis=0)
            obj_rot = np.concatenate([obj_rot, [0]], axis=0)
            color_map = np.concatenate([color_map, [[190, 200, 216]]], axis=0)

        if order_vis:
            obj_label = np.arange(len(obj_label)).reshape(obj_label.shape) + 1
            gradient_value = (np.linspace(0, 1, len(obj_label)) * 255).astype(np.uint8)
            color_map = np.stack([gradient_value, np.zeros_like(gradient_value), gradient_value[::-1]], axis=1)
            color_map = np.concatenate([[[255, 255, 255]], color_map], axis=0)
            vis_path = f'{vis_path}_order' if vis_path else None

        if np.all(obj_rot == 0):
            obj_rot = None
        bbox_img_color = visualization_utilize.VisualizationUtilize.visualize_scene_graph(
            obj_label, obj_center, obj_size, obj_rot, color_map, vis_path, 3, image_size, transpose)
        return bbox_img_color


class BaseDataGen(object):
    def __init__(self, data_dir, out_dir, process_pipelines, **kargs):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.process_pipelines = process_pipelines
        self.out_assemble_dir = g_str.mkdir_automated(os.path.join(self.out_dir, 'AssembleData'))

    @abstractmethod
    def mediate_process(self): pass

    @staticmethod
    def get_scene_list(scene_zip_path, filter_regex=r'.*\.zip', return_zip=False):
        if os.path.isdir(scene_zip_path):
            files = [os.path.join(scene_zip_path, f) for f in os.listdir(scene_zip_path) if re.match(filter_regex, f)]
            scene_reader = g_io.GroupZipIO(files)
        else:
            scene_reader = g_io.ZipIO(scene_zip_path)
        scene_list = sorted(scene_reader.namelist())
        if return_zip:
            return scene_reader, scene_list
        scene_reader.close()
        return scene_list

    @staticmethod
    def get_label_info(cfg):
        label_type = cfg.label_type
        data_label = getattr(g_data, label_type.upper())()
        label_list = data_label.label_id_map_arr()
        color_map = data_label.color_map_arr()
        return label_type, data_label, label_list, color_map

    @staticmethod
    def translate_to_local(uni_scene: UniformScene, target_room_center):
        room_center = uni_scene.bbox.center_floor()
        center_floor_trans = np.array(target_room_center) - room_center
        world_pose = np.eye(4)
        world_pose[3, :3] = center_floor_trans
        uni_scene.world_pose = world_pose.tolist()
        uni_scene.bbox.translation(center_floor_trans)
        for obj in uni_scene.objects:
            obj.bbox.translation(center_floor_trans)
            obj.world_pose[3][:3] = (obj.world_pose[3][:3] + center_floor_trans).tolist()
        return uni_scene

    @staticmethod
    def merge_statistics(out_dir, out_name, num_workers=8):
        out_tmp_dir = g_str.mkdir_automated(os.path.join(out_dir, 'tmp'))
        statistics_df = pd.DataFrame()
        for w_i in range(num_workers):
            w_data_df = pd.read_csv(os.path.join(out_tmp_dir, f'{out_name}_work{w_i}.csv'), index_col='Unnamed: 0')
            statistics_df = statistics_df.append(w_data_df, ignore_index=True)
        statistics_df.to_csv(os.path.join(out_dir, f'{out_name}.csv'))
        return statistics_df

    @staticmethod
    def bincount_statistics(statistics_list, out_path):
        statistics_list = np.asarray(statistics_list)
        statistics_unique_list = list(set(statistics_list.tolist()))
        statistics_index_list = [statistics_unique_list.index(r_t) for r_t in statistics_list]
        statistics_bincount = np.bincount(statistics_index_list, minlength=len(statistics_unique_list))
        statistics_bincount_df = pd.DataFrame(statistics_bincount.reshape([-1, 1]), index=statistics_unique_list)
        statistics_bincount_df.to_csv(out_path)
        return statistics_bincount_df

    @staticmethod
    def filter_far_view(cam_t, cam_r): pass

    # def filter_far_views(self, cam_t_list, cam_r_list, camera_id_list=None):
    #     filtered_cam_t_list, filtered_cam_r_list, filtered_cam_id_list = list(), list(), list()
    #     assert len(cam_t_list) == len(cam_r_list)
    #     for cam_i in range(len(cam_t_list)):
    #         cam_t, cam_r = cam_t_list[cam_i], cam_r_list[cam_i]
    #         if self.filter_far_view(cam_t, cam_r):
    #             filtered_cam_t_list.append(cam_t)
    #             filtered_cam_r_list.append(cam_r)
    #             if camera_id_list is not None:
    #                 filtered_cam_id_list.append(camera_id_list[cam_i])
    #     return np.asarray(filtered_cam_t_list), np.asarray(filtered_cam_r_list), np.array(filtered_cam_id_list)

    def filter_far_views(self, cam_t_list, cam_r_list, data_list=None):
        assert len(cam_t_list) == len(cam_r_list)
        cam_indices = np.array([self.filter_far_view(cam_t, cam_r) for cam_t, cam_r in zip(cam_t_list, cam_r_list)])
        if isinstance(data_list, list):
            return [np.asarray(d)[cam_indices] if d is not None else d for d in data_list]
        else:
            return [np.asarray(d)[cam_indices] for d in [cam_t_list, cam_r_list, data_list]]

    @staticmethod
    def split_info_in_dict(info_dict, info_type, split_list=('x', 'y', 'z')):
        info_dict = copy.deepcopy(info_dict)
        data_list = np.asarray(info_dict.pop(info_type))
        for s_i, s in enumerate(split_list):
            info_dict[f'{info_type}_{s}'] = data_list[:, s_i]
        return info_dict

    @staticmethod
    def statistics_scene_object_info(process_pipeline: ProcessPipeline, out_dir, out_name='', label_list=None):
        uni_scene_dir = g_str.mkdir_automated(os.path.join(out_dir, process_pipeline.label_type, 'uniform_scene'))
        analysis_dir = g_str.mkdir_automated(os.path.join(out_dir, process_pipeline.label_type, 'analysis'))
        for room_type in process_pipeline.room_types:
            out_room_info_path = os.path.join(analysis_dir, f'{room_type}_room_info{out_name}.csv')
            out_obj_info_path = os.path.join(analysis_dir, f'{room_type}_obj_info{out_name}.csv')
            if os.path.exists(out_room_info_path) and os.path.exists(out_obj_info_path):
                logging.info(f'Skip {out_room_info_path} and {out_obj_info_path} generation')
                return

            uniform_scene_reader = g_io.ZipIO(os.path.join(uni_scene_dir, f'{room_type}_uniform_scene.zip'))
            scene_list = uniform_scene_reader.namelist()

            label_count_list = list()
            room_info_dict = dict(room=list(), room_size=list(), room_obj_size=list(), num_obj=list())
            obj_info_dict = dict(room=list(), model_id=list(), label=list(), floor_center=list(), center=list(),
                                 size=list(), rotation=list(), scale=list())

            def get_scale_from_t(transform):
                return np.sqrt(np.sum(np.square(np.asarray(transform).T[:3, :3]), axis=0))

            for s_i, scene_id in enumerate(scene_list):
                scene_info = UniformScene()
                scene_info.load(json.loads(uniform_scene_reader.read(scene_id)))
                room_info_dict['room'].append(scene_info.scene_id)
                room_info_dict['room_size'].append(scene_info.bbox.box_size())

                def get_objects_box_size(uni_scene: UniformScene):
                    obj_box_max = np.max([obj.bbox.max for obj in uni_scene.objects], axis=0)
                    obj_box_min = np.min([obj.bbox.min for obj in uni_scene.objects], axis=0)
                    return obj_box_max - obj_box_min
                room_info_dict['room_obj_size'].append(get_objects_box_size(scene_info))
                room_info_dict['num_obj'].append(len(scene_info.objects))
                if label_list is not None:
                    obj_label_id_list = [obj.label_id for obj in scene_info.objects]                  
                    obj_label_count = np.bincount(obj_label_id_list, minlength=len(label_list))
                    label_count_list.append(obj_label_count)

                obj_model_id_list = [obj.model_id for obj in scene_info.objects]
                obj_label_list = [obj.label for obj in scene_info.objects]
                obj_floor_center_list = np.asarray([obj.bbox.center_floor() for obj in scene_info.objects]).tolist()
                obj_center_list = np.asarray([obj.bbox.center() for obj in scene_info.objects]).tolist()
                obj_size_list = np.asarray([obj.bbox.box_size() for obj in scene_info.objects]).tolist()
                obj_rotation_list = np.asarray([obj.rotation for obj in scene_info.objects]).tolist()
                obj_scale_list = np.asarray([get_scale_from_t(obj.world_pose) for obj in scene_info.objects]).tolist()

                obj_info_dict['room'].extend([scene_info.scene_id] * len(obj_model_id_list))
                obj_info_dict['model_id'].extend(obj_model_id_list)
                obj_info_dict['label'].extend(obj_label_list)
                obj_info_dict['floor_center'].extend(obj_floor_center_list)
                obj_info_dict['center'].extend(obj_center_list)
                obj_info_dict['size'].extend(obj_size_list)
                obj_info_dict['rotation'].extend(obj_rotation_list)
                obj_info_dict['scale'].extend(obj_scale_list)

            uniform_scene_reader.close()

            room_info_dict = BaseDataGen.split_info_in_dict(room_info_dict, 'room_size')
            room_info_dict = BaseDataGen.split_info_in_dict(room_info_dict, 'room_obj_size')
            if label_list is not None:
                label_count_list = np.asarray(label_count_list)
                label_count_dict = {t: label_count_list[..., t_i] for t_i, t in enumerate(label_list)}
                room_info_dict = {**room_info_dict, **label_count_dict}
            pd.DataFrame(room_info_dict).to_csv(out_room_info_path)

            obj_info_dict = BaseDataGen.split_info_in_dict(obj_info_dict, 'floor_center')
            obj_info_dict = BaseDataGen.split_info_in_dict(obj_info_dict, 'center')
            obj_info_dict = BaseDataGen.split_info_in_dict(obj_info_dict, 'size')
            obj_info_dict = BaseDataGen.split_info_in_dict(obj_info_dict, 'rotation')
            obj_info_dict = BaseDataGen.split_info_in_dict(obj_info_dict, 'scale')
            obj_info_df = pd.DataFrame(obj_info_dict)
            obj_info_df.to_csv(out_obj_info_path)

            if label_list is None:
                continue
            model_count_list = np.zeros(len(label_list), dtype=np.int32)
            category_dir = g_str.mkdir_automated(os.path.join(analysis_dir, 'category_info'))
            for l_id, label_name in enumerate(label_list):
                if label_name in ['void']:
                    continue
                label_obj_info_df = obj_info_df[obj_info_df['label'] == label_name]
                out_path = os.path.join(category_dir, f'{room_type}_{label_name.replace("/", "")}_info{out_name}.csv')
                label_obj_info_df.to_csv(out_path)

                model_list = np.unique(np.asarray(label_obj_info_df['model_id']))
                model_count_list[l_id] = len(model_list)
            model_count_path = os.path.join(analysis_dir, 'model_count.csv')
            pd.DataFrame(dict(label=label_list, model_count=model_count_list)).to_csv(model_count_path)

    @staticmethod
    def sort_obj_by_rules(scene_info, obj_info_list, sort_rule='random', reverse_order=False):
        obj_info_list = np.asarray(obj_info_list)
        if sort_rule == 'random':
            obj_argsort = np.arange(len(obj_info_list)).astype(np.int32)
            np.random.shuffle(obj_argsort)
        elif sort_rule == 'size':
            # Kai Wang
            # obj_size = np.array([np.prod(obj.bbox.box_size()[[0, 2]]) for obj in obj_info_list])
            obj_size = np.array([obj.bbox.box_area() for obj in obj_info_list])
            obj_argsort = np.argsort(obj_size)[::-1]
        elif sort_rule == 'position':
            # Variational Transformer Network
            obj_position = np.array([obj.bbox.center() for obj in obj_info_list])
            flatten_position = obj_position[:, 0] * 1e8 + obj_position[:, 2] * 1e4 + obj_position[:, 1]
            obj_argsort = np.argsort(flatten_position)
        elif sort_rule == 'distance':
            obj_y_pos = np.array([obj.bbox.min[1] for obj in obj_info_list])
            obj_y_pos_argsort = np.argsort(obj_y_pos, kind='stable')
            obj_info_list = obj_info_list[obj_y_pos_argsort]

            obj_floor_pos = np.array([obj.bbox.center_floor() for obj in obj_info_list])[:, [0, 2]]
            obj_distance = np.linalg.norm(obj_floor_pos - scene_info.bbox.center_floor()[[0, 2]], axis=-1)
            obj_argsort = np.argsort(obj_distance, kind='stable')[::-1]
        else:
            raise NotImplementedError
        obj_info_list = obj_info_list[obj_argsort]
        if reverse_order:
            # obj_info_list = obj_info_list[::-1]
            raise NotImplementedError
        return obj_info_list

    @staticmethod
    def visualize_multi_views_images(vis_path, color_image=None, depth_image=None, label_image=None, color_map=None,
                                     depth_scale=1000*6.4, padding=False):

        def get_multi_views_image(cube_images, pad_value=0):
            if padding:
                cube_images = np.pad(cube_images, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=pad_value)
            cube_images_t = np.transpose(cube_images, (1, 0, 2, 3))
            cube_images_reshape = cube_images_t.reshape((cube_images_t.shape[0], -1, cube_images_t.shape[-1]))
            return cube_images_reshape

        if color_image is not None:
            color_image_vis = color_image.astype(np.uint8)
            color_image_vis = get_multi_views_image(color_image_vis) if color_image_vis.ndim == 4 else color_image_vis
            cv2.imwrite(vis_path + '_color.png', color_image_vis[..., ::-1])
        if depth_image is not None:
            depth_image_vis = depth_image if depth_image.shape[-1] == 1 else depth_image[..., None]
            depth_image_vis = np.where(depth_image_vis < 65535, depth_image_vis, 0)
            if depth_scale is not None:
                depth_image_vis = (depth_image_vis / depth_scale * 255).astype(np.uint8)
            depth_image_vis = get_multi_views_image(depth_image_vis) if depth_image_vis.ndim == 4 else depth_image_vis
            cv2.imwrite(vis_path + '_depth.png', depth_image_vis[..., 0])
        if label_image is not None:
            label_image_vis = color_map[label_image].astype(np.uint8)
            label_image_vis = get_multi_views_image(label_image_vis) if label_image_vis.ndim == 4 else label_image_vis
            cv2.imwrite(vis_path + '_category.png', label_image_vis[..., ::-1])

    @staticmethod
    def visualize_rotated_views(vis_path, color_image=None, depth_image=None, label_image=None, color_map=None,
                                depth_scale=1000*6.4, video_time=5):
        all_images_vis = None
        if color_image is not None:
            color_image_vis = color_image.astype(np.uint8)
            all_images_vis = color_image_vis
            imageio.mimsave(vis_path + f'_color_{video_time}s.gif', color_image_vis, duration=video_time / color_image_vis.shape[0])
        if depth_image is not None:
            depth_image_vis = depth_image if depth_image.shape[-1] == 1 else depth_image[..., None]
            depth_image_vis = (np.where(depth_image_vis < 65535, depth_image_vis, 0)/depth_scale * 255).astype(np.uint8)
            depth_image_vis = np.tile(depth_image_vis, [1, 1, 1, 3])
            all_images_vis = depth_image_vis if all_images_vis is None else np.concatenate([all_images_vis, depth_image_vis], axis=2)
            imageio.mimsave(vis_path + f'_depth_{video_time}s.gif', depth_image_vis, duration=video_time / depth_image_vis.shape[0])
        if label_image is not None:
            label_image_vis = color_map[label_image].astype(np.uint8)
            all_images_vis = label_image_vis if all_images_vis is None else np.concatenate([all_images_vis, label_image_vis], axis=2)
            imageio.mimsave(vis_path + f'_category_{video_time}s.gif', label_image_vis, duration=video_time / label_image_vis.shape[0])
        if all_images_vis is not None:
            imageio.mimsave(vis_path + f'_{video_time}s.gif', all_images_vis, duration=video_time / all_images_vis.shape[0])
