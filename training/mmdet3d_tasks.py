"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
import pickle
import logging

from utils.config import ProcessUnit
from .base_downstreaming import DownStreamingBase


class MMDet3DDownStreaming(DownStreamingBase):
    """
    Down streaming interface for MMDet3D platform
    """
    def segmentation(self, proc_unit: ProcessUnit):
        logging.info('Segmentation is contained by detection task')
        raise NotImplementedError

    def _generate_detection_split(self, samples: str, split: str,
        proc_unit: ProcessUnit):
        out_folder = self.envs.get_env_path(proc_unit.out_paths[0])
        samples_desc = list()
        for sample in samples:
            sample_desc = dict()
            sample_desc['point_cloud'] = dict(num_features=6, \
                lidar_idx=os.path.splitext(sample)[0])
            sample_desc['pts_path'] = f'points/{sample}'
            sample_desc['pts_semantic_mask_path'] = \
                f'semantic_mask/{sample}'
            sample_desc['pts_instance_mask_path'] = ''

            with open(os.path.join(out_folder, 'anno_mask', \
                sample), 'rb') as a_fp:
                sample_desc['annos'] = pickle.load(a_fp)

            samples_desc.append(sample_desc)
        with open(os.path.join(out_folder, \
            f'scannet_infos_{split}.pkl'), 'wb') as s_fp:
            pickle.dump(samples_desc, s_fp)

    def detection(self, proc_unit: ProcessUnit):
        points_dir = os.path.join(self.envs.get_env_path(\
            proc_unit.in_paths[0]), 'points')
        samples = os.listdir(points_dir)

        def _is_train_split(_sample_id: str):
            return int(_sample_id.split('_')[1]) < 3000

        def _is_val_split(_sample_id: str):
            _scene_id = int(_sample_id.split('_')[1])
            return _scene_id < 3250 and _scene_id >= 3000

        def _is_test_split(_sample_id: str):
            return int(_sample_id.split('_')[1]) >= 3250

        train_samples = [_s for _s in samples if _is_train_split(_s)]
        self._generate_detection_split(train_samples, 'train', proc_unit)
        val_samples = [_s for _s in samples if _is_val_split(_s)]
        self._generate_detection_split(val_samples, 'val', proc_unit)
        # test_samples = [_s for _s in samples if _is_test_split(_s)]
