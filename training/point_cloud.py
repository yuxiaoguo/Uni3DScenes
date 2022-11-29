"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
from typing import List

import numpy as np

from graphics_utils import g_perf

from utils.config import ProcessUnit
from .backend import TorchWriter
from .base_downstreaming import DownStreamingBase


class PointCloudDownStreaming(DownStreamingBase):
    """
    To deal with point cloud down-streaming task
    """
    def _segmentation_mp(self, samples: List[str], proc_unit: ProcessUnit, split=str, offset=0, \
        worker_id: int = 0):
        del offset, worker_id
        trainsets_dir = self.envs.get_env_path(proc_unit.out_paths[0])
        point_cloud_dir = self.envs.get_env_path(proc_unit.in_paths[0])
        os.makedirs(trainsets_dir, exist_ok=True)

        remapper = np.ones(150, dtype=np.int32) * (-100)
        for i_id, s_id in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15,\
            16, 17, 18, 19, 22, 24, 25, 32, 34, 35, 38, 39, 40]):
            remapper[s_id] = i_id

        torch_writer = TorchWriter(trainsets_dir)

        for sample in samples:
            sample_meta = np.load(os.path.join(point_cloud_dir, sample))
            points = sample_meta['points']
            colors = (sample_meta['colors'].astype(np.float32) / 127.5) - 1
            labels = remapper[sample_meta['labels']]
            torch_writer.write_item((points, colors, labels), os.path.splitext(sample)[0])

    def _split_samples(self, samples: List[str]):
        train_samples, val_samples, test_samples = list(), list(), list()
        for sample in samples:
            scene_id = int(sample.split('_')[1])
            if scene_id < 3000:
                train_samples.append(sample)
            elif scene_id < 3250:
                val_samples.append(sample)
            else:
                test_samples.append(sample) 
        return train_samples, val_samples, test_samples

    def segmentation(self, proc_unit: ProcessUnit):
        point_cloud_dir = self.envs.get_env_path(proc_unit.in_paths[0])

        train_samples, val_samples, test_samples = self._split_samples(os.listdir(point_cloud_dir))
        g_perf.multiple_processor(self._segmentation_mp, train_samples, 8, (proc_unit, 'train'))
        g_perf.multiple_processor(self._segmentation_mp, val_samples, 8, (proc_unit, 'val'))
        g_perf.multiple_processor(self._segmentation_mp, test_samples, 8, (proc_unit, 'test'))
