"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
import multiprocessing as mp

from abc import abstractmethod
from typing import List

import numpy as np

from graphics_utils import g_perf

from utils.config import ProcessUnit, EntryBase


class StatisticsBase(EntryBase):
    """
    The base class of dataset
    """
    @abstractmethod
    def distribution(self, proc_unit: ProcessUnit):
        """
        Construct 3D point cloud from views
        """


class PointCloudStatistics(StatisticsBase):
    """
    Statistics for point cloud data
    """
    def _distribution_mp(self, samples, stat_out: List, proc_unit: ProcessUnit, \
        offset=0, worker_id=0):
        del offset, worker_id, proc_unit
        label_dists = np.zeros(41, dtype=np.int64)
        for sample in samples:
            meta = np.load(sample)
            labels = meta['labels']
            label_dists += np.bincount(np.reshape(labels, [-1]), minlength=41)
        stat_out.append(label_dists)

    def distribution(self, proc_unit: ProcessUnit):
        point_cloud_folder = self.envs.get_env_path(proc_unit.in_paths[0])
        point_cloud_files = [os.path.join(point_cloud_folder, _f) \
            for _f in os.listdir(point_cloud_folder)]

        mp_stat = mp.Manager().list()

        g_perf.multiple_processor(self._distribution_mp, point_cloud_files, 8, (mp_stat, proc_unit))

        ar_stat: np.ndarray = np.sum(np.asarray(mp_stat), axis=0)
        ar_total: np.ndarray = np.sum(ar_stat)
        print(np.argwhere(ar_stat > 0)[..., 0])

        ar_dist = ar_stat.astype(np.float32) / ar_total.astype(np.float32)
        print([f'{_d:.4f}' for _d in ar_dist[ar_stat > 0]])
