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

from datasets.utils.base import ProcessUnit


class StatisticsBase:
    """
    The base class for down-streaming tasks
    """
    def __init__(self, root_dir: str, proc_units: List[ProcessUnit]) -> None:
        self.root_dir = root_dir
        self.proc_units = proc_units

    @abstractmethod
    def distribution(self, proc_unit: ProcessUnit):
        """
        Construct 3D point cloud from views
        """

    def execute_pipeline(self):
        """
        execute the data processing pipeline
        """
        for proc_unit in self.proc_units:
            proc_func = getattr(self, proc_unit.assemble_function)
            proc_func(proc_unit)


class CoreStatistics(StatisticsBase):
    def distribution_mp(self, samples, stat_out: List, proc_unit: ProcessUnit, \
        offset=0, worker_id=0):
        label_dists = np.zeros(41, dtype=np.int64)
        for sample in samples:
            meta = np.load(sample)
            labels = meta['labels']
            label_dists += np.bincount(np.reshape(labels, [-1]), minlength=41)
        stat_out.append(label_dists)

    def distribution(self, proc_unit: ProcessUnit):
        target_folder = os.path.join(self.root_dir, 'point_cloud')
        target_files = [os.path.join(target_folder, _f) for _f in os.listdir(target_folder)]

        mp_statistics = mp.Manager().list()

        g_perf.multiple_processor(self.distribution_mp, target_files, 1, (mp_statistics, proc_unit))

        ar_statistics = np.sum(np.asarray(mp_statistics), axis=0)
        print(np.argwhere(ar_statistics > 0)[..., 0])

        ar_dist = ar_statistics.astype(np.float32) / np.sum(ar_statistics).astype(np.float32)
        print([f'{_d:.4f}' for _d in ar_dist[ar_statistics > 0]])
