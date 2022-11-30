"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
import multiprocessing as mp

from typing import List, Dict

import numpy as np

from graphics_utils import g_perf

from .tables.excel_io import DistributionTable
from utils.config import ProcessUnit, EnvsConfig, MPEntryBase


class PointCloudStatistics(MPEntryBase):
    """
    Statistics for point cloud data
    """
    def __init__(self, proc_units: List[ProcessUnit], envs: EnvsConfig) -> None:
        super().__init__(proc_units, envs)
        self._enable_mp = False

    def _sample_list(self):
        point_cloud_dir = self.envs.get_env_path(self.proc_units[0].in_paths[0])
        point_cloud_samples = [os.path.join(point_cloud_dir, _f) for _f in \
            os.listdir(point_cloud_dir)]
        return point_cloud_samples

    def _load_sample_from_vars(self, sample: str, shared_vars: Dict):
        sample_name = 'sample_name'
        sample_data = 'sample_data'

        shared_vars.setdefault(sample_name, '')
        if sample == shared_vars[sample_name]:
            return shared_vars[sample_data]
        else:
            shared_vars[sample_name] = sample
            shared_vars[sample_data] = np.load(sample)
        return shared_vars[sample_data]

    def _merged_within_processing(self, shared_vars: Dict, ipc_vars: List):
        ipc_info = list()
        for proc_unit in self.proc_units:
            ipc_info.append(shared_vars[proc_unit.assemble_function])
        ipc_vars.append(ipc_info)

    def _merged_cross_processing(self, ipc_vars):
        for proc_idx, proc_unit in enumerate(self.proc_units):
            post_func = getattr(self, f'post_{proc_unit.assemble_function}')
            post_func(proc_unit, [_f[proc_idx] for _f in ipc_vars])

    def post_category_distribution(self, proc_unit: ProcessUnit, ipc_vars):
        """
        Post statistics the category distribution
        """
        count_raw = np.asarray(ipc_vars, dtype=np.float32)
        count_sum = np.sum(count_raw, axis=0)
        dist: np.ndarray = count_sum / np.sum(count_sum)

        out_path = self.envs.get_env_path(proc_unit.out_paths[0])
        table = DistributionTable(out_path)
        table.write(np.arange(dist.shape[0], dtype=np.int32), dist)

    def category_distribution(self, sample: str, proc_unit: ProcessUnit, shared_vars: Dict):
        """
        Statistics the category distribution
        """
        num_categories = proc_unit.attrs['num_categories']
        data_name = proc_unit.assemble_function
        shared_vars.setdefault(data_name, np.zeros(num_categories, \
            dtype=np.int64))
        data = self._load_sample_from_vars(sample, shared_vars)
        shared_vars[data_name] += np.bincount(np.reshape(data['labels'], [-1]), \
            minlength=num_categories)

    # def _distribution_mp(self, samples, stat_out: List, proc_unit: ProcessUnit, \
    #     offset=0, worker_id=0):
    #     del offset, worker_id, proc_unit
    #     label_dists = np.zeros(41, dtype=np.int64)
    #     for sample in samples:
    #         meta = np.load(sample)
    #         labels = meta['labels']
    #         label_dists += np.bincount(np.reshape(labels, [-1]), minlength=41)
    #     stat_out.append(label_dists)

    # def distribution(self, proc_unit: ProcessUnit):
    #     point_cloud_folder = self.envs.get_env_path(proc_unit.in_paths[0])
    #     point_cloud_files = [os.path.join(point_cloud_folder, _f) \
    #         for _f in os.listdir(point_cloud_folder)]

    #     mp_stat = mp.Manager().list()

    #     g_perf.multiple_processor(self._distribution_mp, point_cloud_files, 8, (mp_stat, proc_unit))

    #     ar_stat: np.ndarray = np.sum(np.asarray(mp_stat), axis=0)
    #     ar_total: np.ndarray = np.sum(ar_stat)
    #     print(np.argwhere(ar_stat > 0)[..., 0])

    #     ar_dist = ar_stat.astype(np.float32) / ar_total.astype(np.float32)
    #     print([f'{_d:.4f}' for _d in ar_dist[ar_stat > 0]])
