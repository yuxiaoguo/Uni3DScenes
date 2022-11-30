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

    def post_num_points_distribution(self, proc_unit: ProcessUnit, ipc_vars):
        """
        Post statistics the number of points distribution
        """
        all_dict = dict()
        for i_var in ipc_vars:
            all_dict.update(i_var)

        out_path = self.envs.get_env_path(proc_unit.out_paths[0])
        table = DistributionTable(out_path)
        table.write(['Points'], None, all_dict)

    def num_points_distribution(self, sample: str, proc_unit: ProcessUnit, shared_vars: Dict):
        """
        Statistics the number of points distribution
        """
        data_name = proc_unit.assemble_function
        shared_vars.setdefault(data_name, dict())
        data: Dict[str, np.ndarray] = self._load_sample_from_vars(sample, shared_vars)
        sample_name = os.path.splitext(os.path.basename(sample))[0]
        shared_vars[data_name][sample_name] = [data['points'].shape[0]]

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
