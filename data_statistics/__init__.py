"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os

from typing import List, Dict, Type

import numpy as np

from utils.config import ProcessUnit, EnvsConfig, MPEntryBase

from . import dist_func


class PointCloudStatistics(MPEntryBase):
    """
    Statistics for point cloud data
    """
    def __init__(self, proc_units: List[ProcessUnit], envs: EnvsConfig) -> None:
        super().__init__(proc_units, envs)
        self._enable_mp = False

        self.proc_func_dict: Dict[str, dist_func.DistFuncBase] = dict()
        for proc_unit in proc_units:
            dist_cls: Type[dist_func.DistFuncBase] = getattr(dist_func, proc_unit.assemble_function)
            self.proc_func_dict[proc_unit.assemble_function] = dist_cls(proc_unit, envs)

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

    def _execute_proc_unit(self, sample: str, proc_unit: ProcessUnit, shared_vars: Dict):
        proc_func = self.proc_func_dict[proc_unit.assemble_function]
        proc_func.processing(self._load_sample_from_vars(sample, shared_vars), shared_vars, sample)

    def _merged_within_processing(self, shared_vars: Dict, ipc_vars: List):
        ipc_info = list()
        for proc_unit in self.proc_units:
            ipc_info.append(shared_vars[proc_unit.assemble_function])
        ipc_vars.append(ipc_info)

    def _merged_cross_processing(self, ipc_vars):
        for proc_idx, proc_unit in enumerate(self.proc_units):
            proc_func = self.proc_func_dict[proc_unit.assemble_function]
            proc_func.post([_f[proc_idx] for _f in ipc_vars])
