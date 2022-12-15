"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
import pickle

from typing import List, Dict, Type

import numpy as np

from utils.config import ProcessUnit, EnvsConfig, MPEntryBase

from . import proc_func


class SampleWiseProcessing(MPEntryBase):
    """
    Statistics for point cloud data
    """
    def __init__(self, proc_units: List[ProcessUnit], envs: EnvsConfig) -> None:
        super().__init__(proc_units, envs)
        # self._enable_mp = False

        self.proc_func_dict: Dict[str, proc_func.FuncBase] = dict()
        for proc_unit in proc_units:
            dist_cls: Type[proc_func.FuncBase] = getattr(proc_func, proc_unit.assemble_function)
            self.proc_func_dict[proc_unit.name] = dist_cls(proc_unit, envs)

    def _sample_list(self):
        point_cloud_dir = self.envs.get_env_path(self.proc_units[0].in_paths[0])
        point_cloud_samples = os.listdir(point_cloud_dir)
        return point_cloud_samples

    def _load_sample_from_vars(self, sample: str, shared_vars: Dict, input_alias):
        sample_name = f'{input_alias}_name'
        sample_data = f'{input_alias}_data'

        shared_vars.setdefault(sample_name, '')
        if sample == shared_vars[sample_name]:
            return shared_vars[sample_data]
        else:
            shared_vars[sample_name] = sample
            try:
                shared_vars[sample_data] = np.load(sample)
            except ValueError:
                with open(os.path.join(self.envs.out_data_root, 'desc', \
                    input_alias), 'rb') as b_fp:
                    data_desc: np.ndarray = pickle.load(b_fp)
                dtype = data_desc.dtype
                shape = [_d if _d != 0 else -1 for _d in data_desc.shape]
                shared_vars[sample_data] = np.fromfile(sample, dtype).reshape(shape)
        return shared_vars[sample_data]

    def _execute_proc_unit(self, sample: str, proc_unit: ProcessUnit, shared_vars: Dict):
        func = self.proc_func_dict[proc_unit.name]
        inputs = list()
        for in_data in proc_unit.in_paths:
            sample_path = os.path.join(self.envs.get_env_path(in_data), sample)
            inputs.append(self._load_sample_from_vars(sample_path, shared_vars, in_data))
        func.processing(inputs, shared_vars, sample)

    def _merged_within_processing(self, shared_vars: Dict, ipc_vars: List):
        ipc_info = list()
        for proc_unit in self.proc_units:
            if proc_unit.name in shared_vars:
                ipc_info.append(shared_vars[proc_unit.name])
        ipc_vars.append(ipc_info)

    def _merged_cross_processing(self, ipc_vars):
        for proc_idx, proc_unit in enumerate(self.proc_units):
            func = self.proc_func_dict[proc_unit.name]
            func.post([_f[proc_idx] for _f in ipc_vars if _f])
