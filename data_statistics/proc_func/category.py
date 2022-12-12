"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
from typing import List, Dict

import numpy as np

from .base import FuncBase
from ..tables.excel_io import DistributionTable


class CategoryDistribution(FuncBase):
    """
    Object category distribution
    """
    def processing(self, data: Dict[str, np.ndarray], shared_vars: dict, sample_name: str = None):
        del sample_name
        num_categories = self.proc_unit.attrs['num_categories']
        data_name = self.proc_unit.name
        shared_vars.setdefault(data_name, np.zeros(num_categories, dtype=np.int64))
        # data = self._load_sample_from_vars(sample, shared_vars)
        shared_vars[data_name] += np.bincount(np.reshape(data['labels'], [-1]), \
            minlength=num_categories)

    def post(self, ipc_vars: List):
        count_raw = np.asarray(ipc_vars, dtype=np.float32)
        count_sum = np.sum(count_raw, axis=0)
        dist: np.ndarray = count_sum / np.sum(count_sum)

        out_path = self.envs.get_env_path(self.proc_unit.out_paths[0])
        table = DistributionTable(out_path)
        table.write_overall(np.arange(dist.shape[0], dtype=np.int32), dist)
        table.close()
