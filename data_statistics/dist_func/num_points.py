"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
from typing import List

import numpy as np

from .base import DistFuncBase
from ..tables.excel_io import DistributionTable


class NumPointsDistribution(DistFuncBase):
    """
    Distribution of the number of points in each sample
    """
    def post(self, ipc_vars: List):
        """
        Post statistics the number of points distribution
        """
        out_path = self.envs.get_env_path(self.proc_unit.out_paths[0])
        table = DistributionTable(out_path)

        all_dict = dict()
        for i_var in ipc_vars:
            all_dict.update(i_var)

        all_points = np.asarray(list(all_dict.values())).reshape([-1])
        bin_points = np.bincount(all_points // self.proc_unit.attrs['stride'])

        pdf = bin_points.astype(np.float32) / np.sum(bin_points)
        cdf = np.cumsum(pdf)

        table.write_overall(np.arange(len(bin_points)), pdf, 'PDF')
        table.write_overall(np.arange(len(bin_points)), cdf, 'CDF')
        table.write_overall(np.arange(len(bin_points)), bin_points, 'DF')
        table.write_items(['Points'], all_dict)
        table.close()

    def processing(self, data: np.ndarray, shared_vars: dict, sample_name: str = None):
        """
        Statistics the number of points distribution
        """
        data_name = self.proc_unit.assemble_function
        shared_vars.setdefault(data_name, dict())
        # data: Dict[str, np.ndarray] = self._load_sample_from_vars(sample, shared_vars)
        sample_name = os.path.splitext(os.path.basename(sample_name))[0]
        shared_vars[data_name][sample_name] = [data['points'].shape[0]]
