"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree

from graphics_utils import g_cfg

from .base import DistFuncBase
from ..tables.excel_io import DistributionTable


class NearestNeighborDistribution(DistFuncBase):
    """
    Distribution of the distance of nearest neighborhood
    """
    class Attrs(g_cfg.DictRecursive):
        """
        Function Attrs
        """
        def __init__(self):
            super().__init__()
            self.max = 0.
            self.min = 0.
            self.stride = 1.

    def processing(self, data: np.ndarray, shared_vars: Dict[str, list], sample_name: str = None):
        del sample_name
        attrs = __class__.Attrs().load(self.proc_unit.attrs)
        assert attrs.max > 0 and attrs.stride > 0
        bin_size = int(attrs.max / attrs.stride)
        bin_scale = (1 / attrs.max) * bin_size

        points = data['points']
        tree = cKDTree(points)
        top2_dist, _ = tree.query(points, k=2)
        # neigh_dist: np.ndarray = np.clip(top2_dist[..., 1], attrs.min, attrs.max)
        neigh_scaled_dist: np.ndarray = np.clip(top2_dist[..., 1] * bin_scale, 0, bin_size - 1)
        dist_bincount = np.bincount(neigh_scaled_dist.astype(np.int64), minlength=bin_size)

        shared_vars.setdefault(self.proc_unit.assemble_function, list())
        shared_vars[self.proc_unit.assemble_function].append(dist_bincount)

    def post(self, ipc_vars: List):
        attrs = __class__.Attrs().load(self.proc_unit.attrs)
        bin_size = int(attrs.max / attrs.stride)
        bin_count = np.zeros(bin_size, dtype=np.int64)

        for ipc_var in ipc_vars:
            for item in ipc_var:
                bin_count += item

        pdf = bin_count.astype(np.float32) / np.sum(bin_count, dtype=np.float32)
        cdf = np.cumsum(pdf)

        labels = np.arange(bin_size, dtype=np.float32) * attrs.stride

        out_path = self.envs.get_env_path(self.proc_unit.out_paths[0])
        table = DistributionTable(out_path)
        table.write_overall(labels, pdf, 'PDF')
        table.write_overall(labels, cdf, 'CDF')
        table.close()
