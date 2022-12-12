"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
from typing import Dict, List

import numpy as np

from graphics_utils import g_cfg, g_io

from utils import palette
from utils.config import ProcessUnit, EnvsConfig
from .base import FuncBase


class VisAttrs(g_cfg.DictRecursive):
    """
    Attrs properties
    """
    def __init__(self):
        super().__init__()
        self.key_pos = str()
        self.key_vis = str()
        self.color_scheme = str()


class TopViewVisualization(FuncBase):
    """
    Top view visualization
    """
    class Attrs(g_cfg.DictRecursive):
        """
        Attrs properties
        """
        def __init__(self):
            super().__init__()
            self.stride = 1
            self.resolution = 1.
            self.key_pos = str()
            self.key_vis = str()
            self.top_axis = str()

    def __init__(self, proc_unit: ProcessUnit, envs: EnvsConfig) -> None:
        super().__init__(proc_unit, envs)
        self.attrs = __class__.Attrs().load(proc_unit.attrs)

    def processing(self, data: Dict[str, np.ndarray], shared_vars: dict, sample_name: str = None):
        points = data[self.attrs.key_pos]
        
        return super().processing(data, shared_vars, sample_name)


class PLY3DVisualization(FuncBase):
    """
    PLY 3D visualization
    """
    def __init__(self, proc_unit: ProcessUnit, envs: EnvsConfig) -> None:
        super().__init__(proc_unit, envs)
        self.attrs = VisAttrs().load(proc_unit.attrs)

    def processing(self, data: List[np.ndarray], shared_vars: dict, sample_name: str = None):
        out_folder = self.envs.get_env_path(self.proc_unit.out_paths[0])
        os.makedirs(out_folder, exist_ok=True)
        sample_alias, _ = os.path.splitext(os.path.basename(sample_name))
        file_path = os.path.join(out_folder, f'{sample_alias}_{self.proc_unit.name}.ply')
        pos_xyz = data[0][..., :3]
        if len(data) == 2:
            color_palette = getattr(palette, f'{self.attrs.color_scheme}_color_palette')()
            color = np.asarray(color_palette)[data[1]]
        else:
            color = data[0][..., 3:]
        g_io.PlyIO().add_vertices(pos_xyz, color).dump(file_path)
        shared_vars.setdefault(self.proc_unit.name, list())
