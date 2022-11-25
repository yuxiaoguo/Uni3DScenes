"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
from typing import List
from abc import abstractmethod

from graphics_utils.config import DictRecursive


class ProcessUnit(DictRecursive):
    """
    Pipeline units
    """
    def __init__(self):
        super().__init__()
        self.assemble_function = str()
        self.room_types = list([str()])


class DatasetBase:
    """
    The base class of dataset
    """
    def __init__(self, root_dir: str, proc_units: List[ProcessUnit]) -> None:
        self.root_dir = root_dir
        self.proc_units = proc_units

    @abstractmethod
    def view2pointcloud(self, proc_unit: ProcessUnit):
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
