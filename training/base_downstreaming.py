"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
from abc import abstractmethod

from utils.config import ProcessUnit, EntryBase


class DownStreamingBase(EntryBase):
    """
    The base class for down-streaming tasks
    """
    @abstractmethod
    def segmentation(self, proc_unit: ProcessUnit):
        """
        Construct 3D point cloud from views
        """
