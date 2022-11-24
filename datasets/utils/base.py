"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
from abc import abstractmethod

class DatasetBase:
    """
    The base class of dataset
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def view2pointcloud(self):
        """
        Construct 3D point cloud from views
        """
