"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
from ..utils import base


class Structured3DDataGen(base.DatasetBase):
    """
    Dataset generation for Structured3D
    """
    def __init__(self) -> None:
        super().__init__()

    def view2pointcloud(self):
        return super().view2pointcloud()
