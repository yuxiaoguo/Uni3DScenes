"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
from abc import abstractmethod
from typing import List, Tuple

import torch
import numpy as np

from graphics_utils import g_perf

from utils.config import ProcessUnit


class BaseWriter:
    """
    The base writer to be implemented by PyTorch/TensorFlow
    """
    def __init__(self, out_dir, **kwargs) -> None:
        del kwargs
        self.out_dir = out_dir

    def open(self):
        """
        Open the dataset
        """

    def close(self):
        """
        Close the dataset
        """

    @abstractmethod
    def write_item(self, data_item: Tuple, data_alias: str = None):
        """
        Write a single item
        """


class TorchWriter(BaseWriter):
    """
    Torch writer
    """
    def write_item(self, data_item: Tuple, data_alias: str = None):
        assert isinstance(data_item, (tuple, list))
        assert data_alias is not None
        torch.save(data_item, os.path.join(self.out_dir, f'{data_alias}_seg.pth'))
