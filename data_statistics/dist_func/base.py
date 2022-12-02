"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
from typing import List

import numpy as np

from utils.config import ProcessUnit, EnvsConfig


class DistFuncBase:
    """
    The basic class of distribution statistics
    """
    def __init__(self, proc_unit: ProcessUnit, envs: EnvsConfig) -> None:
        self.proc_unit = proc_unit
        self.envs = envs

    def pre(self):
        """
        pre-processing of distribution statistics
        """

    def processing(self, data: np.ndarray, shared_vars: dict, sample_name: str = None):
        """
        processing of distribution statistics
        """

    def post(self, ipc_vars: List):
        """
        Post processing of distribution statistics. Post processing may be happened after multi-
            processing. The IPC vars will gather all results collected from each processor.
        """
