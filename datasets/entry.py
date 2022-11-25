"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
# pylint: disable=unused-import
from graphics_utils.config import DictRecursive

from .utils.base import DatasetBase, ProcessUnit
from .structured3d import Structured3DDataGen


class EntryConfig(DictRecursive):
    """
    Main entry
    """
    def __init__(self):
        super().__init__()
        self.assemble_class = str()
        self.process_pipelines = list([ProcessUnit()])


def processing_entry(cfg_path: str, data_root: str):
    """
    The entry of dataset processing

    Args:
        cfg_path (str): config path
        data_root (str): data root path
    """
    entry_config = EntryConfig()
    entry_config.load_from_yaml(cfg_path)

    assemble_instance: DatasetBase = globals()[entry_config.assemble_class](data_root, \
        entry_config.process_pipelines)

    assemble_instance.execute_pipeline()
