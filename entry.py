"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
# pylint: disable=unused-import
from typing import Dict

import yaml

from datasets.utils.base import DatasetBase, EntryConfig
from datasets.structured3d import Structured3DDataGen
from trainsets.torch import PointCloudDownStreaming


def processing_entry(cfg_dict: Dict, data_root: str):
    """
    Process a single entry
    """
    entry_config = EntryConfig()
    entry_config.load(cfg_dict)

    assemble_instance: DatasetBase = globals()[entry_config.assemble_class](data_root, \
        entry_config.process_pipelines)

    assemble_instance.execute_pipeline()

def processing_entries(cfg_path: str, data_root: str):
    """
    The entry of dataset processing

    Args:
        cfg_path (str): config path
        data_root (str): data root path
    """
    with open(cfg_path, encoding='utf-8') as cfg_fp:
        cfg_dict = yaml.load(cfg_fp, Loader=yaml.BaseLoader)

    for e_key in ['raw_data_process', 'train_sets_process']:
        if e_key not in cfg_dict:
            continue
        processing_entry(cfg_dict[e_key], data_root)
