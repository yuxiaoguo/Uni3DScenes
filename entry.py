"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
# pylint: disable=unused-import
from typing import Dict

import yaml

from utils.config import EntryConfig, EnvsConfig, StreamingTasks
from datasets.base_dataset import DatasetBase
from datasets.structured3d import Structured3DDataGen
from training.point_cloud import PointCloudDownStreaming
from data_statistics import PointCloudStatistics


def processing_entry(entry_config: EntryConfig, envs: EnvsConfig):
    """
    Process a single entry
    """
    assemble_instance: DatasetBase = globals()[entry_config.assemble_class](\
        entry_config.process_pipelines, envs)

    assemble_instance.execute_pipeline()

def processing_entries(cfg_path: str, data_in: str, data_out: str):
    """
    The entry of dataset processing

    Args:
        cfg_path (str): config path
        data_in (str): data root path
    """
    task_configs = StreamingTasks()
    task_configs.load_from_yaml(cfg_path)
    task_configs.envs.in_data_root = data_in
    task_configs.envs.out_data_root = data_out

    for t_cfg in task_configs.streaming_lines:
        processing_entry(t_cfg, task_configs.envs)
