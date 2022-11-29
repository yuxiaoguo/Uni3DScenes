"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import re
from typing import List, Dict
from graphics_utils.config import DictRecursive


class ProcessUnit(DictRecursive):
    """
    Pipeline units
    """
    def __init__(self):
        super().__init__()
        self.assemble_function = str()
        self.attrs = dict()
        self.in_paths = list()
        self.out_paths = list()


class EntryConfig(DictRecursive):
    """
    Main entry of each task
    """
    def __init__(self):
        super().__init__()
        self.assemble_class = str()
        self.process_pipelines = list([ProcessUnit()])


class EnvsConfig(DictRecursive):
    """
    Global environments
    """
    def __init__(self):
        super().__init__()
        self.in_data_root = str()
        self.out_data_root = str()
        self.io_paths: Dict[str, str] = dict()

    def get_env_path(self, env_name: str):
        """
        Get the absolute folder path by the env name
        """
        if 'in_data_root' not in self.io_paths:
            self.io_paths['in_data_root'] = self.in_data_root
            self.io_paths['out_data_root'] = self.out_data_root
        rel_path = self.io_paths[env_name]
        while True:
            regex_pattern = r'\$.*\$'
            patterns = re.findall(regex_pattern, rel_path)
            if not patterns:
                break
            rel_path = rel_path.replace(patterns[0], self.io_paths[patterns[0][1:-1]])
        return rel_path


class StreamingTasks(DictRecursive):
    """
    Main entry of streaming tasks
    """
    def __init__(self):
        super().__init__()
        self.envs = EnvsConfig()
        self.streaming_lines = list([EntryConfig()])


class EntryBase:
    """
    The basic config of entry
    """
    def __init__(self, proc_units: List[ProcessUnit], envs: EnvsConfig) -> None:
        self.proc_units = proc_units
        self.envs = envs

    def execute_pipeline(self):
        """
        execute the data processing pipeline
        """
        for proc_unit in self.proc_units:
            proc_func = getattr(self, proc_unit.assemble_function)
            proc_func(proc_unit)
