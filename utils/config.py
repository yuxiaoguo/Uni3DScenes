"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import re
import multiprocessing as mp
from abc import abstractmethod
from typing import List, Dict

from graphics_utils import g_perf
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


class MPEntryBase(EntryBase):
    """
    The multi-process config of entry
    """
    def __init__(self, proc_units: List[ProcessUnit], envs: EnvsConfig) -> None:
        super().__init__(proc_units, envs)
        self._enable_mp = False
        self._num_worker = 8

    @abstractmethod
    def _sample_list(self):
        """
        Return the list of samples to be processed
        """

    def _merged_cross_processing(self, ipc_vars):
        """
        Merge all shared list information cross all processors
        """

    def _merged_within_processing(self, shared_vars, ipc_vars):
        """
        Merge all information within a processor
        """

    def _mp_execute_pipeline(self, samples, ipc_vars: List, worker_offset=0, worker_id=0):
        del worker_offset, worker_id
        shared_vars = dict()
        for sample in samples:
            for proc_unit in self.proc_units:
                proc_func = getattr(self, proc_unit.assemble_function)
                proc_func(sample, proc_unit, shared_vars)
        self._merged_within_processing(shared_vars, ipc_vars)

    def execute_pipeline(self):
        samples = self._sample_list()
        ipc_vars = mp.Manager().list()
        if self._enable_mp:
            g_perf.multiple_processor(self._mp_execute_pipeline, samples, workers=8, \
                args=tuple(ipc_vars))
        else:
            self._mp_execute_pipeline(samples, ipc_vars)
        self._merged_cross_processing(list(ipc_vars))
