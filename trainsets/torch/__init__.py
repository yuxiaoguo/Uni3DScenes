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

from datasets.utils.base import ProcessUnit


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


class TaskBase:
    """
    The base class for down-streaming tasks
    """
    def __init__(self, root_dir: str, proc_units: List[ProcessUnit]) -> None:
        self.root_dir = root_dir
        self.proc_units = proc_units

    @abstractmethod
    def segmentation(self, proc_unit: ProcessUnit):
        """
        Construct 3D point cloud from views
        """

    def execute_pipeline(self):
        """
        execute the data processing pipeline
        """
        for proc_unit in self.proc_units:
            proc_func = getattr(self, proc_unit.assemble_function)
            proc_func(proc_unit)


class PointCloudDownStreaming(TaskBase):
    """
    To deal with point cloud down-streaming task
    """
    def _segmentation_mp(self, samples: List[str], proc_unit: ProcessUnit, split=str, offset=0, \
        worker_id: int = 0):
        del proc_unit, offset, worker_id
        output_dir = os.path.join(self.root_dir, 'torch_s3d', split)
        os.makedirs(output_dir, exist_ok=True)

        remapper = np.ones(150, dtype=np.int32) * (-100)
        for i_id, s_id in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15,\
            16, 17, 18, 19, 22, 24, 25, 32, 34, 35, 38, 39, 40]):
            remapper[s_id] = i_id

        torch_writer = TorchWriter(output_dir)

        for sample in samples:
            sample_meta = np.load(os.path.join(self.root_dir, 'point_cloud', sample))
            points = sample_meta['points']
            colors = (sample_meta['colors'].astype(np.float32) / 127.5) - 1
            labels = remapper[sample_meta['labels']]
            torch_writer.write_item((points, colors, labels), os.path.splitext(sample)[0])

    def _split_samples(self, samples: List[str]):
        train_samples, val_samples, test_samples = list(), list(), list()
        for sample in samples:
            scene_id = int(sample.split('_')[1])
            if scene_id < 3000:
                train_samples.append(sample)
            elif scene_id < 3250:
                val_samples.append(sample)
            else:
                test_samples.append(sample) 
        return train_samples, val_samples, test_samples

    def segmentation(self, proc_unit: ProcessUnit):
        point_cloud_dir = os.path.join(self.root_dir, 'point_cloud')

        train_samples, val_samples, test_samples = self._split_samples(os.listdir(point_cloud_dir))
        g_perf.multiple_processor(self._segmentation_mp, train_samples, 8, (proc_unit, 'train'))
        g_perf.multiple_processor(self._segmentation_mp, val_samples, 8, (proc_unit, 'val'))
        g_perf.multiple_processor(self._segmentation_mp, test_samples, 8, (proc_unit, 'test'))
"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""