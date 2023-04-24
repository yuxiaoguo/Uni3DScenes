"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import numpy as np


class Annotations:
    """
    Annotation information
    """
    def __init__(self) -> None:
        self.gt_num = 0
        self.name = list()
        self.location = list()
        self.dimensions = list()
        self.gt_boxes_upright_depth = list()
        self.unaligned_location = list()
        self.unaligned_dimensions = list()
        self.unaligned_gt_boxes_upright_depth = list()
        self.index = list()
        self.classes = list()
        self.axis_align_matrix = list()

    def dump(self):
        """
        Dump information into dict
        """
        anno_dict = dict()
        anno_dict['gt_num'] = int(self.gt_num)
        anno_dict['name'] = np.asarray(self.name)
        anno_dict['location'] = np.asarray(self.location, dtype=np.float64)
        anno_dict['dimensions'] = np.asarray(self.dimensions, dtype=np.float64)
        anno_dict['gt_boxes_upright_depth'] = np.asarray(self.gt_boxes_upright_depth, \
            dtype=np.float64)
        anno_dict['unaligned_location'] = np.asarray(self.unaligned_location, \
            dtype=np.float64)
        anno_dict['unaligned_dimensions'] = np.asarray(self.unaligned_dimensions, \
            dtype=np.float64)
        anno_dict['unaligned_gt_boxes_upright_depth'] = np.asarray(
            self.unaligned_gt_boxes_upright_depth, dtype=np.float64)
        anno_dict['index'] = np.asarray(self.index, dtype=np.int32)
        anno_dict['class'] = np.asarray(self.classes, dtype=np.int64)
        anno_dict['axis_align_matrix'] = np.asarray(self.axis_align_matrix, dtype=np.float64)
        return anno_dict
