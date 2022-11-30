"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
from typing import Dict

import openpyxl


class DistributionTable:
    """
    The IO interface for distribution table
    """
    def __init__(self, excel_path: str) -> None:
        self.excel_path = excel_path

    def write(self, label_set, overall=None, items: Dict=None):
        """
        Write to a excel file
        """
        dir_path = os.path.dirname(self.excel_path)
        os.makedirs(dir_path, exist_ok=True)

        work_book = openpyxl.Workbook()
        work_book.remove(work_book.active)

        if overall is not None:
            sheet_oa = work_book.create_sheet('Overall')
            for c_idx, label in enumerate(label_set):
                sheet_oa.cell(1, c_idx + 1, label)
            for c_idx, dist in enumerate(overall):
                sheet_oa.cell(2, c_idx + 1, dist)

        if items is not None:
            sheet_items = work_book.create_sheet('Items')
            for c_idx, label in enumerate(label_set):
                sheet_items.cell(1, c_idx + 2, label)
            for r_idx, (r_key, r_value) in enumerate(items.items()):
                sheet_items.cell(r_idx + 2, 1, r_key)
                for c_idx, c_v in enumerate(r_value):
                    sheet_items.cell(r_idx + 2, c_idx + 2, c_v)
        work_book.save(self.excel_path)

    def read(self):
        """
        Read from a excel file
        """
        pass
