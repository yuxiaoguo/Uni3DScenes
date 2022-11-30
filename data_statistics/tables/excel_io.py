"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os

import openpyxl


class DistributionTable:
    """
    The IO interface for distribution table
    """
    def __init__(self, excel_path: str) -> None:
        self.excel_path = excel_path

    def write(self, label_set, overall, items=None):
        """
        Write to a excel file
        """
        assert items is None
        dir_path = os.path.dirname(self.excel_path)
        os.makedirs(dir_path, exist_ok=True)

        work_book = openpyxl.Workbook()
        work_book.remove(work_book.active)
        sheet = work_book.create_sheet('Overall', 0)

        for c_idx, label in enumerate(label_set):
            sheet.cell(1, c_idx + 1, label)
        for c_idx, dist in enumerate(overall):
            sheet.cell(2, c_idx + 1, dist)
        work_book.save(self.excel_path)

    def read(self):
        """
        Read from a excel file
        """
        pass
