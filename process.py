"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import logging
import argparse

import entry

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)


def parse_arguments():
    """
    Input kwargs
    """
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--cfg_path', required=True, type=str, \
        help='the config file')
    parser.add_argument('--data_in', required=True, type=str, \
        help='The root of raw dataset')
    parser.add_argument('--data_out', required=True, type=str, \
        help='The root of output folder')
    return parser.parse_known_args()


if __name__ == '__main__':
    (args, unknown) = parse_arguments()

    entry.processing_entries(cfg_path=args.cfg_path, data_in=args.data_in, \
        data_out=args.data_out)
