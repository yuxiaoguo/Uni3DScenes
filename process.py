# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import argparse

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', \
    level=logging.DEBUG)


def parse_arguments():
    """
    Input kwargs
    """
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--cfg_path', required=True, type=str, \
        help='the config file')
    parser.add_argument('--data_root', required=True, type=str, \
        help='The root of raw dataset')
    return parser.parse_known_args()


if __name__ == '__main__':
    (args, unknown) = parse_arguments()
