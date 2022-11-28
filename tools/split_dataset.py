"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
import random
import shutil
import argparse


def split_dataset_with_ratio(src_folder: str, dst_folder:str, num_pieces: int, kept_pieces: int, \
    seed=666, scopes=['train']):
    """_summary_

    Args:
        src_folder (str): the source folder path
        dst_folder (str): the destination folder path
        num_pieces (int): the number of pieces to be divided of samples
        kept_pieces (int): the number of pieces to be kept after divided
        seed (int, optional): the random seed to shuffle the list. Defaults to 666.
    """
    random.seed(seed)
    phases = os.listdir(src_folder)

    for phase in phases:
        src_phase_path = os.path.join(src_folder, phase)
        samples = sorted(os.listdir(samples))
        random.shuffle(samples)
        p_pieces = num_pieces if phase in scopes else 1
        k_pieces = kept_pieces if phase in scopes else 1
        for p_idx in range(p_pieces):
            if p_idx > k_pieces:
                continue
            dst_phase_path = os.path.join(f'{dst_folder}_p{num_pieces}s{p_idx}', phase)
            os.makedirs(dst_phase_path, exist_ok=True)
            for s_idx in range(p_idx, len(samples), p_pieces):
                shutil.move(os.path.join(src_phase_path, samples[s_idx]), \
                    os.path.join(dst_phase_path, samples[s_idx]))

def parse_arguments():
    """
    Input kwargs
    """
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--src_folder', required=True, type=str, \
        help='the config file')
    parser.add_argument('--dst_folder', required=True, type=str, \
        help='The root of raw dataset')
    parser.add_argument('--num_pieces', required=True, type=int, \
        help='The number of pieces to be divided of samples')
    parser.add_argument('--kept_pieces', required=True, type=int, \
        help='The number of pieces to be kept after divided')
    parser.add_argument('--seed', required=True, type=int, \
        help='the random seed')
    return parser.parse_known_args()


if __name__ == '__main__':
    (args, unknown) = parse_arguments()
    split_dataset_with_ratio(args.src_folder, args.dst_folder, args.num_pieces, args.kept_pieces, \
        args.seed)
