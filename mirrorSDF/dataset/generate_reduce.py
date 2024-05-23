import argparse
import warnings
from os import path
from os import remove
from typing import Tuple

import numba as nb
import numpy as np
from tqdm import tqdm

from ..config import MirrorSDFConfig
from ..dataset.dataset import MirrorSDFDataset


@nb.njit
def non_zero_min(array):
    current_min = np.inf
    for v in array.ravel():
        if v != 0:
            current_min = min(current_min, v)
    return current_min


def merge_slices(prefix: str, config: MirrorSDFConfig, remove_shards: bool = False) -> Tuple[float, float]:
    """
    Merges individual dataset shards into a single cohesive file.

    Parameters
    ----------
    prefix : str
        The prefix identifying the dataset shards to be merged.
    config : MirrorSDFConfig
        The config object
    remove_shards : bool, optional
        If True, deletes the individual shard files after merging. Defaults to False.

    Returns
    -------
    Tuple[float, float]
        - The minimum measurement seen in the dataset
        - The maximum measurement seen in the dataset
    """

    # Determine the size needed
    num_data_points = 0
    for image_id in range(config.dataset.num_images):
        try:
            file_name = path.join(config.dataset.root, f'partial_{prefix}_{image_id}.npy')
            dataset_shard = np.lib.format.open_memmap(file_name, mode='r')
            num_data_points += dataset_shard.shape[0]
        except FileNotFoundError:
            pass

    if num_data_points == 0:
        warnings.warn(f"Nothing to merge for dataset {prefix}")
        return +np.inf, -np.inf

    # Allocate a file
    final_dataset = np.lib.format.open_memmap(path.join(config.dataset.root, f'{prefix}.npy'),
                                              dtype=MirrorSDFDataset.storage_dtype,
                                              shape=(num_data_points,), mode='w+')

    # Copy the data
    i = 0
    max_measurement = -np.inf
    min_measurement = np.inf
    for image_id in tqdm(range(config.dataset.num_images), desc=f"Aggregating slices for the {prefix} dataset"):
        try:
            file_name = path.join(config.dataset.root, f'partial_{prefix}_{image_id}.npy')
            dataset_shard = np.lib.format.open_memmap(file_name, mode='r', dtype=MirrorSDFDataset.storage_dtype)
        except FileNotFoundError:
            continue
        max_measurement = max(max_measurement, dataset_shard['measurement'].max())
        min_measurement = min(min_measurement, non_zero_min(dataset_shard['measurement']))

        n = dataset_shard.shape[0]
        final_dataset[i:i+n] = dataset_shard
        i += n

        # Remove the shard file if the flag is set
        if remove_shards:
            remove(file_name)

    assert i == num_data_points, "Mismatch in expected and actual data points after merging."

    return min_measurement, max_measurement


def main(cli_args):
    print(f"Config file path: {cli_args.config_file}")

    config = MirrorSDFConfig.from_disk(cli_args.config_file)

    # Pass the remove_shards argument based on the CLI flag
    min_shape, max_shape = merge_slices('shape', config, cli_args.remove_shards)
    min_env, max_env = merge_slices('env', config, cli_args.remove_shards)

    config.dataset.maximum_measurement = max(max_env, max_shape)
    config.dataset.minimum_measurement = min(min_env, min_shape)
    config.to_disk(cli_args.config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to merge all slices of the dataset into single cohesive files."
    )

    parser.add_argument("-c", "--config_file", required=True,
                        help="Path to the configuration file.")
    parser.add_argument("-r", "--remove_shards", action='store_true',
                        help="Remove individual shard files after merging.")

    args = parser.parse_args()

    main(args)
