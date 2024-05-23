from dataclasses import dataclass, field
from functools import cached_property
from os import path
from typing import Optional

import numpy as np


@dataclass
class DatasetConfig:
    """
    Configuration for managing dataset-specific parameters and paths.

    Attributes:
    -----------
    original_source : str
        Path to the original dataset source folder, where the RAW images where located
    num_images: int
        Number of images in the dataset
    num_environments: int
        Number of environments in the dataset
    images_height: int
        Height of the source images (potentially after halving during RAW Debayerization)
    images_width: int
        Width of the source images (potentially after halving during RAW Debayerization)
    source_extension : str
        File extension for source images (e.g., 'CR3').
    num_bracketed_shots : int
        Number of different exposure taken for each viewpoint.
    root : str
        Root directory where the processed dataset is stored.
    white_point : float
        The white point used to clip images before performing detection of the fiducials
    black_point : float
        The black point used to clip images before performing detection of the fiducials
    object_center : np.ndarray
        Center coordinates of the object of interest as a numpy array. Shape (3, ).
        Unit: mm
    object_bounding_box : np.ndarray
        array containing width (X), depth (Y), and height (Z) the dimensions enclosing the
        object we are scanning. Shape (3,). Unit: mm
    mirror_safety_padding : float
        Additional padding around the mirror to ensure we avoid training on bevels and borders.
        Unit: mm
    marker_safety_padding : float
        Additional padding around markers to ensure we do not train on the markers at all.
        Unit: mm
    minimum_distance_to_env : float
        The closest distance from the center of the object to the surrounding environment.
    maximum_distance_to_env : float
        largest from the center of the object to the surrounding environment.
    maximum_measurement: Optional[float]
        The largest measured radiance in the dataset (used for normalization). None if it hasn't been
        computed yet
    minimum_measurement: Optional[float]
        The smallest (nonzero!) measured radiance in the dataset (used for normalization). None if it hasn't been
        computed yet
    linear_tonemap_gamma : np.ndarray, default=np.array([0.5, 3])
        Gamma correction values used when training in linear space to map high dynamic ranges to [0, 1]
        Performs Ax^gamma
        - First entry is the exponent (gamma)
        - Second entry is the factor (A)
    """
    original_source: str
    num_images: int
    num_environments: int
    images_height: int
    images_width: int
    source_extension: str
    num_bracketed_shots: int
    root: str
    white_point: float
    black_point: float
    object_center: np.ndarray
    object_bounding_box: np.ndarray
    mirror_safety_padding: float
    marker_safety_padding: float
    minimum_distance_to_env: float
    maximum_distance_to_env: float
    maximum_measurement: Optional[float] = None
    minimum_measurement: Optional[float] = None
    linear_tonemap_gamma: np.ndarray = field(default_factory=lambda: np.array([0.5, 3], dtype=np.float32))

    @cached_property
    def env_memmap(self):
        return np.lib.format.open_memmap(path.join(self.root, 'env.npy'), mode='r')

    @cached_property
    def shape_memmap(self):
        return np.lib.format.open_memmap(path.join(self.root, 'shape.npy'), mode='r')
