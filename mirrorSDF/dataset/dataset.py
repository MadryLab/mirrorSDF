from typing import Mapping, Optional

import cv2
import numpy as np
import torch as ch
from tqdm.autonotebook import tqdm

from .precomputed_background import PrecomputedBackgroundDataset
from .shot_metadata import ShotMetadata
from ..config import MirrorSDFConfig
from ..optical_models import Lens, Mirror
from ..utils.light_simulation import generate_bounce_LUT


def calculate_max_with_progress(array, progress_bar, chunk_size=100000):
    max_val = float('-inf')
    total_size = array.shape[0]

    for start in range(0, total_size, chunk_size):
        end = min(start + chunk_size, total_size)
        current_max = array[start:end].max()
        max_val = max(max_val, current_max)
        progress_bar.update(end - start)

    return max_val

class MirrorSDFDataset:

    storage_dtype = np.dtype([
        ('shot_id', np.uint16),
        ('coord', (np.uint16, 2)),
        ('measurement', (np.float32, 3)),
    ])

    def __init__(self,
                 env_memmap: np.ndarray,
                 shape_memmap: np.ndarray,
                 lens: Lens,
                 shots_metadata: Mapping[int, ShotMetadata],
                 image_width, image_height,
                 mirror_thickness,
                 mirror_corner_vertices,
                 object_center,
                 object_bounding_box,
                 min_distance_to_env,
                 max_distance_to_env,
                 eta,
                 largest_measurement,
                 smallest_measurement,
                 linear_tonemap_gamma: np.ndarray,
                 precomputed_bg: Optional[PrecomputedBackgroundDataset] = None):

        self.precomputed_bg = precomputed_bg

        self.shots = shots_metadata
        self.lens = lens

        self.image_width = image_width
        self.image_height = image_height

        self.mirror_thickness = mirror_thickness
        self.eta = eta
        self.mirror_corner_vertices = mirror_corner_vertices
        self.object_center = object_center
        self.object_bounding_box = object_bounding_box
        self.min_distance_to_env = np.array(min_distance_to_env)
        self.max_distance_to_env = np.array(max_distance_to_env)

        self.linear_tonemap_gamma = linear_tonemap_gamma

        self.smallest_measurement = smallest_measurement
        self.largest_measurement = largest_measurement

        self.env_memmap = env_memmap
        self.shape_memmap = shape_memmap

        # Normalization in log space
        dynamic_range = np.log(largest_measurement) - np.log(smallest_measurement)
        dynamic_range_center = 0.5 * (np.log(largest_measurement) + np.log(smallest_measurement))
        self.log_normalization_scale = 2 / dynamic_range
        self.log_normalization_offset = - dynamic_range_center

        self.world_scale_normalization_factor = self.compute_world_scale_normalization(object_bounding_box)
        self.mirror_offset_lut = generate_bounce_LUT(128, self.eta,
                                                     self.mirror_thickness * self.world_scale_normalization_factor)

        self.n_shots = max(self.shots.keys()) + 1

        self.directions_cam_space = self.precompute_directions()
        self.rotations, self.origins = self.precompute_transforms()
        self.environment_mapping = self.precompute_environment_mapping()

    @staticmethod
    def compute_world_scale_normalization(object_bounding_box) -> float:
        return 1 / (np.linalg.norm(object_bounding_box * np.array([0.5, 0.5, 1.0])))

    def linearize_prediction(self, rgb):
        rgb = rgb / self.log_normalization_scale - self.log_normalization_offset
        rgb = ch.exp(rgb)
        return rgb

    def logcompress_prediction(self, rgb):
        if not ch.is_tensor(rgb):
            rgb = ch.from_numpy(rgb.copy()).float()
        normalized_log_measurements = ch.log(rgb + self.smallest_measurement)
        normalized_log_measurements += self.log_normalization_offset
        normalized_log_measurements *= self.log_normalization_scale
        return normalized_log_measurements

    @property
    def num_env(self):
        return self.environment_mapping.max() + 1

    def precompute_environment_mapping(self):
        environment_mapping = np.zeros(self.n_shots, dtype=np.int32)
        for k, shot_info in self.shots.items():
            environment_mapping[k] = shot_info.env_id

        return environment_mapping

    def precompute_transforms(self):
        rotations = np.zeros((self.n_shots, 3, 3))
        origins = np.zeros((self.n_shots, 3))

        for k, shot_info in self.shots.items():
            rotation = cv2.Rodrigues(shot_info.r_vec)[0]
            rotations[k] = rotation
            origins[k] = -rotation.T @ shot_info.t_vec

        return rotations, origins.astype(np.float32)

    def precompute_directions(self):
        with tqdm(total=None, desc='Precomputing ray directions') as progress_bar:
            xes = np.arange(self.image_width, dtype=np.float32)
            yes = np.arange(self.image_height, dtype=np.float32)
            pixel_coords = np.stack(np.meshgrid(xes, yes), -1).reshape(-1, 2)

            directions_cam_space = np.ones((pixel_coords.shape[0], 3))
            directions_cam_space[:, :2] = cv2.undistortPoints(pixel_coords,
                                                              self.lens.intrinsic,
                                                              self.lens.distortions, None)[:, 0]
            directions_cam_space /= np.linalg.norm(directions_cam_space, axis=1)[:, None]
            directions_cam_space = directions_cam_space.astype(np.float32)
            return directions_cam_space.reshape(self.image_height, self.image_width, 3)

    @staticmethod
    def from_config(config: MirrorSDFConfig):
        lens = Lens.from_disk(config.calibration_files.lens)
        mirror = Mirror.from_disk(config.calibration_files.mirror)

        shots_metadata = {}
        for i in range(config.dataset.num_images):
            try:
                shots_metadata[i] = ShotMetadata.from_file(config.dataset.root, i)
            except FileNotFoundError:
                pass

        try:
            bg_ds = PrecomputedBackgroundDataset.from_config(config, create=False)
        except FileNotFoundError:
            bg_ds = None

        return MirrorSDFDataset(
            config.dataset.env_memmap,
            config.dataset.shape_memmap,
            lens,
            shots_metadata,
            config.dataset.images_width,
            config.dataset.images_height,
            mirror.thickness_mm,
            mirror.mirror_corners_clockwise,
            config.dataset.object_center,
            config.dataset.object_bounding_box,
            config.dataset.minimum_distance_to_env,
            config.dataset.maximum_distance_to_env,
            1 / mirror.ior,
            config.dataset.maximum_measurement,
            config.dataset.minimum_measurement,
            config.dataset.linear_tonemap_gamma,
            precomputed_bg=bg_ds
        )

