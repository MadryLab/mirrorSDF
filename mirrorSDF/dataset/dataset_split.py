from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch as ch
from torch.utils.data import DataLoader, Dataset, BatchSampler, SequentialSampler

from .batch import Batch
from .loader_wrapper import LoaderWrapper
from .sampler_with_replacement import SamplerWithReplacement
from ..utils.geometry import quads_to_tris, compute_mirror_normal

if TYPE_CHECKING:
    from .dataset import MirrorSDFDataset
    from ..config.training import TrainingConfig


class MirrorSDFDatasetSplit(Dataset):

    def __init__(self, dataset: 'MirrorSDFDataset', rows: np.ndarray):
        super().__init__()
        self.dataset = dataset
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def create_loader(self, config: 'TrainingConfig', device: ch.device, shuffle=True) -> Iterable:
        if shuffle:
            sampler = SamplerWithReplacement(self, config.batch_size)
            result = DataLoader(self,
                                batch_size=None,
                                sampler=sampler,
                                num_workers=config.num_workers,
                                pin_memory=True)
        else:
            sampler = BatchSampler(SequentialSampler(self), batch_size=config.batch_size, drop_last=False)
            result = DataLoader(self,
                                batch_size=None,
                                sampler=sampler,
                                num_workers=config.num_workers,
                                pin_memory=True)

        return LoaderWrapper(result, device=device)

    def extract_single_image(self, image_index: int) -> 'MirrorSDFDatasetSplit':
        """
        Extracts a subset of the dataset corresponding to a single image based on the given image index.

        This method locates the range of rows in the dataset associated with the specified image index by searching
        for the start and end positions where the image index appears in the 'shot_id' column of the dataset's metadata.
        It then creates and returns a new dataset split object containing only the data for the specified image.

        Parameters
        ----------
        image_index : int
            The index of the image to extract from the dataset.

        Returns
        -------
        MirrorSDFDatasetSplit
            A new dataset split object containing only the rows associated with the specified image index.

        """
        low_ix = np.searchsorted(self.rows['shot_id'], image_index)
        high_ix = low_ix + np.searchsorted(self.rows['shot_id'][low_ix:], image_index, side='right')
        return MirrorSDFDatasetSplit(self.dataset, self.rows[low_ix:high_ix])

    def __getitem__(self, indices) -> Batch:

        if isinstance(indices, list):
            indices = np.array(indices)
        elif isinstance(indices, int):
            indices = np.array([indices])
        elif isinstance(indices, ch.Tensor):
            indices = indices.numpy()

        selected_rows = self.rows[indices]
        selected_shots = selected_rows['shot_id']

        selected_rotations = self.dataset.rotations[selected_shots]
        offset = - self.dataset.object_center[None, :].astype(np.float32)

        # Retrieve directions for each pixel
        cx, cy = selected_rows['coord'].T
        directions_cam_space = self.dataset.directions_cam_space[cy, cx]
        direction_world_space = np.einsum('bcw,bc->bw',
                                          selected_rotations,
                                          directions_cam_space)

        env = self.dataset.environment_mapping[selected_shots]
        factor = np.float32(self.dataset.world_scale_normalization_factor)

        mirror_corners = (self.dataset.mirror_corner_vertices + offset) * factor
        mirror_normal = compute_mirror_normal(mirror_corners)
        mirror_vertices = quads_to_tris(mirror_corners[None])

        mirror_normal = mirror_normal[None]

        log_measurement = ch.log(ch.from_numpy(selected_rows['measurement'].copy()).float()
                                 + self.dataset.smallest_measurement)

        tonemapped_measurements = selected_rows['measurement'] ** self.dataset.linear_tonemap_gamma[0]
        tonemapped_measurements *= self.dataset.linear_tonemap_gamma[1]
        tonemapped_measurements = ch.from_numpy(tonemapped_measurements).float().clip(0, 1)

        result: Batch = {
            'device': ch.device('cpu'),
            'size': len(indices),
            'cx': ch.from_numpy(cx.astype(np.int16)),
            'cy': ch.from_numpy(cy.astype(np.int16)),
            'indices': ch.from_numpy(indices),
            'origin': ch.from_numpy((self.dataset.origins[selected_shots] + offset) * factor).float(),
            'direction': ch.from_numpy(direction_world_space).float(),
            'offset_lut': self.dataset.mirror_offset_lut,
            'env_id': ch.from_numpy(env),
            'object_bounding_sphere_radius': ch.tensor(np.max(self.dataset.object_bounding_box) * factor,
                                                       dtype=ch.float32),
            'mirror_vertices': ch.from_numpy(mirror_vertices).float(),
            'mirror_normal': ch.from_numpy(mirror_normal).float(),
            'mirror_thickness': float(self.dataset.mirror_thickness * factor),
            'eta': float(self.dataset.eta),
            'log_measurement': log_measurement,
            'tonemapped_measurement': tonemapped_measurements,
            'background_radius_range': (
                float(self.dataset.min_distance_to_env * factor),
                float(self.dataset.max_distance_to_env * factor),
            )
        }

        if self.dataset.precomputed_bg is not None:
            result['precomputed_background'] = ch.stack([
                self.dataset.precomputed_bg[x] for x in indices
            ])

        return result
