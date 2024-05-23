from typing import Iterator

import torch as ch
from torch.utils.data import DataLoader

from ..dataset.batch import Batch
from ..utils.geometry import intersect_with_sphere
from ..utils.light_simulation import mirror_bounce_if_intersects


class LoaderWrapper:

    def __init__(self, loader: DataLoader[Batch], device: ch.device):
        self.loader = loader
        self.device = device

    def __iter__(self) -> Iterator[Batch]:
        for pre_batch in self.loader:
            batch = {}

            for k, v in pre_batch.items():
                if ch.is_tensor(v):
                    v = v.to(self.device, non_blocking=True).detach()
                batch[k] = v

            self.precompute_useful_batch_quantities(batch)
            batch['device'] = self.device
            yield batch

    def __len__(self):
        return len(self.loader)

    @staticmethod
    def compute_ray_bounce(batch: Batch):
        origin = batch['origin']

        exit_bounce_coord, bounce_direction, enter_bounce_coord = mirror_bounce_if_intersects(
            origin, batch['direction'], batch['mirror_normal'],
            batch['mirror_vertices'], batch['mirror_thickness'], batch['eta'])

        batch['exit_bounce_coord'] = exit_bounce_coord
        batch['bounce_direction'] = bounce_direction
        batch['enter_bounce_coord'] = enter_bounce_coord
        batch['t_bounce'] = (enter_bounce_coord - origin).norm(p=2, dim=1)[:, None]

    @staticmethod
    def compute_ray_range_background(batch: Batch):
        bounce_coords = batch['exit_bounce_coord']
        bounce_direction = batch['bounce_direction']

        # Unpack the background radius range from the batch
        background_start_radius, background_end_radius = batch['background_radius_range']

        # Intersect rays with the start and end background spheres
        _, start_bg_ray = intersect_with_sphere(bounce_coords, bounce_direction,
                                                radius=background_start_radius)
        _, end_bg_ray = intersect_with_sphere(bounce_coords, bounce_direction,
                                              radius=background_end_radius)

        start_bg_ray: ch.Tensor = ch.nan_to_num(start_bg_ray, 0)
        # We make sure of offset the bounce
        batch['background_t_range'] = (start_bg_ray + batch['t_bounce'], end_bg_ray + batch['t_bounce'])

    @staticmethod
    def compute_ray_range_foreground(batch: Batch):
        exit_loc: ch.Tensor = batch['exit_bounce_coord']
        exit_direction: ch.Tensor = batch['bounce_direction']
        t_bounce: ch.Tensor = batch['t_bounce']

        radius = 1.0  # Foreground is normalized to have a radius of 1

        primary_entrance, primary_exit = intersect_with_sphere(batch['origin'], batch['direction'], radius=radius)
        primary_entrance = ch.nan_to_num(primary_entrance, ch.inf)  # We never enter

        secondary_entrance, secondary_exit = intersect_with_sphere(exit_loc, exit_direction, radius=radius)
        secondary_entrance = ch.nan_to_num(secondary_entrance, ch.inf)  # We never enter
        secondary_entrance = secondary_entrance.clip(min=0)  # Can't consider what happens before the bounce
        secondary_entrance = secondary_entrance + t_bounce
        secondary_exit = secondary_exit.clip(min=0)  # Can't consider what happens before the bounce
        secondary_exit = secondary_exit + t_bounce

        true_entrance: ch.Tensor = ch.where(primary_entrance < t_bounce,
                                            primary_entrance,
                                            secondary_entrance)

        true_exit: ch.Tensor = ch.where(primary_exit < t_bounce,
                                        primary_exit,
                                        secondary_exit)

        batch['foreground_t_range'] = (
            true_entrance,
            true_exit
        )

    @staticmethod
    def precompute_useful_batch_quantities(batch):
        LoaderWrapper.compute_ray_bounce(batch)
        LoaderWrapper.compute_ray_range_background(batch)
        LoaderWrapper.compute_ray_range_foreground(batch)
