from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch as ch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from ...dataset.loader_wrapper import LoaderWrapper

if TYPE_CHECKING:
    from .nerf_background import NerfBackground
    from ...dataset.batch import Batch


class LightSourceSampler:
    """
    A class for sampling light source positions and their weights from a NeRF++  model,
    given a specific environment. The sampler uses rejection sampling based on the emitted light intensity
    and density estimated by the NeRF model to sample points in space that are likely to contribute to
    the scene illumination.

    Parameters
    ----------
    model : NerfBackground
        The NeRF model from which to sample light sources.
    batch_size : int
        The number of samples to generate in each batch during sampling.
    min_bg : float
        The minimum distance from the origin beyond which to sample light sources.
    device : torch.device
        The device (CPU/GPU) on which to perform computations.
    sampling_eps : float, optional
        A small epsilon value to avoid division by zero during sampling, by default 1e-6.
    num_samples_density_estimate : float, optional
        The number of samples to use when estimating the density for normalization, by default 1e7.

    Attributes
    ----------
    device : torch.device
        The computation device.
    model : NerfBackground
        The NeRF model for background environment.
    min_bg : float
        Minimum background distance for sampling.
    num_envs : int
        The number of environment embeddings in the NeRF model.
    batch_size : int
        Batch size for sampling.
    sampling_eps : float
        Epsilon value for sampling.
    num_samples_density_estimate : float
        Number of samples for density estimation.
    c_estimates : dict
        Cached estimates of normalization constants for each environment.

    Methods
    -------
    get_random_samples(env_id: int):
        Generates random samples within the environment specified by env_id.
    estimate_c(env_id: int):
        Estimates the normalization constant for rejection sampling in the specified environment.
    collect_samples(num_samples: int, env_id: int):
        Collects a specified number of light source samples for a given environment.
    """

    def __init__(self, model: 'NerfBackground', batch: 'Batch',
                 sampling_eps: float = 1e-6, num_samples_density_estimate=1e6,
                 query_points_per_batch: int = 32,
                 samples_along_ray: int = 32,
                 power: float = 0.3):
        if batch['size'] % query_points_per_batch != 0:
            raise ValueError('query_points_per_batch has to be a divisor of batch size')
        self.batch = batch
        self.device = batch['device']
        self.model = model
        self.num_query_points_per_batch = query_points_per_batch
        self.num_envs = model.env_embedding.weight.shape[0]
        self.samples_along_ray = samples_along_ray
        self.batch_size = self.batch['size']
        self.power = power
        self.sampling_eps = sampling_eps
        self.num_samples_density_estimate = num_samples_density_estimate
        self.c_estimates = {}

    def get_random_samples(self, env_id: int) -> Tuple[ch.Tensor, ch.Tensor]:
        """
        Generates random samples within the specified environment. This function generates random
        directions and distances for sample points and queries the NeRF model for their emission
        and density. It computes weights for each sample based on emission and density.

        Parameters
        ----------
        env_id : int
            The ID of the environment from which to sample.

        Returns
        -------
        query_points : torch.Tensor
            The sampled points in the environment.
        weights : torch.Tensor
            The weights of the sampled points, based on their emission and density.
        """

        with ch.inference_mode():
            # We want the random sampling along the ray
            self.model.train()

            batch = self.batch
            min_bg = self.batch['background_radius_range'][0]

            random_directions = F.normalize(ch.randn(batch['size'], 3, device=self.device), p=2, dim=-1)
            coords = ch.rand(batch['size'], 3, device=batch['device']) * 2 - 1
            # We do not want to be under or exactly on the mirror
            coords[:, -1].abs_().add_(1e-5)
            batch['origin'] = coords
            batch['direction'] = random_directions
            batch['env_id'].data.fill_(env_id)
            LoaderWrapper.precompute_useful_batch_quantities(batch)

            points, log_emission, weights = self.model.evaluate_along_ray(batch, n_samples=self.samples_along_ray)
            emission = ch.exp(log_emission)
            contribution = (emission * weights[..., None]).sum(-1)

            return points, contribution

    def estimate_c(self, env_id: int):
        """
        Estimates the normalization constant for importance sampling within the specified environment.
        This is achieved by sampling a large number of points, computing their weights, and finding the
        maximum weight value to use as a normalization constant.

        Parameters
        ----------
        env_id : int
            The ID of the environment for which to estimate the normalization constant.

        Returns
        -------
        max_so_far : float
            The estimated normalization constant for the specified environment.
        """
        max_so_far = -np.inf
        with tqdm(desc=f'Estimating pdf coef for env {env_id}', total=self.num_samples_density_estimate) as pbar:
            done = 0
            while done < self.num_samples_density_estimate:
                _, weights = self.get_random_samples(env_id)
                done += weights.shape[0]
                max_so_far = max(max_so_far, weights.max().item())
                pbar.update(weights.shape[0])
        return max_so_far

    def collect_samples(self, num_samples: int, env_id: int):
        """
        Collects a specified number of light source samples for a given environment by performing
        rejection sampling based on the estimated normalization constant.

        Parameters
        ----------
        num_samples : int
            The number of samples to collect.
        env_id : int
            The ID of the environment from which to collect samples.

        Returns
        -------
        sampled_points : torch.Tensor
            The collected sample points, truncated to the specified number of samples.
        sampled_weights : torch.Tensor
            The weights of the collected samples.
        """

        if env_id not in self.c_estimates:
            self.c_estimates[env_id] = self.estimate_c(env_id)

        c = self.c_estimates[env_id]

        result_points, result_weights = [], []

        counter = 0
        with tqdm(desc=f"Sampling light sources for env {env_id}", total=num_samples) as pbar:
            while counter < num_samples:
                points, weights = self.get_random_samples(env_id)
                mask = ch.rand_like(weights) <= (weights / c)
                valid_points = points[mask]
                valid_weights = weights[mask]
                result_points.append(valid_points.data.cpu())
                result_weights.append(valid_weights.data.cpu())
                new_points = valid_points.shape[0]
                pbar.update(new_points)
                counter += new_points

        return ch.concat(result_points)[:num_samples], ch.concat(result_weights)[:num_samples]
