from typing import Tuple, TYPE_CHECKING

import torch as ch
import torch.nn.functional as F

from ..architectures import MLPWithSkip
from ...dataset.loader_wrapper import LoaderWrapper
from ...utils.sampling import sample_nerf_pp

if TYPE_CHECKING:
    from ...dataset.batch import Batch


class NerfBackground(ch.nn.Module):
    """
       A module for modeling an ensemble of environments as a Neural Radiance Fields (NeRF)
       The main application is to model the actual same location but with different lighting
       conditions.

       Attributes
       ----------
       env_embedding : torch.nn.Embedding
           An embedding layer for encoding environment IDs into a high-dimensional vector space.
       coord_embedding : Module
           A module for embedding 3D coordinates into a high-dimensional space.
       input_dim : int
           The dimensionality of the input to the MLP (3 + environment embedding dimension +
           coordinate embedding output features).
       inner : MLPWithSkip
           A multi-layer perceptron with skip connections for processing the combined input features.
       max_coord : torch.Tensor
           A buffer storing the maximum coordinate value for normalization.
        train_density_noise: float, default=0
            Amount of noise to add to the density of the model (before the activation)

       Methods
       -------
       forward(points: torch.Tensor, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
           Processes input points and environment IDs to predict emission and density values.
       render(batch: Dict[str, torch.Tensor], n_samples: int, hierarchical: bool = True) -> torch.Tensor:
           Renders an image by sampling points along rays and computing emissions weighted by density and distance.
       """
    def __init__(self, n_envs, env_embedding_size, coord_embedder,
                 max_coord,
                 train_density_noise: float = 0.0,
                 **base_model_args):
        super().__init__()

        self.env_embedding = ch.nn.Embedding(n_envs, env_embedding_size)
        self.coord_embedding = coord_embedder
        self.input_dim = 4 + self.env_embedding.embedding_dim + coord_embedder.output_features
        self.inner = MLPWithSkip(self.input_dim, 4, **base_model_args)
        self.training_noise = train_density_noise

        self.register_buffer('max_coord', ch.tensor(max_coord))

    def forward(self, points: ch.Tensor, env_ids: ch.Tensor, min_norm: float) -> Tuple[ch.Tensor, ch.Tensor]:
        """
        Forward pass to predict emission and density at specified points for given environment IDs.

        Parameters
        ----------
        points : torch.Tensor
            The 3D points in space at which to predict emission and density, shaped (B, N, 3).
        env_ids : torch.Tensor
            The environment IDs for each point, shaped (B,).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing tensors for emission (B, N, 3) and density (B, N, 1).
        """

        batch_size, n_samples, _ = points.shape
        p_flat = points.reshape(-1, 3)

        norms = ch.norm(p_flat, p=2, dim=-1, keepdim=True)
        normalized_p = p_flat / norms
        inv_depth = min_norm / norms

        input_with_depth = ch.cat([normalized_p, inv_depth], dim=1)

        embedded_coords = self.coord_embedding(input_with_depth)

        env_feature = self.env_embedding(env_ids)
        env_feature = env_feature[:, None].expand(-1, n_samples, -1).reshape(-1, env_feature.shape[1])
        all_inputs = ch.cat([input_with_depth, env_feature, embedded_coords], dim=1)

        prediction = self.inner(all_inputs).reshape(batch_size, n_samples, 4)
        emission, density = ch.split(prediction, [3, 1], dim=2)

        if self.training and self.training_noise > 0.0:
            density = density + ch.randn_like(density) * self.training_noise

        return emission, F.softplus(density, beta=5)

    def render(self, batch: 'Batch', n_samples: int) -> Tuple[ch.Tensor, ch.Tensor]:
        """
        Renders an image by sampling background points and computing their log_emission and density.

        Parameters
        ----------
        batch : Batch
            Batch object containing the information about the rays to evaluate
        n_samples : int
            The number of samples to take along each ray.

        Returns
        -------
        torch.Tensor
            The **log** RGB luminance coming from the environment at each ray, shaped (B, 3), where B is the batch size.
        torch.Tensor
            The weighted length of the ray before hitting an obstacle
        """
        points, log_emission, weights = self.evaluate_along_ray(batch, n_samples)

        log_w = ch.log(1e-7 + weights[..., None])
        em = log_emission + log_w
        rgb = ch.logsumexp(em, dim=1)

        distances = (
                (batch['exit_bounce_coord'][:, None] - points).norm(dim=-1)
                + batch['t_bounce']
        ).detach()

        # Depth
        depth = (distances * weights).sum(dim=1)

        return rgb, depth

    def evaluate_along_ray(self, batch: 'Batch', n_samples: int) -> Tuple[ch.Tensor, ch.Tensor, ch.Tensor]:
        """
        Evaluates the NeRF model along specified rays in a batch. This function samples points
        along each ray, computes the log_emission and density at these points, and calculates the
        weights for each sampled point based on its density.

        Parameters
        ----------
        batch : Batch
            Batch object containing the information about the rays to evaluate
        n_samples : int
            The number of points to sample along each ray.

        Returns
        -------
        points : torch.Tensor
            The sampled points along each ray. Shape: [batch_size, n_samples, 3].
        log_emission : torch.Tensor
            The log emission (color) values at the sampled points. Shape: [batch_size, n_samples, 3].
        weights : torch.Tensor
            The computed weights for each sampled point, based on its density and the overall
            ray length. Shape: [batch_size, n_samples].
        """
        batch_size = batch['origin'].shape[0]
        points = sample_nerf_pp(batch, n_samples, randomized=self.training)
        min_bg, max_bg = batch['background_radius_range']
        scale = max_bg - min_bg
        log_emission, density = self(points, batch['env_id'], min_norm=min_bg)
        real_t = (points - points[:, 0][:, None]).norm(dim=-1)
        log_emission = log_emission.reshape(batch_size, -1, 3)
        density = density.reshape(batch_size, -1)
        weights = compute_nerf_weights(real_t / scale, density)
        return points, log_emission, weights

    def compute_diffuse_irradiance(self, origins: ch.Tensor, normals: ch.Tensor, env_ids: ch.Tensor,
                                   batch: 'Batch', num_dir_samples: int = 1024, num_depth_samples: int = 64,
                                   log_space: bool = False) -> ch.Tensor:
        """
        Compute the diffuse irradiance for a specified set of viewpoints, environments,
        hitting a surface at given normals.

        Notes
        -----
        This uses Monte-Carlo estimation. Results will incur a significant amount of noise.
        Especially if the environment contains small, powerful light sources.

        Parameters
        ----------
        origins : ch.Tensor
            A tensor of shape (N, 3) representing the origins of the viewpoints.
        normals : ch.Tensor
            A tensor of shape (N, M, 3) representing the normals from each viewpoint.
        env_ids : ch.Tensor
            A tensor of shape (N,) containing environment map IDs for each viewpoint.
        batch : Batch
            A Batch object to store intermediate computation results for rendering.
        num_dir_samples : int, optional. Default: 1024
            The number of directions to sample per viewpoint for the Monte Carlo estimation.
            We recommend using many more (>256k) for very accurate results, but it wouldn't fit on most GPU.
            It's up to the user to average further to reach desired levels of noise.
        num_depth_samples : int. Default: 64
            The number of depth samples to use in rendering.
        log_space : bool, optional. Default: False
            If True, returns the computed diffuse irradiance in logarithmic space. Otherwise, returns in linear space

        Returns
        -------
        ch.Tensor
            A tensor of the computed diffuse irradiance for each viewpoint / direction combination,
            in either linear or logarithmic space depending on `log_space`.

        """
        num_viewpoints, num_directions = normals.shape[:2]

        random_directions = F.normalize(ch.randn(num_viewpoints, num_dir_samples, 3, device=normals.device), p=2,
                                        dim=-1)
        all_directions = random_directions.reshape(-1, 3)
        cos_theta = (normals[:, None, :] * random_directions[:, :, None]).sum(-1)

        bs = all_directions.shape[0]
        batch['size'] = bs
        batch['origin'] = origins[:, None].expand(-1, num_dir_samples, -1).reshape(-1, 3)
        batch['direction'] = all_directions
        batch['env_id'] = env_ids[:, None].expand(-1, num_dir_samples).reshape(-1)
        LoaderWrapper.precompute_useful_batch_quantities(batch)

        log_rgb, depth = self.render(batch, num_depth_samples)

        rgb = ch.exp(log_rgb)
        factors = cos_theta.relu() / ch.pi
        counts = (cos_theta >= 0).float().sum(1)
        rgb = rgb.reshape(num_viewpoints, num_dir_samples, 3)
        current_diffuse = (rgb[:, :, None] * factors[:, :, :, None]).sum(1) / counts[:, :, None]
        if log_space:
            return ch.log(current_diffuse)
        else:
            return current_diffuse


def compute_nerf_weights(t: ch.Tensor, densities: ch.Tensor) -> ch.Tensor:
    """
    Computes NeRF rendering weights based on distance t and density along rays.

    Parameters
    ----------
    t : torch.Tensor
        The distances at which densities are sampled, shaped (B, N).
    densities : torch.Tensor
        The density coefficients at each sampled point, shaped (B, N).

    Returns
    -------
    torch.Tensor
        The computed weights for NeRF rendering, shaped (B, N).
    """
    max_dist = t[:, -1:]
    t = ch.cat([t, max_dist], dim=1)
    deltas = t.diff(dim=1)
    prods = densities * deltas
    transmittance = ch.exp(-prods.cumsum(-1))
    shifted_transmittance = ch.nn.functional.pad(
        transmittance[:, :-1],
        (1, 0, 0, 0), mode="constant", value=1.0)
    individual_factors = (1 - ch.exp(-prods))
    result = shifted_transmittance * individual_factors
    result[:, -1] += 1 - result[:].sum(dim=-1)
    return result.clip(0, 1)
