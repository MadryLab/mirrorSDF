from typing import Tuple

import torch as ch

from ...utils.geometry import spherical_to_cartesian


class EnvMapSampler:
    """
    A sampler for environment maps represented in equirectangular format for physically based rendering.

    This class precomputes a distribution over the environment map that takes into account the solid angle
    each pixel represents. It allows for sampling directions from the environment map and computing the
    probability density function (PDF) and incoming radiance (Li) for given directions.

    Parameters
    ----------
    env_map : torch.Tensor
        The environment map in equirectangular format. Expected to have shape `[height, width, channels]`,
        where `channels` (usually 3) represents color information (e.g., RGB).

    Attributes
    ----------
    env_map : torch.Tensor
        The input environment map stored for sampling.
    normalized_weights : torch.Tensor
        The precomputed weights for each pixel in the environment map, normalized to sum to 1.
    probs : torch.Tensor
        The probabilities for sampling each pixel, adjusted for the solid angle each pixel represents.
    """

    def __init__(self, env_map):
        self.env_map = env_map

        full_weights = env_map.detach().sum(-1)
        # Compute the angles of the poles of the envmap
        a_thetas = ch.linspace(0, ch.pi, env_map.shape[0] + 2, device=full_weights.device)[1:-1, None]
        # self.env_map = self.env_map * ch.sin(a_thetas)[:, None]
        # scale pixels by the actual area they represent
        normalized_weights = full_weights / full_weights.sum()
        # Convert the solid angles
        normalized_weights *= env_map.shape[0] * env_map.shape[1] / (4 * ch.pi)

        self.normalized_weights = normalized_weights.ravel()
        self.probs = (normalized_weights / ch.sin(a_thetas)).ravel()

    def sample(self, batch_size: int, num_samples: int) -> Tuple[ch.Tensor, ch.Tensor]:
        """
        Samples directions from the environment map.

        Parameters
        ----------
        batch_size : int
            The number of separate instances for which to generate samples.
        num_samples : int
            The number of samples to generate per instance.
        device : torch.device
            The device on which the samples should be stored.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - The samples in spherical coordinates (theta, phi).
            - The samples in Cartesian coordinates (x, y, z).
        """
        return env_map_sample_kernel(ch.tensor(batch_size), num_samples, self.normalized_weights, self.env_map)

    def pdf_and_Li(self, L: ch.Tensor, Ls: ch.Tensor, *args, **kwargs) -> Tuple[ch.Tensor, ch.Tensor]:
        """
        Computes the PDF and incoming radiance (Li) for a set of directions.

        Parameters
        ----------
        L : torch.Tensor
            The light directions in Cartesian coordinates.
        Ls : torch.Tensor
            The light directions in spherical coordinates (theta, phi).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - The PDF values for each direction.
            - The incoming radiance (Li) for each direction.
        """
        return pdf_and_Li_kernel(self.env_map, self.probs, L, Ls)


@ch.jit.script
def env_map_sample_kernel(batch_size, num_samples, normalized_weights, env_map):
    ch.manual_seed(42)
    selected_indices = ch.multinomial(normalized_weights, num_samples)

    Ls = ch.stack([
        (selected_indices // env_map.shape[1] + 0.5) / (env_map.shape[0]),
        (selected_indices % env_map.shape[1] + 0.5) / (env_map.shape[1])
    ], dim=-1)

    Ls *= ch.pi
    Ls[..., 1] *= 2
    Ls[..., 1] -= ch.pi

    L = spherical_to_cartesian(Ls)

    Ls = Ls[None].expand(batch_size, num_samples, 2)
    L = L[None].expand(batch_size, num_samples, 3)

    return Ls, L


@ch.jit.script
def pdf_and_Li_kernel(env_map, probs, L, Ls):
    batch_size, num_samples = L.shape[:2]
    theta, phi = ch.unbind(Ls, dim=-1)
    theta = theta / ch.pi
    phi = (phi + ch.pi) / (ch.pi * 2)
    y_coord = ch.round(theta * env_map.shape[0] - 0.5).long().clip(0, env_map.shape[0] - 1)
    x_coord = ch.round(phi * env_map.shape[1] - 0.5).long().clip(0, env_map.shape[1] - 1)
    ix_coords = (x_coord + y_coord * env_map.shape[1]).ravel()
    probs = ch.gather(probs, 0, ix_coords).reshape(batch_size, num_samples)
    Li = ch.gather(env_map.reshape(-1, 3), 0, ix_coords[:, None].expand(-1, 3)).reshape(batch_size, num_samples, 3)

    return probs, Li
