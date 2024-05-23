from typing import TYPE_CHECKING

import torch as ch
import torch.nn.functional as F

from .ggx import ggx_distribution
from ...utils.light_simulation import batch_dot

if TYPE_CHECKING:
    from ...ml_models.envnet import EnvNet


class EnvNetSampler:
    """
    A sampler for retrieving light source samples from an EnvNet model.
    """

    def __init__(self, envnet: 'EnvNet', light_source_count: int = 16, r2: float = 1e-5):
        self.light_source_coords = envnet.point_lights_loc[:, :light_source_count]
        self.light_source_count = self.light_source_coords.shape[1]
        self.r2 = r2

    def sample(self, coords: ch.Tensor, env_id: ch.IntTensor) -> ch.Tensor:
        batch_size = env_id.shape[0]
        L_c = self.light_source_coords[env_id]
        directions = F.normalize(L_c - coords[:, None], p=2, dim=-1)

        # We sample according to the ggx distribution
        # This is a little more complex than usual because this is not in shading space
        u = ch.rand(batch_size, self.light_source_count, device=coords.device)
        tan_theta_2 = self.r2 * u / (1 - u)
        sin_theta = ch.sqrt(tan_theta_2 / (1 + tan_theta_2))
        rand_dir = F.normalize(ch.randn(batch_size, self.light_source_count, 3, device=coords.device), p=2, dim=-1)
        projected = F.normalize(rand_dir - directions * batch_dot(directions, rand_dir)[..., None], p=2, dim=-1)
        final_direction = F.normalize(directions + projected * sin_theta[..., None], p=2, dim=-1)

        return final_direction

    def pdf(self, coords, env_id, L):
        L_c = self.light_source_coords[env_id]
        directions = F.normalize(L_c - coords[:, None], p=2, dim=-1)
        dot = batch_dot(L[:, :, None], directions[:, None])
        return ggx_distribution(dot, self.r2)
