from typing import TYPE_CHECKING

import torch as ch
import torch.nn.functional as F

from .base import BaseMaterialModel
from .samplers import GGXSampler
from .samplers.ggx import ggx_distribution
from ..utils.light_simulation import batch_dot
from ..utils.materials import fresnel_schlick, geometry_smith, impute_L, shading_transforms, apply_transform, \
    NdotX

if TYPE_CHECKING:
    from ..ml_models.envnet import EnvNet


class PBRGGXFresnelSmithMaterial(BaseMaterialModel):

    def __init__(self, envnet: 'EnvNet', train_samples: int = 32, eval_samples: int = 64,
                 min_roughness: float = 1e-4, min_angle: float = 1e-3):
        super().__init__()

        self.envnet = envnet
        self.min_angle = min_angle
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.min_roughness = min_roughness

        self.ggx_sampler = GGXSampler()

    @property
    def num_samples(self):
        return self.train_samples if self.training else self.eval_samples

    @property
    def _split_params(self):
        return {
            'albedo': 3,
            'specular_color': 3,
            'roughness': 1,
        }

    def forward(self, x: ch.Tensor, env_ids: ch.IntTensor, normal: ch.Tensor,
                view_dir: ch.Tensor, flat_params: ch.Tensor):
        params = self.check_extract_params(x, env_ids, normal, view_dir, flat_params)
        params['roughness'] = F.sigmoid(params['roughness'])
        params['albedo'] = F.softplus(params['albedo'], beta=20)
        params['specular_color'] = F.softplus(params['specular_color'], beta=20)
        roughness_squared = params['roughness'].clip(self.min_roughness).ravel().square()

        L_world, valid_samples, probs, specular_brdf = self.sample_specular_reflections(roughness_squared, normal,
                                                                                        -view_dir,
                                                                                        F0=params['specular_color'])

        Li_specular, Li_diffuse = self.envnet.forward(x, env_ids, L_world, normal[:, None])
        Li_diffuse = ch.exp(Li_diffuse[:, 0])
        Li_specular = ch.exp(Li_specular)


        L_o_diffuse = Li_diffuse * params['albedo']
        L_o_specular = (Li_specular * specular_brdf)

        num_valid_samples = valid_samples.sum(-1)
        denominator = probs * num_valid_samples[:, None]

        # Avoid division by zero and invalid samples
        denominator = ch.where(valid_samples & (denominator > 0.0), denominator, 1.0)

        # Make sure we don't take any contribution of bad samples
        L_o_specular = ch.where(valid_samples[..., None], L_o_specular, 0.0)

        L_o_specular = L_o_specular / denominator[..., None].detach()
        L_o_specular = L_o_specular.sum(-2)


        return L_o_specular + L_o_diffuse

    def sample_specular_reflections(self, roughness_squared: ch.Tensor, normal: ch.Tensor, view_dir: ch.Tensor,
                                    F0: ch.Tensor):
        # From now everything will be done in a shading reference Frame
        shading2world, world2shading = shading_transforms(normal)
        V = apply_transform(world2shading, view_dir)[:, None]

        H = self.ggx_sampler.sample(roughness_squared, self.num_samples)
        L = impute_L(V, H)

        NdotL = NdotX(L)
        NdotH = NdotX(H)
        NdotV = NdotX(V)
        VdotH = batch_dot(V, H)

        valid_view_point = NdotV >= self.min_angle
        valid_reflection = NdotL >= self.min_angle

        valid_samples = valid_view_point & valid_reflection
        NdotV = ch.where(valid_samples, NdotV, 0.5)
        NdotL = ch.where(valid_samples, NdotL, 0.5)

        specular_brdf = self.specular_brdf(F0[:, None], roughness_squared[:, None], NdotL, NdotV, NdotH, VdotH)
        probs = self.ggx_sampler.pdf(roughness_squared[:, None], NdotH, VdotH)

        # Go back in world coordinates
        L_world = apply_transform(shading2world[:, None], L)

        # specular_brdf = ch.where(valid_samples[..., None], specular_brdf, 0)
        L_world = ch.where(valid_samples[..., None], L_world, 0)

        return L_world, valid_samples, probs, specular_brdf

    @staticmethod
    def specular_brdf(F0: ch.Tensor, roughness_squared: ch.Tensor, NdotL: ch.Tensor, NdotV: ch.Tensor,
                      NdotH: ch.Tensor, VdotH: ch.Tensor):
        F = fresnel_schlick(VdotH[..., None], F0)
        G = geometry_smith(NdotV, NdotL, roughness_squared)
        D = ggx_distribution(NdotH, roughness_squared)
        return (D * G / (4 * NdotV))[..., None] * F
