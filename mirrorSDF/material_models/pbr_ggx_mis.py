from typing import TYPE_CHECKING

import torch as ch

from .base import BaseMaterialModel
from .samplers.ggx import ggx_distribution
from ..utils.light_simulation import batch_dot, solve_bounce_location
from ..utils.materials import fresnel_schlick, geometry_smith, impute_L, impute_H

if TYPE_CHECKING:
    from ..ml_models.envnet import EnvNet
    from ..dataset.batch import Batch


class PBRGGXMIS(BaseMaterialModel):

    def __init__(self, envnet: 'EnvNet', train_samples: int = 32, eval_samples: int = 64,
                 num_light_sources: int = 16,
                 min_roughness: float = 1e-4, min_angle: float = 1e-3,
                 beta_mis: float = 2, light_phong_power: float = 11):
        super().__init__()

        self.envnet = envnet
        self.min_angle = min_angle
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.min_roughness = min_roughness
        self.register_buffer('light_sources', envnet.point_lights_loc[:, :num_light_sources])
        self.num_light_sources = self.light_sources.shape[1]
        self.light_phong_power = light_phong_power
        self.beta_mis = beta_mis

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

    def forward(self, coords: ch.Tensor, env_ids: ch.IntTensor, normal: ch.Tensor,
                view_dir: ch.Tensor, flat_params: ch.Tensor, batch: 'Batch'):
        params = self.check_extract_params(coords, env_ids, normal, view_dir, flat_params)
        params['roughness'] = ch.nn.functional.sigmoid(params['roughness'])
        params['albedo'] = ch.nn.functional.softplus(params['albedo'])
        params['specular_color'] = ch.nn.functional.softplus(params['specular_color'])
        roughness2 = params['roughness'].clip(self.min_roughness).square()

        N = normal
        V = -view_dir
        F0 = params['specular_color']

        num_bsdf_samples = self.train_samples if self.training else self.eval_samples

        light_coords = self.light_sources[env_ids]

        # Comppute the center of the ggx distributions
        directions_bsdf = N[:, None].expand(-1, num_bsdf_samples, -1)
        directions_env_direct = normalize(light_coords - coords[:, None])
        coords_expanded = coords[:, None].expand_as(light_coords)
        # TODO figure out why bounce that take into account the thickness of the mirror don't work
        _, bounce_location, valid_bounce = solve_bounce_location(coords_expanded.reshape(-1, 3),
                                                                 light_coords.reshape(-1, 3),
                                                                 mirror_point=batch['mirror_vertices'][0, 0],
                                                                 mirror_normal=batch['mirror_normal'],
                                                                 lut=batch['offset_lut'].float())
        bounce_location_true = bounce_location.reshape(coords.shape[0], self.num_light_sources, 3)
        direction_env_indirect = normalize(bounce_location_true - coords[:, None])
        directions_env = ch.cat([directions_env_direct, direction_env_indirect], dim=1)

        with ch.no_grad():
            bsdf_H = ggx_sampling(directions_bsdf, roughness2)
            bsdf_L = impute_L(V[:, None], bsdf_H)

            # We sample the directions directly for light sources so no half-direction transform
            env_L = sample_phong(directions_env, self.light_phong_power)
            # But we still need the H vector for the rest of calculations
            env_H = impute_H(V[:, None], env_L)

            L = ch.cat([bsdf_L, env_L], dim=1)
            H = ch.cat([bsdf_H, env_H], dim=1)

        # Compute all the necessary dot products
        NdotL = batch_dot(N[:, None], L)
        NdotH = batch_dot(N[:, None], H)
        NdotV = batch_dot(N, V)[:, None]
        VdotH = batch_dot(V[:, None], H)

        # Sample validity
        valid_view_point = NdotV >= self.min_angle
        valid_reflection = NdotL >= self.min_angle
        valid_samples = valid_view_point & valid_reflection
        NdotV = ch.where(valid_samples, NdotV, 0.5)
        NdotL = ch.where(valid_samples, NdotL, 0.5)
        NdotH = ch.where(valid_samples, NdotH, 0.5)

        # Evluate BSDF
        D = ggx_distribution(NdotH, roughness2)
        F = fresnel_schlick(VdotH[..., None], F0[:, None])
        G = geometry_smith(NdotV, NdotL, roughness2)
        specular_bsdf = (D * G / (4 * NdotV))[..., None] * F
        specular_bsdf = ch.where(valid_samples[..., None], specular_bsdf, 0)

        # compute the probabilities now (no gradients needed)
        with ch.no_grad():
            # This represents the pdf of a sample if taken from the bsdf
            pdf_bsdf = D / 4 / VdotH
            # Now we want to compute the probabilities of all samples wrt each light source
            dot_prod_pairs = batch_dot(L[:, :, None], directions_env[:, None, :])
            pdf_env = phong_pdf(dot_prod_pairs, self.light_phong_power)

            # MIS
            n_sample_bsdf = valid_samples[:, :num_bsdf_samples].sum(1)
            n_sample_env = valid_samples[:, num_bsdf_samples:].float()
            w_bsdf = n_sample_bsdf[:, None] * pdf_bsdf
            w_env = n_sample_env[:, None] * pdf_env
            mis_w_denominator = w_bsdf + w_env.sum(-1)
            mis_w_denominator = ch.where(valid_samples & (mis_w_denominator > 0), mis_w_denominator, 1.0)

            # all factors
            indices = ch.arange(directions_env.shape[1])
            mis_total_bsdf = (w_bsdf[:, :num_bsdf_samples] ** (self.beta_mis - 1)
                              / mis_w_denominator[:, :num_bsdf_samples] ** self.beta_mis)
            mis_total_env = (w_env[:, num_bsdf_samples:][:, indices, indices] ** (self.beta_mis - 1)
                             / mis_w_denominator[:, num_bsdf_samples:] ** self.beta_mis)
            weights = ch.cat([mis_total_bsdf, mis_total_env], dim=1)
            weights = ch.where(valid_samples, weights, 0)

        # Evaluate the envnet
        Li_specular, Li_diffuse = self.envnet(coords, env_ids, L, N[:, None])
        Li_specular = ch.exp(Li_specular)
        Li_diffuse = ch.exp(Li_diffuse[:, 0])

        L_o_specular = (Li_specular * specular_bsdf * weights[..., None])
        L_o_specular = ch.where(valid_samples[..., None], L_o_specular, 0)
        L_o_specular = L_o_specular.sum(1)

        L_o_diffuse = Li_diffuse * params['albedo']

        # TODO remove
        self.last = vars()
        for k in self.last.values():
            if ch.is_tensor(k) and k.requires_grad:
                k.retain_grad()

        return L_o_diffuse + L_o_specular


def normalize(x):
    return ch.nn.functional.normalize(x, p=2, dim=-1)


def phong_pdf(NdotH, n):
    result = NdotH ** n * (n + 1) / 2 / ch.pi
    result = ch.where(NdotH > 0, result, 0)
    return result


def sample_phong(directions, phong_power):
    perturbations = normalize(ch.randn_like(directions))
    perturbations = normalize(perturbations - directions * batch_dot(directions, perturbations)[..., None])

    u = ch.rand(directions.shape[0], directions.shape[1], device=directions.device)
    cos_theta2 = u ** (2 / (phong_power + 1))
    sin_theta = ch.sqrt(1 - cos_theta2)

    phong_directions = normalize(directions + sin_theta[..., None] * perturbations)
    return phong_directions


def ggx_sampling(directions, roughness2):
    # sample random direction
    perturbations = normalize(ch.randn_like(directions))
    # project on the relevant plane
    perturbations = normalize(perturbations - directions * batch_dot(directions, perturbations)[..., None])

    u = ch.rand(directions.shape[0], directions.shape[1], device=directions.device)
    costheta = ch.sqrt((1.0 - u) / ((roughness2 - 1) * u + 1))
    sintheta = ch.sqrt(1 - costheta ** 2)
    ggx_directions = normalize(directions + sintheta[..., None] * perturbations)

    return ggx_directions


def ggx_pdf(base_direction, sampled_direction, roughness2, NdotH):
    # We assume NdotH was computed early and do not redo it
    return ggx_distribution(NdotH, roughness2), roughness2
