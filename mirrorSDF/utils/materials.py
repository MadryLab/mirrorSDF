from typing import Tuple

import torch as ch
import torch.nn.functional as F

from .light_simulation import batch_dot


def impute_L(V, H):
    dot_product = batch_dot(H, V)[..., None]
    return ch.nn.functional.normalize(2 * dot_product * H - V, dim=-1)


def impute_H(V, L):
    return ch.nn.functional.normalize(V + L, dim=-1)


def fresnel_schlick(cos_theta, F0):
    return F0 + (1 - F0) * (1 - cos_theta) ** 5


def geometry_schlick_ggx(NdotX, k):
    return NdotX / (NdotX * (1 - k) + k)


def geometry_smith(NdotV, NdotL, roughness_squared):
    k = roughness_squared / 2
    return geometry_schlick_ggx(NdotV, k) * geometry_schlick_ggx(NdotL, k)


def shading_transforms(normals: ch.Tensor) -> Tuple[ch.Tensor, ch.Tensor]:
    u_1 = ch.zeros_like(normals)
    best_second_dir = normals.abs().argmin(-1, keepdim=True)
    u_1.scatter_(-1, best_second_dir, 1)
    u_1 = u_1 - normals * batch_dot(u_1, normals)[..., None]
    u_1 = F.normalize(u_1, p=2, dim=-1)
    u_2 = ch.cross(u_1, normals, dim=-1)
    shading_to_world = ch.stack([u_1, u_2, normals], dim=-1)
    world_to_shading = shading_to_world.transpose(-1, -2)
    return shading_to_world, world_to_shading


def apply_transform(transform: ch.Tensor, x: ch.Tensor):
    return batch_dot(transform, x.unsqueeze(-2))


def NdotX(X: ch.Tensor):
    return X[..., -1]
