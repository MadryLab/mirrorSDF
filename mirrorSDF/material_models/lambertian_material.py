from typing import TYPE_CHECKING

import torch as ch

from .base import BaseMaterialModel

if TYPE_CHECKING:
    from ..ml_models.envnet import EnvNet


class LambertianMaterial(BaseMaterialModel):
    def __init__(self, envnet: 'EnvNet'):
        super().__init__()
        self.envnet = envnet

    @property
    def _split_params(self):
        return {
            'albedo': 3
        }

    def forward(self, x: ch.Tensor, env_ids: ch.IntTensor, normal: ch.Tensor, view_dir: ch.Tensor,
                params: ch.Tensor):
        material_params = self.check_extract_params(x, env_ids, normal, view_dir, params)
        _, Li = self.envnet(x, env_ids, None, normal[:, None])
        Li = ch.exp(Li)
        return material_params['albedo'] * Li[:, 0]
