from typing import TYPE_CHECKING

import torch as ch

from .base import BaseMaterialModel
from ..utils.materials import impute_L

if TYPE_CHECKING:
    from ..ml_models.envnet import EnvNet


class MirrorMaterial(BaseMaterialModel):
    def __init__(self, envnet: 'EnvNet'):
        super().__init__()
        self.envnet = envnet

    @property
    def _split_params(self):
        return {
            'reflectance': 3
        }

    def forward(self, x: ch.Tensor, env_ids: ch.IntTensor, normal: ch.Tensor, view_dir: ch.Tensor,
                params: ch.Tensor):
        extracted_params = self.check_extract_params(x, env_ids, normal, view_dir, params)
        direction = impute_L(view_dir, normal)
        Li, _ = self.envnet(x, env_ids, direction[:, None], None)
        Li = ch.exp(Li)
        return extracted_params['reflectance'] * Li[:, 0]
