from typing import TYPE_CHECKING

import torch as ch

from .base import BaseMaterialModel

if TYPE_CHECKING:
    from ..dataset.batch import Batch


class NullMaterial(BaseMaterialModel):

    @property
    def _split_params(self):
        return {
            'albedo': 3,
            'specular_color': 3,
            'roughness': 1,
        }

    def forward(self, x: ch.Tensor, env_ids: ch.IntTensor, normal: ch.Tensor, view_dir: ch.Tensor,
                params: ch.Tensor, batch: 'Batch'):
        return ch.zeros_like(x)
