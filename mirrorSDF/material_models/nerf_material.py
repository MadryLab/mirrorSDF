from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch as ch

from .base import BaseMaterialModel
from ..ml_models.architectures.mlp_skip_connection import MLPWithSkip
from ..ml_models.embeddings import RandomFourierEmbedding
from ..utils.materials import impute_L

if TYPE_CHECKING:
    from ..dataset.batch import Batch


class NeRFMaterial(BaseMaterialModel):

    def __init__(self, num_features: int, num_coord_embedding: int, num_dir_embedding: int, num_env_embedding: int,
                 num_envs: int, clip_outputs: Tuple[float, float] = (1e-6, 1e6),
                 **base_model_args):
        super().__init__()

        self.num_features = num_features
        self.direction_embedder = RandomFourierEmbedding(3, num_dir_embedding, 1, 250)
        self.coord_embedder = RandomFourierEmbedding(3, num_coord_embedding, 1, 1000)
        self.env_embedder = ch.nn.Embedding(num_envs, num_env_embedding)
        self.num_inputs = 3 * 3 + num_coord_embedding + 2 * num_dir_embedding + num_features + num_env_embedding
        self.inner = MLPWithSkip(self.num_inputs, 3, 128, **base_model_args)
        self.clip_outputs_log = (
            np.log(clip_outputs[0]),
            np.log(clip_outputs[1])
        )

    @property
    def _split_params(self):
        return {
            'features': self.num_features,
        }

    def forward(self, x: ch.Tensor, env_ids: ch.IntTensor, normal: ch.Tensor,
                view_dir: ch.Tensor, flat_params: ch.Tensor, batch: 'Batch'):
        light_dir = impute_L(-view_dir, normal)
        light_dir_encoded = self.direction_embedder(light_dir)
        normal_encoded = self.direction_embedder(normal)
        coord_encoded = self.coord_embedder(x)
        env_encoded = self.env_embedder(env_ids)
        all_inputs = ch.cat([x,
                             normal,
                             light_dir,
                             coord_encoded,
                             normal_encoded,
                             light_dir_encoded,
                             env_encoded,
                             flat_params], dim=-1)
        shifted = self.inner(all_inputs) - 5
        result = ch.exp(shifted.clip(self.clip_outputs_log[0], self.clip_outputs_log[1]))
        return result
