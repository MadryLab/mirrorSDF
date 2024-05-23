from typing import Optional, Tuple

import torch as ch

from ..architectures.mlp_sdf_with_skip import MLPSDF
from ..embeddings import RandomFourierEmbedding


class SDFModel(ch.nn.Module):

    def __init__(self, base_model: MLPSDF, positional_encoder: Optional[RandomFourierEmbedding] = None):
        super().__init__()
        self.base_model = base_model
        self.positional_encoder = positional_encoder

    def _forward(self, coords: ch.Tensor) -> Tuple[ch.Tensor, ch.Tensor]:
        encodings = None
        if self.positional_encoder is not None:
            encodings = self.positional_encoder(coords)

        sdf, extra_features = self.base_model(coords, encodings)
        return sdf, extra_features

    def forward(self, coords: ch.Tensor,
                with_grad: bool = False):

        if len(coords.shape) != 2:
            o_shape = coords.shape
            na = coords.reshape(-1, o_shape[-1])
            result = self(na, with_grad=with_grad)
            reshaped_result = []
            for r in result:
                try:
                    reshaped_result.append(r.reshape(*o_shape[:-1], r.shape[-1]))
                except AttributeError:
                    reshaped_result.append(r)
            return tuple(reshaped_result)

        if not with_grad:
            sdf, features = self._forward(coords)
            gradients, hessian_diagonal = None, None
        else:
            required_grad = coords.requires_grad
            coords.requires_grad_(True)
            with ch.enable_grad():
                sdf, features = self._forward(coords)
                gradients = ch.autograd.grad(
                    sdf.sum(),
                    coords,
                    create_graph=self.training)[0]

                if self.training:
                    hessian_diagonal = ch.autograd.grad(gradients.sum(), coords, create_graph=True)
                else:
                    hessian_diagonal = None
            coords.requires_grad_(required_grad)

        return sdf, features, gradients, hessian_diagonal
