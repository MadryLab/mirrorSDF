from abc import abstractmethod
from typing import Dict

import torch as ch

from ..dataset.batch import Batch


class BaseMaterialModel(ch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def num_parameters(self) -> int:
        return sum(self._split_params.values())

    @property
    @abstractmethod
    def _split_params(self) -> Dict[str, int]:
        raise NotImplementedError()

    def check_extract_params(self, x: ch.Tensor, env_ids: ch.IntTensor, normal: ch.Tensor, view_dir: ch.Tensor,
                             params: ch.Tensor) -> Dict[str, ch.Tensor]:
        batch_size = x.shape[0]
        if x.shape[1] != 3:
            raise ValueError("Locations have to be 3D")
        if env_ids.shape != (batch_size,):
            raise ValueError("Env id should be a tensor of indices")
        if normal.shape != x.shape:
            raise ValueError("normals and locations need to have the same shape")
        if view_dir.shape != x.shape:
            raise ValueError("view directions and locations need to have the same shape")
        if len(params) == 1:
            params = params[None]
        if params.shape[0] == 1:
            params = params.expand(batch_size, -1)
        if params.shape[1] < self.num_parameters:
            raise ValueError("Not enough parameters expected:"
                             f"{self.num_parameters}, actual: {params.shape[1]}")

        split_params = ch.split(params[:, :self.num_parameters], list(self._split_params.values()), dim=1)
        return dict(zip(self._split_params.keys(), split_params))

    @abstractmethod
    def forward(self, x: ch.Tensor, env_ids: ch.IntTensor, normal: ch.Tensor, view_dir: ch.Tensor,
                params: ch.Tensor, batch: 'Batch'):
        raise NotImplementedError()
