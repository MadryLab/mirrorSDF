from typing import Tuple, Optional, TYPE_CHECKING

import torch as ch

from ..architectures import MLPWithSkip
from ..background.light_source_sampler import LightSourceSampler
from ..embeddings import RandomFourierEmbedding

if TYPE_CHECKING:
    from ..background import NerfBackground
    from ...dataset.batch import Batch


class EnvNet(ch.nn.Module):
    """
    A neural network model designed for environmental lighting estimation, which is a distilled version
    of a Neural Radiance Field (NeRF) model. Unlike NeRF, EnvNet directly predicts lighting without the
    need for sampling along rays. It uses embeddings for both coordinates and directions, and separate
    MLPs for specular and diffuse reflections, leveraging pre-computed light sources for efficiency.

    Parameters
    ----------
    n_envs : int
        The number of distinct environments the model is expected to handle.
    env_embedding_size : int
        The size of the embedding vector for each environment.
    coord_embedder : RandomFourierEmbedding
        The embedding module for spatial coordinates.
    direction_embedder : RandomFourierEmbedding
        The embedding module for light directions.
    width_specular : int
        The width of the layers in the specular MLP.
    n_layers_specular : int
        The number of layers in the specular MLP.
    width_diffuse : int
        The width of the layers in the diffuse MLP.
    n_layers_diffuse : int
        The number of layers in the diffuse MLP.
    num_light_sources : int
        The number of light sources to pre-compute for each environment.
    **base_model_args : dict
        Additional arguments to pass to the MLPWithSkip constructors.

    Attributes
    ----------
    coord_embedder : RandomFourierEmbedding
        Embedding for spatial coordinates.
    direction_embedder : RandomFourierEmbedding
        Embedding for directions.
    n_envs : int
        Number of environments.
    env_embedding : torch.nn.Embedding
        Embedding layer for environment identification.
    inner_specular : MLPWithSkip
        MLP for specular reflection estimation.
    inner_diffuse : MLPWithSkip
        MLP for diffuse reflection estimation.
    point_lights_loc : torch.Tensor
        Pre-computed locations of point light sources for each environment.
    point_lights_weights : torch.Tensor
        Pre-computed weights of point light sources for each environment.
    """

    point_lights_loc: ch.Tensor
    point_lights_weights: ch.Tensor
    point_lights_normalization: ch.Tensor

    def __init__(self,
                 n_envs, env_embedding_size,
                 coord_embedder: RandomFourierEmbedding,
                 direction_embedder: RandomFourierEmbedding,
                 width_specular: int,
                 n_layers_specular: int,
                 width_diffuse: int,
                 n_layers_diffuse: int,
                 num_light_sources: int,
                 **base_model_args):
        super().__init__()

        # - 3 for coordinates
        # - 3 for direction
        # - the features created by the embedders
        n_inputs = 3 + 3 + coord_embedder.output_features + direction_embedder.output_features + env_embedding_size

        self.coord_embedder = coord_embedder
        self.direction_embedder = direction_embedder
        self.n_envs = n_envs

        self.env_embedding = ch.nn.Embedding(n_envs, env_embedding_size)
        self.inner_specular = MLPWithSkip(n_inputs, 3, n_layers=n_layers_specular,
                                          width=width_specular, **base_model_args)
        self.inner_diffuse = MLPWithSkip(n_inputs, 3, n_layers=n_layers_diffuse,
                                         width=width_diffuse, **base_model_args)

        self.register_buffer('point_lights_loc', ch.zeros(n_envs, num_light_sources, 3))
        self.register_buffer('point_lights_weights', ch.zeros(n_envs, num_light_sources))
        self.register_buffer('point_lights_normalization', ch.zeros(n_envs))

    def pre_compute_light_sources(self, model: 'NerfBackground', batch: 'Batch'):
        """
        Pre-computes light source locations and their weights for each environment using a provided NeRF model
        and batch configuration. This process leverages the LightSourceSampler to gather light source information
        which is then stored in the model for faster future computations.

        Parameters
        ----------
        model : NerfBackground
            The NeRF model from which to sample light sources.
        batch : Batch
            The batch configuration containing parameters such as size, background radius range, and device.
        """
        sampler = LightSourceSampler(model, batch)
        for env_id in range(self.n_envs):
            points, weights = sampler.collect_samples(self.point_lights_loc.shape[1], env_id)
            self.point_lights_loc[env_id] = points
            self.point_lights_weights[env_id] = weights / weights.sum()

    @staticmethod
    def concat_and_forward(model: MLPWithSkip, coords: ch.Tensor,
                           directions: ch.Tensor, coords_features: ch.tensor,
                           env_features: ch.Tensor, direction_features: ch.Tensor) -> ch.Tensor:
        """
        A static helper method to concatenate various features and forward them through a specified MLP model.
        It is used to process both specular and diffuse reflection predictions.

        Parameters
        ----------
        model : MLPWithSkip
            The MLP model to forward the features through.
        coords : torch.Tensor
            The spatial coordinates.
        directions : torch.Tensor
            The light or normal directions.
        coords_features : torch.Tensor
            The embedded spatial coordinates.
        env_features : torch.Tensor
            The embedded environment features.
        direction_features : torch.Tensor
            The embedded direction features.

        Returns
        -------
        torch.Tensor
            The reshaped predictions from the model, corresponding to either specular or diffuse reflections.
        """
        batch_size = direction_features.shape[0]
        num_rays = direction_features.shape[1]

        features = ch.cat([
            coords[:, None].expand(-1, num_rays, -1),
            directions,
            coords_features[:, None].expand(-1, num_rays, -1),
            env_features[:, None].expand(-1, num_rays, -1),
            direction_features
        ], dim=-1)

        features_flat = features.reshape(-1, features.shape[-1])
        prediction = model(features_flat)
        return prediction.reshape(batch_size, num_rays, 3)

    def compute_specular_probabilities(self, env_ids: ch.IntTensor, Li: ch.Tensor) -> ch.Tensor:
        batch_size, num_samples, _ = Li.shape
        intensity = Li.sum(-1)
        sampling_probability = intensity / self.point_lights_normalization[env_ids][:, None]
        return 1 / self.point_lights_loc.shape[1]
        return sampling_probability

    def forward(self, coords: ch.Tensor, env_ids: ch.IntTensor, normals_specular: Optional[ch.Tensor] = None,
                normals_diffuse: Optional[ch.Tensor] = None) -> Tuple[ch.Tensor, ch.Tensor]:

        if normals_diffuse is None and normals_specular is None:
            raise ValueError("You need to specify at least some normals to evaluate")

        coords_features = self.coord_embedder(coords)
        env_feature = self.env_embedding(env_ids)

        if normals_specular is not None:
            log_prediction_specular = self.concat_and_forward(self.inner_specular, coords, normals_specular,
                                                              coords_features, env_feature,
                                                              self.direction_embedder(normals_specular))
        else:
            log_prediction_specular = None

        if normals_diffuse is not None:
            log_prediction_diffuse = self.concat_and_forward(self.inner_diffuse, coords, normals_diffuse,
                                                             coords_features, env_feature,
                                                             self.direction_embedder(normals_diffuse))
        else:
            log_prediction_diffuse = None

        return log_prediction_specular, log_prediction_diffuse
