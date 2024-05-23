import torch as ch

from .background import NerfBackground
from .embeddings import RandomFourierEmbedding
from .envnet import EnvNet
from .foreground import PBRNeRF
from ..config import MirrorSDFConfig
from ..material_models.nerf_material import NeRFMaterial
from ..material_models import PBRGGXMIS


class MirrorSDFModel(ch.nn.Module):

    def __init__(self, background: NerfBackground, envnet: EnvNet, foreground: 'PBRNeRF'):
        super().__init__()

        self.background = background
        self.envnet = envnet
        self.foreground = foreground

    @staticmethod
    def from_config(config: MirrorSDFConfig):
        bg_config = config.architecture.background
        fg_config = config.architecture.foreground
        envnet_config = config.architecture.envnet
        foreground_config = config.architecture.foreground

        bg_coord_embedder = RandomFourierEmbedding(4,
                                                   bg_config.coord_encoding_dim,
                                                   bg_config.min_frequency,
                                                   bg_config.max_frequency)
        background = NerfBackground(config.dataset.num_environments,
                                    bg_config.environ_encoding_dim,
                                    bg_coord_embedder,
                                    config.dataset.maximum_distance_to_env / config.dataset.object_bounding_box.max(),
                                    train_density_noise=bg_config.train_density_noise,
                                    n_layers=bg_config.num_layers,
                                    width=bg_config.width)

        fg_coord_embedder = RandomFourierEmbedding(3,
                                                   foreground_config.coord_encoding_dim,
                                                   foreground_config.min_frequency,
                                                   foreground_config.max_frequency)

        env_coord_embedder = RandomFourierEmbedding(3,
                                                    envnet_config.coord_encoding_dim,
                                                    envnet_config.min_frequency,
                                                    envnet_config.max_frequency,
                                                    is_trainable=envnet_config.trainable_fourrier_bases)

        env_direction_embedder = RandomFourierEmbedding(3,
                                                        envnet_config.direction_encoding_dim,
                                                        envnet_config.min_frequency,
                                                        envnet_config.max_frequency,
                                                        is_trainable=envnet_config.trainable_fourrier_bases)

        envnet = EnvNet(config.dataset.num_environments,
                        envnet_config.environ_encoding_dim,
                        env_coord_embedder,
                        env_direction_embedder,
                        n_layers_specular=envnet_config.num_layers_specular,
                        n_layers_diffuse=envnet_config.num_layers_diffuse,
                        width_specular=envnet_config.width_specular,
                        width_diffuse=envnet_config.width_diffuse,
                        num_light_sources=envnet_config.num_point_lights)

        # material = PBRGGXFresnelSmithMaterial(envnet,
        #                                       train_samples=config.rendering_train.num_bsdf_samples,
        #                                       eval_samples=config.rendering_eval.num_bsdf_samples)
        material = NeRFMaterial(256, 32, 32,
                                16, config.dataset.num_environments,
                                clip_outputs=(
                                    config.dataset.minimum_measurement,
                                    config.dataset.maximum_measurement
                                ),
                                n_layers=4,
                                )

        pbr_material = PBRGGXMIS(envnet, train_samples=32, eval_samples=64, num_light_sources=16,
                                 light_phong_power=20_000, beta_mis=1)

        foreground_model = PBRNeRF(fg_coord_embedder,
                                   material_model=material,
                                   pbr_material_model=pbr_material,
                                   train_density_noise=fg_config.train_density_noise,
                                   n_layers=fg_config.num_layers,
                                   coarse_width=fg_config.coarse_width,
                                   coarse_n_layers=fg_config.coarse_num_layers,
                                   width=fg_config.width)

        return MirrorSDFModel(background, envnet, foreground_model)
