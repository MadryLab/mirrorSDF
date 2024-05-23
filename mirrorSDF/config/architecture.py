from dataclasses import dataclass, field

import numpy as np


@dataclass
class BackgroundConfig:
    """
    An object describing the architecture used to learn the environment

    Parameters
    ----------
    num_layers : int, default=5
        The number of layers in the background model.
    environ_encoding_dim : int, default=32
        The dimensionality of the environment encoding.
    coord_encoding_dim : int, default=256
        The dimensionality of the coordinate encoding.
    min_frequency : float, default=1
        The minimum frequency for the background model's operations.
    max_frequency : float, default=1000
        The maximum frequency for the background model's operations.
    width : int, default=128
        The width of the background model.
    train_density_noise: float, default=0
        Adds noise to the density (Before the final activation) produced by the background NeRF

    """
    num_layers: int = 5
    environ_encoding_dim: int = 32
    coord_encoding_dim: int = 256
    min_frequency: float = 1
    max_frequency: float = 1000
    width: int = 128
    train_density_noise: float = 0.0


@dataclass
class ForegroundConfig:
    """
    An object describing the architecture used to learn the object under test

    Parameters
    ----------
    num_layers : int, default=5
        The number of layers in the background model.
    coord_encoding_dim : int, default=256
        The dimensionality of the coordinate encoding.
    min_frequency : float, default=1
        The minimum frequency for the background model's operations.
    max_frequency : float, default=1000
        The maximum frequency for the background model's operations.
    width : int, default=128
        The width of the background model.
    train_density_noise: float, default=0
        Adds noise to the density (Before the final activation) produced by the background NeRF

    """
    num_layers: int = 9
    coarse_num_layers: int = 4
    coarse_width: int = 128
    coord_encoding_dim: int = 256
    min_frequency: float = 1
    max_frequency: float = 512
    width: int = 256
    train_density_noise: float = 0.01


@dataclass
class EnvNetConfig:
    """
    Configuration class for EnvNet specifying the architecture and parameters
    for the neural network and embeddings.

    Attributes
    ----------
    num_layers_specular : int, optional
        The number of layers in the specular MLP, by default 5.
    num_layers_diffuse : int, optional
        The number of layers in the diffuse MLP, by default 2.
    environ_encoding_dim : int, optional
        The dimensionality of the environmental encoding, by default 16.
    coord_encoding_dim : int, optional
        The dimensionality of the coordinate encoding, by default 32.
    direction_encoding_dim : int, optional
        The dimensionality of the direction encoding, by default 128.
    min_frequency : float, optional
        The minimum frequency for the Fourier features, by default 1.
    max_frequency : float, optional
        The maximum frequency for the Fourier features, by default 512.
    width_specular : int, optional
        The width (number of neurons per layer) of the specular MLP, by default 128.
    width_diffuse : int, optional
        The width (number of neurons per layer) of the diffuse MLP, by default 32.
    num_point_lights : int, optional
        The number of point light sources to be considered, by default 128.
    trainable_fourrier_bases : bool, optional
        Whether the Fourier bases are trainable, by default False.
    """
    num_layers_specular: int = 5
    num_layers_diffuse: int = 2
    environ_encoding_dim: int = 16
    coord_encoding_dim: int = 32
    direction_encoding_dim: int = 128
    min_frequency: float = 1
    max_frequency: float = 512
    width_specular: int = 128
    width_diffuse: int = 32
    num_point_lights: int = 384
    trainable_fourrier_bases: bool = False


@dataclass
class SDFModelConfig:
    """
    Configuration class for the architecture of the MLP representing an SDF (Signed distance function)

    Attributes
    ----------
    num_layers : int
        Number of layers in the neural network. Default is 6.
    coord_encoding_dim : int, default=256
        The dimensionality of the coordinate encoding.
    min_frequency : float
        Minimum frequency for positional encoding. Default is 1.
    max_frequency : float
        Maximum frequency for positional encoding. Default is 1000.
    width : int
        Width of each layer in the neural network. Default is 128.
    skip_connections : np.ndarray
        Indices of layers after which skip connections are added. Default is np.array([3])
    """
    num_layers: int = 6
    coord_encoding_dim: int = 256
    min_frequency: float = 1
    max_frequency: float = 1000
    width: int = 128
    skip_connections: np.ndarray = field(default_factory=lambda: np.array([3]))


@dataclass
class ArchitectureConfig:
    """
    A configuration class containing sub-config for each component of the model

    Attributes
    ----------
    background : BackgroundConfig, default=BackgroundConfig()
        An instance of BackgroundConfig class to hold background layer configurations.

    """
    background: BackgroundConfig = field(default_factory=lambda: BackgroundConfig())
    envnet: EnvNetConfig = field(default_factory=lambda: EnvNetConfig())
    sdf_model: SDFModelConfig = field(default_factory=lambda: SDFModelConfig())
    foreground: ForegroundConfig = field(default_factory=lambda: ForegroundConfig())
