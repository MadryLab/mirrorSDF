from typing import Tuple, Callable

import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def geometric_init_middle_layer(layer: ch.nn.Linear, input_features: int, output_features: int) -> None:
    """
    Initializes the weights and biases of a middle layer of a neural network using the geometric init
    To obtain a sphere

    Parameters
    ----------
    layer : nn.Module
        The neural network layer to initialize. This layer should have 'weight' and 'bias' attributes.
    input_features : int
        The number of input features to the layer. This influences the initialization of part of the weights.
    output_features : int
        The number of output features from the layer. This is used to calculate the standard deviation for
        weight initialization.    
    """
    ch.nn.init.normal_(layer.weight[:, input_features:], 0, 0)
    ch.nn.init.constant_(layer.bias, 0)

    std = (2 / output_features) ** 0.5
    ch.nn.init.normal_(layer.weight[:, :input_features], 0, std)


def geometric_init_last_layer(layer: ch.nn.Linear, input_features: int):
    """
    Initializes the weights and biases of the last layer  of a neural network using the geometric init
    To obtain a sphere

    Parameters
    ----------
    layer : nn.Module
        The neural network layer to initialize. This layer should have 'weight' and 'bias' attributes.
    input_features : int
        The number of input features to the layer. This influences the initialization of part of the weights.
    """

    ch.nn.init.normal_(layer.weight[:1], (np.pi / input_features) ** 0.5, 1e-5)
    ch.nn.init.constant_(layer.bias[:1], 0)  # We init this later


class MLPSDF(ch.nn.Module):
    """
    A Multi-Layer Perceptron (MLP) designed for Signed Distance Function (SDF) prediction,
    with support for encoding features, skip connections, weight normalization, and
    geometric initialization.

    Parameters
    ----------
    encoding_features : int
        The number of encoding features to be concatenated with the input coordinates.
    n_layers : int
        The number of layers in the MLP.
    width : int
        The width (number of neurons) in each hidden layer of the MLP.
    extra_output_features : int, optional
        The number of additional output features beyond the SDF value. Defaults to: 0
    skip_connections : Tuple[int, ...], optional
        Indices of layers after which a skip connection is added at each layer indicated in this list.
        Defaults to: (4,)
    use_weight_norm : bool, optional
        If True, applies weight normalization to the layers.
    geometric_init : bool, optional
        If True, initializes the weights of the network using the geometric init to obtain a sphere.
        Defaults to: True
    initial_sphere_radius : float, optional
        The initial radius of the sphere for the geometric initialization. Defaults to: 0.25
    max_sphere_radius : float, optional
        Controls the maximum size of the shape learned by this network.
        Defaults to: 1.0

    Attributes
    ----------
    layers : ch.nn.ModuleList
        The list of layers in the MLP, excluding the final output layer.
    last_layer : ch.nn.Linear
        The final output layer of the MLP.
    original_encoding_importance: float
        The amount of importance to give to the coordinate embedding coefficients at init time
    activation: Callable[[ch.Tensor], ch.Tensor] = ch.nn.functional.relu):
        The activation to use in the middle layers, Defaults to ReLU
    """

    def __init__(self, encoding_features: int,
                 n_layers: int, width: int, extra_output_features: int = 0,
                 skip_connections: Tuple[int, ...] = (4, 8, 12),
                 use_weight_norm: bool = True, geometric_init: bool = True,
                 initial_sphere_radius=0.25,
                 max_sphere_radius=1.0,
                 original_encoding_importance: float = 2e-3,
                 activation: Callable[[ch.Tensor], ch.Tensor] = ch.nn.functional.relu):

        super().__init__()

        self.register_buffer('encoding_features', ch.tensor(encoding_features))
        self.register_buffer('initial_sphere_radius', ch.tensor(initial_sphere_radius))
        self.register_buffer('max_sphere_radius', ch.tensor(max_sphere_radius))

        self.activation = activation

        layers = []

        total_input_features = 3 + encoding_features
        previous_output_features = total_input_features

        for i in range(n_layers):
            current_input_features = previous_output_features
            current_output_features = width

            if i != 0 and i != (n_layers - 1) and i in skip_connections:
                current_input_features += total_input_features

            layer = ch.nn.Linear(current_input_features, current_output_features,
                                 bias=True)

            if geometric_init:
                # If this is the first layer we want to give no weight to any other feature than the xyz coords
                if i == 0:
                    geometric_init_middle_layer(layer, 3, current_output_features)
                    ch.nn.init.normal_(layer.weight[:, 3:], 0, original_encoding_importance / encoding_features ** 0.5)
                # We also don't want to give weights to the skip connection by default
                else:
                    geometric_init_middle_layer(layer, previous_output_features, current_output_features)

            if use_weight_norm:
                layer = weight_norm(layer)

            layers.append(layer)
            previous_output_features = current_output_features

        last_layer = ch.nn.Linear(previous_output_features, 1 + extra_output_features)
        geometric_init_last_layer(last_layer, previous_output_features)
        self.last_layer = last_layer

        self.layers = ch.nn.ModuleList(layers)
        self.post_init()

    def post_init(self):
        """Adjusts the weights and biases of the final layer based on a heuristic to
        approximate the desired initial sphere radius. Also tries to get the average gradient
        as close to 1 as possible (which should be the case for an SDF
        """
        batch_size = 128
        points = F.normalize(ch.randn(batch_size, 3), dim=1) * self.initial_sphere_radius
        points[0, :] = 0

        null_encoding = ch.zeros((batch_size, self.encoding_features))

        measured_sdf = self(points, null_encoding)[0]

        avg_around_circle = measured_sdf[1:].mean()

        at_zero = measured_sdf[0]
        measured_slope = (avg_around_circle.mean() - at_zero) / self.initial_sphere_radius
        self.last_layer.weight.data[0] /= measured_slope
        self.last_layer.bias.data[0] += -at_zero.item() / measured_slope.item() - self.initial_sphere_radius

    def forward(self, coordinates, encodings=None):
        """
        Forward pass through the MLP-SDF model.

        Parameters
        ----------
        coordinates : ch.Tensor
            The input coordinates tensor. Expected shape: (batch_size, 3).
        encodings : ch.Tensor, optional
            The encoding tensor to be concatenated with coordinates. Expected shape: (batch_size, encoding_features).

        Returns
        -------
        Tuple[ch.Tensor, ch.Tensor]
            A tuple containing the predicted SDF values and any extra output features.

        Raises
        ------
        AssertionError
            If the shape of `coordinates` or `encodings` does not match the expected dimensions.
        """
        assert coordinates.shape[1] == 3

        x_0 = coordinates

        if self.encoding_features > 0:
            assert encodings.shape[1] == self.encoding_features
            x_0 = ch.cat([x_0, encodings], dim=1)

        x = x_0

        for i, layer in enumerate(self.layers):
            # Skip connection
            if layer.in_features != x.shape[1]:
                x = ch.cat([x, x_0], dim=1)

            x = layer(x)
            if self.activation == F.softplus:
                x = ch.nn.functional.softplus(x, beta=500)
            else:
                x = self.activation(x)

        x = self.last_layer(x)

        extra_output_features = self.last_layer.out_features - 1
        sdf, features = ch.split(x, [1, extra_output_features], dim=1)

        dist_to_origin = coordinates.norm(p=2, dim=-1)[:, None]

        sdf = ch.minimum(sdf, dist_to_origin + self.max_sphere_radius)
        sdf = ch.maximum(sdf, -dist_to_origin - self.max_sphere_radius)

        return sdf, features
