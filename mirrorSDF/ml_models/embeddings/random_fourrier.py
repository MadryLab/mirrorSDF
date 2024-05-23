import numpy as np
import torch as ch
import torch.nn.functional as F


class RandomFourierEmbedding(ch.nn.Module):
    """
      A module for generating Random Fourier Embeddings

      This module implements a random Fourier embedding layer that transforms input
      features into a higher-dimensional space using a combination of random projections
      and Fourier transformations. The transformations can be either linear
      or logarithmic in frequency space, and the module supports trainable phase offsets
      and frequency magnitudes with optional gradient limiting.

      Parameters
      ----------
      input_features : int
          The number of input features.
      num_bases : int
          The number of bases (output dimension) of the embedding.
      min_feq : float
          The minimum frequency of the random Fourier features.
      max_freq : float
          The maximum frequency of the random Fourier features.
      transition_slope : float, optional
          allows starting training with low frequency and gradually adding the higher frequencies over time. The
          bigger the slope the less smooth is the transition. values <= 0 disable the feature
      is_linear : bool, optional
          Whether to use a linear scale for frequencies. If False, a logarithmic scale is used. Default is False.
      is_trainable : bool, optional
          Whether the biases and weights of the embedding layer are trainable. Default is False.
      limit_gradient : bool, optional
          If True, limits the gradient magnitude by scaling the output. Default is False. Might be useful for models
          that require bounded gradients.
      """

    def __init__(self, input_features: int, num_bases: int, min_feq: float, max_freq: float,
                 transition_slope: float = 0.0, is_linear=False, is_trainable=False, limit_gradient=False):
        super().__init__()

        self.is_trainable = is_trainable
        self.limit_gradient = limit_gradient
        self.register_buffer('progress', ch.zeros(1, dtype=ch.float32))
        self.transition_slope = transition_slope

        # We leverage the efficient implementation linear layers
        self.embedding_layer = ch.nn.Linear(input_features, num_bases)

        # Random directions and scale
        if is_linear:
            lengths = ch.linspace(min_feq, max_freq, num_bases)
        else:
            lengths = 2 ** (ch.linspace(np.log2(min_feq), np.log2(max_freq), num_bases))

        self.register_buffer('lengths', lengths)

        self.embedding_layer.weight.data[:] = F.normalize(ch.randn(num_bases, input_features), p=2, dim=1)
        self.embedding_layer.weight.data *= lengths[:, None]

        # Unlike other papers we add a random phase offset,
        # this is better as we don't need to explicitly have sin and cos functions separately
        self.embedding_layer.bias.data = ch.rand_like(self.embedding_layer.bias) * 2 * ch.pi

        self.embedding_layer.bias.requires_grad_(is_trainable)
        self.embedding_layer.weight.requires_grad_(is_trainable)

        self.input_features = input_features
        self.output_features = num_bases

    def set_training_progress(self, progress: float):
        """
        Sets the current training progress, adjusting the behavior of the embedding based on the training stage.

        Parameters
        ----------
        progress : float
            The current training progress as a float between 0 and 1.
        """
        progress = np.clip(progress, 0, 1)
        self.progress.fill_(progress)

    def forward(self, x: ch.Tensor, disable_limit_gradient: bool = False) -> ch.Tensor:
        """
        Applies the random Fourier embedding transformation to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        disable_limit_gradient : bool, optional
            If True, disables the gradient limiting feature for this forward pass, even if it
            was enabled in the constructor. Default is False.

        Returns
        -------
        torch.Tensor
            The transformed tensor after applying the random Fourier embedding.
        """
        result = ch.sin(self.embedding_layer(x))

        if self.limit_gradient and not disable_limit_gradient:
            result = result / (self.lengths[None])

        if self.transition_slope <= 0:
            return result
        n_bases = result.shape[-1]

        individual_progress = ch.linspace(0, -1, n_bases, device=x.device) + self.progress
        factor = ch.sigmoid(
            individual_progress * n_bases / self.transition_slope
        )

        return result * factor[None]
