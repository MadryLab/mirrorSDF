from dataclasses import dataclass, field

import numpy as np


@dataclass
class ValidationConfig:
    """
    Configuration settings for validation processes during model evaluation.

    Parameters
    ----------
    render_resolution : int, default=2048
        The resolution at which validation images are rendered
    batch_size : int, default=512
        The number of samples processed in one batch during validation. Affects
        memory usage and computational speed.
    validate_every_iter : int, default=15000
        The frequency, in number of iterations, at which validation is performed.
    visualize_gamma : np.ndarray, default=np.array([0.5, 3])
        Gamma correction values used for visualization purposes during validation.
        Performs Ax^gamma
        - First entry is the exponent (gamma)
        - Second entry is the factor (A)
    """
    render_resolution: int = 2048
    batch_size: int = 512
    validate_every_iter: int = 15_000
    visualize_gamma: np.array = field(default_factory=lambda: np.array([0.5, 3]))


@dataclass
class TrainingConfig:
    """
    Configuration settings for the training process of a machine learning model.

    This class encapsulates various parameters related to the training process,
    including the optimizer used, batch size, learning rate, weight decay, and
    scheduling for learning rate adjustments.

    Parameters
    ----------
    optimizer_name : str, default='AdamW'
        The name of the optimizer to use for training. 'AdamW' is the default.
    batch_size : int, default=2048
        The number of samples processed in one batch of training. Larger batch
        sizes require more memory but can speed up or stabilize the training.
    learning_rate : float, default=1e-3
        The initial learning rate for the optimizer.
    weight_decay : float, default=0
        The weight decay (L2 penalty) that should be applied to the weights
        during optimization.
    num_workers : int, default=10
        The number of subprocesses to use for data loading. More workers can
        increase the data loading speed but require more system resources.
    num_iterations : int, default=400000
        The total number of training iterations to perform.
    drop_lr_at : np.ndarray, default=np.array([150000, 300000])
        Iterations at which the learning rate should be reduced.
    drop_lr_factor : np.ndarray, default=0.1
        The factor by which the learning rate should be reduced at each specified
        point in `drop_lr_at`.
    gradient_accumulation : int, default=1
        The number of gradients to accumulate before performing an optimizer
        step. This can be used to effectively increase the batch size without
        increasing the memory requirements.
    """
    optimizer_name: str = 'AdamW'
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 0
    num_workers: int = 10
    num_iterations: int = 400_000
    drop_lr_at: np.ndarray = field(default_factory=lambda: np.array([150_000, 300_000]))
    drop_lr_factor: np.ndarray = 0.1
    gradient_accumulation: int = 1
