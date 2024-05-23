from dataclasses import dataclass
from os import path
from typing import Optional


@dataclass
class LoggingConfig:
    """
    Configuration settings for logging activities within a model training and evaluation process.

    This class encapsulates settings related to logging, such as output directory,
    checkpoint file naming, logging frequency, and integration with external logging
    frameworks like Weights & Biases (wandb).

    Parameters
    ----------
    output_folder : Optional[str], default=None
        The folder path where logs and other output files should be saved.
        If `None`, nothing will be written to dis
    background_checkpoint_file : str, default="background_model.pth"
        The file name for saving background model checkpoints.
    envnet_checkpoint_file : str, default="envnet_model.pth"
        The file name for saving background model checkpoints.
    log_every_iter : int, default=20
        Frequency of logging information, specified in terms of iterations.
        For example, if `log_every_iter=20`, logging will occur every 20 iterations.
    use_wandb : bool, default=True
        Specifies whether to use Weights & Biases for logging. If `True`, integration
        with Weights & Biases is enabled, requiring an active account and network
        access for synchronization.
    """

    output_folder: Optional[str] = None
    background_checkpoint_file: str = "background_model.pth"
    envnet_checkpoint_file: str = "envnet_model.pth"
    diffuse_irradiance_dataset: str = "diffuse_irradiance.npy"
    precomputed_background_values: str = "background_predictions.npy"
    log_every_iter: int = 20
    use_wandb: bool = True

    def get_full_path(self, filename: str):
        if self.output_folder is not None:
            filename = path.join(self.output_folder, filename)
        return filename
