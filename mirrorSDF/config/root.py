import json
from dataclasses import dataclass, is_dataclass, field
from typing import get_type_hints

import numpy as np

from .architecture import ArchitectureConfig
from .calibration import CalibrationFilesConfig
from .dataset import DatasetConfig
from .diffuse_irradiance import DiffuseIrradianceConfig
from .logging import LoggingConfig
from .rendering import RenderingConfig
from .training import TrainingConfig, ValidationConfig


def custom_serializer(obj):
    """
    Custom JSON serializer for objects not serializable by default json code.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    if isinstance(obj, (np.float32, np.float64, np.float_)):  # Handle numpy floats
        return float(obj)  # Convert to Python float
    if isinstance(obj, (np.int32, np.int64, np.int_)):  # Handle numpy integers
        return int(obj)  # Convert to Python int
    if isinstance(obj, np.bool_):  # Handle numpy bool
        return bool(obj)  # Convert to Python bool
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def serialize_dataclass(obj):
    """
    Recursively convert a dataclass to a dictionary, handling numpy arrays.
    """
    if is_dataclass(obj):
        result = {}
        for current_field in obj.__dataclass_fields__:
            value = getattr(obj, current_field)
            result[current_field] = serialize_dataclass(value)  # Recursively serialize dataclass fields
        return result
    elif isinstance(obj, list):
        return [serialize_dataclass(item) for item in obj]  # Handle lists
    elif isinstance(obj, dict):
        return {key: serialize_dataclass(value) for key, value in obj.items()}  # Handle dictionaries
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    else:
        return obj  # Return the object itself if none of the above


def from_dict(cls, data):
    """Recursively constructs a data class instance from a dictionary."""
    if not data:
        return None  # Handle empty data case

    # Get type hints for the data class fields
    hints = get_type_hints(cls)
    field_values = {}
    for current_field, field_type in hints.items():
        if current_field in data:
            # Check if the field_type is a data class and recursively construct it
            if hasattr(field_type, '__dataclass_fields__'):
                field_values[current_field] = from_dict(field_type, data[current_field])
            elif field_type == np.ndarray:
                field_values[current_field] = np.array(data[current_field])
            else:
                field_values[current_field] = data[current_field]
    return cls(**field_values)


@dataclass
class MirrorSDFConfig:
    """
    Root config object.

    This class aggregates configurations for calibration files, dataset specifics,
    architectural configurations, pre-training settings, logging, validation, and
    rendering for both training and evaluation phases.

    Attributes
    ----------
    calibration_files : CalibrationFilesConfig
        Configuration for calibration files specific to the hardware setup, like sensors,
        lenses, and mirrors.
    dataset : DatasetConfig
        Configuration settings for the dataset used in training or evaluation.
    architecture : ArchitectureConfig
        Configuration related to the neural network architectures
    background_training : TrainingConfig
        Configuration for pre-training the environment. including optimizer, batch size,
        learning rate, etc., with a default set of parameters.
    logging : LoggingConfig
        Settings for logging activities, output directories, and integration with
        external logging tools like Weights & Biases.
    validation : ValidationConfig
        Configuration of model evaluation.
    rendering_train : RenderingConfig, default=RenderingConfig()
        Rendering configurations for the training phase.
    rendering_eval : RenderingConfig
        Rendering configurations for the evaluation phase, with a higher sample
        per pixel (SPP) setting by default for higher quality rendering.
    diffuse_irradiance: DiffuseIrradianceConfig
        Configuration related to the generation of the diffuse irradiance distillation dataset

    Examples
    --------
    >>> loaded_config = MirrorSDFConfig.from_disk("config.json")
    >>> mirror_sdf_config.to_disk("config.json")
    """

    calibration_files: CalibrationFilesConfig
    dataset: DatasetConfig
    architecture: ArchitectureConfig = field(default_factory=lambda: ArchitectureConfig())
    background_training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        learning_rate=0.001,
    ))
    envnet_training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        batch_size=4096,
    ))
    foreground_training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        batch_size=256,
    ))
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())
    validation: ValidationConfig = field(default_factory=lambda: ValidationConfig())
    rendering_train: RenderingConfig = field(default_factory=lambda: RenderingConfig())
    rendering_eval: RenderingConfig = field(default_factory=lambda: RenderingConfig(
        background_spp=64
    ))
    diffuse_irradiance: DiffuseIrradianceConfig = field(default_factory=lambda: DiffuseIrradianceConfig())

    def to_dict(self):
        """Serializes the current configuration to a dictionary."""
        return serialize_dataclass(self)

    def to_disk(self, file_name: str):
        """
        Serializes the configuration to a JSON file on disk.

        Parameters
        ----------
        file_name : str
            The name of the file where the configuration will be saved.
        """
        with open(file_name, 'w') as f:
            json.dump(self.to_dict(), f, default=custom_serializer, indent=4)

    @classmethod
    def from_disk(cls, file_name: str) -> 'MirrorSDFConfig':
        """
        Deserializes the configuration from a JSON file on disk into an instance of MirrorSDFConfig.

        Parameters
        ----------
        file_name : str
            The name of the file from which to load the configuration.

        Returns
        -------
        MirrorSDFConfig
            An instance of MirrorSDFConfig with the loaded settings.
        """
        with open(file_name, 'r') as f:
            config_dict = json.load(f)
        return from_dict(cls, config_dict)