from dataclasses import dataclass


@dataclass
class CalibrationFilesConfig:
    """
    A configuration class for managing calibration file settings for various components
    of an imaging or sensing system.

    This class stores the file names or paths for calibration data related to the system's
    sensor, lens, and mirror. It is designed to simplify the configuration management
    process by grouping these settings into a single, easy-to-manage object.

    Parameters
    ----------
    sensor : str
        The file name or path for the sensor's calibration data.
    lens : str
        The file name or path for the lens's calibration data.
    mirror : str
        The file name or path for the mirror's calibration data.

    Examples
    --------
    Creating a CalibrationFilesConfig object with specific calibration files:

    >>> calibration_config = CalibrationFilesConfig(
    ...     sensor="sensor_calib.npy",
    ...     lens="lens_calib.json",
    ...     mirror="mirror_calib.json"
    ... )
    """

    sensor: str
    lens: str
    mirror: str
