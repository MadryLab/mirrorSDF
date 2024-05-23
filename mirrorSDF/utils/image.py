from typing import Union

import cv2
import numpy as np
import rawpy
import warnings
import subprocess

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..optical_models import RGBImageSensor


def read_exif_entry(file_name: str, tag: str) -> float:
    """
    Extracts a specific EXIF tag value from an image file using the `exiftool` command-line application.

    Parameters
    ----------
    file_name : str
        The path to the image file from which to read the EXIF data.
    tag : str
        The specific EXIF tag (e.g., 'ExposureTime', 'ISO') to extract from the file.

    Returns
    -------
    float
        The value of the specified EXIF tag as a float.

    Raises
    ------
    FileNotFoundError
        If the `exiftool` command fails to execute, possibly indicating that `exiftool` is not installed.

    Notes
    -----
    This function relies on the external application `exiftool` to extract EXIF information.
    Ensure that `exiftool` is installed and accessible from the system's PATH.

    Examples
    --------
    >>> read_exif_entry('/path/to/image.jpg', 'ExposureTime')
    0.0125
    """
    # Prepare the command for subprocess call
    command = ['exiftool', '-b', f'-{tag}', file_name]
    # Execute the command, capturing output and errors in text mode
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Check if the command executed successfully
    if result.returncode != 0:
        raise FileNotFoundError("exiftool command failed, make sure it is installed on your system")
    # Return the tag value converted to float
    return float(result.stdout.strip())


def read_exposure_factor(file_name: str) -> float:
    """
    Calculates the exposure factor for an image based on its EXIF data.

    Parameters
    ----------
    file_name : str
        The file name of the image for which the exposure factor is calculated.

    Returns
    -------
    float
        The exposure factor, calculated as ISO/100 * exposure time.

    Examples
    --------
    >>> read_exposure_factor('/path/to/image.jpg')
    1.28

    Notes
    -----
    The exposure factor is a product of the ISO value divided by 100 and the exposure time.
    This function relies on `read_exif_entry` to retrieve the required EXIF data.
    """
    # Read the exposure time from the image's EXIF data
    exposure_time = read_exif_entry(file_name, 'ExposureTime')
    # Read the ISO from the image's EXIF data
    iso = read_exif_entry(file_name, 'ISO')
    # Calculate the exposure factor
    factor = iso / 100 * exposure_time
    return factor


def read_canon_raw(file_name: str, half_size: bool = True, custom_white_balance: np.ndarray = None) -> np.ndarray:
    """
    Reads a Canon RAW image file and processes it into a NumPy array.

    This function uses `rawpy` to read and process a Canon RAW image file. It allows
    for custom white balance settings and the option to process the image at half
    its original size to reduce noise

    Parameters
    ----------
    file_name : str
        The path to the Canon RAW image file to be read.
    half_size : bool, optional
        Whether to process the image at half its original size. Default is True,
        decreases noise but reduces the image resolution.
    custom_white_balance : np.ndarray, optional
        A custom white balance to apply to the image. If None, the camera's
        white balance is used. If used it should be a tuple/list of 4 values (RGGB).
        Default is None.

    Returns
    -------
    np.ndarray
        The processed image as a NumPy array with 16 bits per sample.

    Examples
    --------
    >>> img = read_canon_raw('path/to/image.CR2')
    >>> img.shape
    (height, width, 3)
    """
    image = rawpy.imread(file_name)  # Load RAW image

    if custom_white_balance is None:
        custom_white_balance = image.camera_whitebalance  # Use camera's white balance if none is provided

    result = image.postprocess(output_bps=16, half_size=half_size,
                               user_wb=custom_white_balance,
                               no_auto_bright=True)  # Process image
    return result


def imread(file_name: str, *reader_args, **reader_kwargs):
    """
    Reads an image file into a NumPy and it's exposure information.

    This function tries to read an image file using OpenCV. If OpenCV cannot read the file
    (likely because it is a RAW file), it falls back to using `read_canon_raw` for reading.
    The function automatically converts BGR images (OpenCV default) to RGB.
    It also returns the exposure factor of the image by calling `read_exposure_factor`.

    Parameters
    ----------
    file_name : str
        The path to the image file to be read.
    *reader_args :
        Variable length argument list for reader function.
    **reader_kwargs :
        Arbitrary keyword arguments for reader function.

    Returns
    -------
    tuple[np.ndarray, float]
        A tuple containing the processed image as a NumPy array and the exposure factor.

    Raises
    ------
    Exception
        Propagates any exceptions raised during image reading and processing.

    Examples
    --------
    >>> img, exposure = imread('path/to/image.jpg')
    >>> img.shape
    (height, width, 3)
    >>> exposure
    1.0
    """
    try:
        image = cv2.imread(file_name, *reader_args, **reader_kwargs)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        image = read_canon_raw(file_name, *reader_args, **reader_kwargs)

    return image, read_exposure_factor(file_name)


def read_hdr_stack(file_names, sensor: 'RGBImageSensor', white_balance=np.ones(3),
                   exponent: float = 2, saturation_threshold: float = 0.05):
    images, exposures = zip(*[imread(file_name) for file_name in file_names])
    images = np.stack(images, 2)
    exposures = np.array(exposures)

    if len(set(exposures.tolist())) != len(exposures):
        raise ValueError('You have images with duplicated exposure, you probably messed up your file ordering')

    fused, under_exposed, over_exposed = sensor.hdr_fusion(images, exposures, exponent, saturation_threshold)
    if under_exposed > 0 or over_exposed > 0:
        warnings.warn(f"{under_exposed} under-exposed and {over_exposed} \
        overexposed pixels in the image stack {file_names[0]}...{file_names[1]}")

    return fused * white_balance[None, None]


def generate_pixel_grid(height: int, width: int) -> np.ndarray:
    """
    Generates a 2D pixel grid coordinates for an image or a 2D array.

    This function creates a grid of (x, y) coordinates that correspond to the
    pixel positions in an image or a matrix of dimensions `height` x `width`.

    Parameters
    ----------
    height : int
        The height of the grid (number of rows).
    width : int
        The width of the grid (number of columns).

    Returns
    -------
    np.ndarray
        An array of shape `(height*width, 2)`, where each row contains the
        (x, y) coordinates of a pixel. The first column contains x coordinates,
        and the second column contains y coordinates.

    Examples
    --------
    >>> generate_pixel_grid(2, 3)
    array([[0., 0.],
           [1., 0.],
           [2., 0.],
           [0., 1.],
           [1., 1.],
           [2., 1.]], dtype=float32)
    """
    xes = np.arange(width, dtype=np.float32)  # Generate x coordinates
    yes = np.arange(height, dtype=np.float32)  # Generate y coordinates
    # Create a grid and reshape it into a list of coordinates
    pixel_coords = np.stack(np.meshgrid(xes, yes), -1).reshape(-1, 2)
    return pixel_coords




