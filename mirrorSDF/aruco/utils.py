from typing import List, Optional, Tuple

import numpy as np
import cv2


def prepare_image_for_aruco(image: np.ndarray, percentiles: Tuple[int, int] = (15, 85),
                            low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
    """
    Prepare an image for ArUco marker detection by adjusting its contrast using percentiles
    or optional explicit low and high values.

    This function converts the input image to grayscale if it is not already, and then adjusts its contrast based
    on the specified lower and upper percentiles or the optional explicit low and high values provided.
    The contrast adjustment is done by scaling the pixel values so that the intensities corresponding to
    the lower percentile or low value become 0 and the upper percentile or high value becomes 255.
    This improves the detection of ArUco markers in the image by making the markers
    more distinguishable from the background.

    Parameters
    ----------
    image : np.ndarray
        The input image as a NumPy array. If the image has more than one channel (e.g., RGB),
        it will be converted to grayscale by averaging the channels.
    percentiles : Tuple[int, int], optional
        A list of two integers specifying the lower and upper percentiles used for contrast adjustment.
        The default values are [15, 85].
    low : Optional[float], optional
        An optional lower bound value for contrast adjustment.
        If not specified, the lower percentile value is used. Defaults to None.
    high : Optional[float], optional
        An optional upper bound value for contrast adjustment.
        If not specified, the upper percentile value is used. Defaults to None.

    Returns
    -------
    np.ndarray
        The contrast-adjusted grayscale image as a NumPy array suitable for ArUco marker detection.

    Examples
    --------
    >>> import cv2
    >>> import numpy as np
    >>> image = cv2.imread('path/to/your/image.jpg')
    >>> prepared_image = prepare_image_for_aruco(image, percentiles=(10, 90))
    >>> # Using explicit low and high values for contrast adjustment
    >>> prepared_image_with_explicit_bounds = prepare_image_for_aruco(image, low=50, high=200)
    """
    if image.ndim == 3:  # Indicates that the image has channels
        image = image.mean(-1)  # Convert to grayscale by averaging channels
    # Calculate the low and high percentiles of the grayscale image if not provided explicitly
    c_low, c_high = np.percentile(image, percentiles)
    if low is None:
        low = c_low
    if high is None:
        high = c_high
    # Calculate the alpha (contrast) and beta (brightness) values for contrast adjustment
    alpha = 1 / (high - low) * 255
    beta = -low * alpha
    # Adjust the contrast of the image and convert it to 8-bit per channel
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def compute_square_centers(square_coords: np.ndarray) -> np.ndarray:
    """
    Computes the centers of squares given their corner coordinates.

    Parameters
    ----------
    square_coords : np.ndarray
        An array of square corner coordinates in the format [top-left, top-right, bottom-right, bottom-left].

    Returns
    -------
    np.ndarray
        The center coordinates of the squares.
    """
    # Unpack the corner coordinates
    A, C, B, D = square_coords.transpose(1, 0, 2)
    d1 = B - A  # Vector from A to B
    d2 = D - C  # Vector from C to D
    t = np.cross((C - A), d2) / np.cross(d1, d2)  # Compute intersection parameter for line AB with CD
    return A + t[:, None] * d1  # Calculate and return the centers
