import numpy as np
import numba as nb
from tqdm.autonotebook import tqdm

from .calibration import calibrate_single_channel
from .calibration import compute_weight_vector


class RGBImageSensor:
    """
    A class representing an RGB image sensor with calibrated response curves.

    This class is used to model the response of an RGB image sensor to different
    levels of illumination, based on calibration data.

    It allows for linearization of the response of a sensor and HDR recovery
    from multiple exposure shots

    Parameters
    ----------
    response_curves : np.ndarray
        A 2D numpy array of shape (3, Z_max), where `Z_max` is determined by
        the bit depth of the input images. Each row corresponds to the calibrated
        response curve for one of the RGB channels.

    Attributes
    ----------
    curves : np.ndarray
        A 2D numpy array of shape (3, Z_max). Stores the calibrated response curves
        for the RGB channels.
    """

    def __init__(self, response_curves: np.ndarray):
        """
        Initializes the RGBImageSensor with the given response curves.

        Parameters
        ----------
        response_curves : np.ndarray
            The calibrated response curves for the RGB channels.
        """
        self.curves = response_curves

    def to_disk(self, file_name: str):
        """
        Saves the calibrated response curves to disk.

        Parameters
        ----------
        file_name : str
            The path where the response curves should be saved, including file name.
        """
        np.save(file_name, self.curves)

    @staticmethod
    def from_disk(file_name: str) -> 'RGBImageSensor':
        """
        Loads calibrated response curves from disk and returns an RGBImageSensor instance.

        Parameters
        ----------
        file_name : str
            The path of the file from which to load the response curves.

        Returns
        -------
        RGBImageSensor
            An instance of RGBImageSensor initialized with the loaded response curves.
        """
        curves = np.load(file_name)
        return RGBImageSensor(curves)

    @staticmethod
    def calibrate_from_measurements(images: np.ndarray, exposures: np.ndarray,
                                    batch_size: int = 16384,
                                    n_iterations: int = 150) -> 'RGBImageSensor':
        """
        Calibrates all three color channels from a set of images taken at different
        exposure levels and returns the calibrated sensor.

        This static method processes each color channel (red, green, and blue) through
        `calibrate_single_channel` to compute the camera response functions. It's
        designed to work with a batch of images taken at different exposure times
        to generate a high dynamic range (HDR) image.

        Parameters
        ----------
        images : np.ndarray
            A 4D numpy array of shape (height, width, num_images, channels) containing
            the images for calibration. Each image must be of the same size and alignment.
        exposures : np.ndarray
            A 1D numpy array containing relative exposure factors for each image.
        batch_size : int, optional
            The size of batches to use for processing images during optimization.
            Default is 16384.
        n_iterations : int, optional
            The number of iterations to run the optimization process for each channel.
            Default is 150.

        Returns
        -------
        RGBImageSensor
            A sensor instance loaded with the curves obtained from the calibration data.

        Examples
        --------
        >>> images = np.random.randint(0, 256, size=(100, 100, 5, 3), dtype=np.uint8)
        >>> exposures = np.array([1/60, 1/30, 1/15, 1/8, 1/4])
        >>> sensor = RGBImageSensor.calibrate_from_measurements(images, exposures)
        >>> sensor.curves.shape
        (3, 256)
        """
        curves = []
        for c in tqdm(range(3), desc='Calibrating channels'):
            # Calibrate each channel individually, using provided images and exposures.
            curve = calibrate_single_channel(images[:, :, :, c], exposures, batch_size, n_iterations)
            curves.append(curve)
        curves = np.array(curves)  # Combine the curves from all channels into a single array.
        return RGBImageSensor(curves)

    def linearize_measurement(self, data: np.ndarray) -> np.ndarray:
        """
        Linearizes the measurements of RGB values based on the sensor's calibrated response curves.

        Parameters
        ----------
        data : np.ndarray
            A 3D numpy array of shape (..., 3), containing the raw data from
            the sensor. The last dimension must be 3, corresponding to the RGB channels.

        Returns
        -------
        np.ndarray
            A 3D numpy array of the same shape as `data`, where each RGB value has been
            linearized according to the sensor's response curves.

        Raises
        ------
        ValueError
            If the last dimension of `data` is not 3, indicating an incorrect shape
            for RGB data.

        """
        if data.shape[-1] != 3:
            raise ValueError("The last dimension of the input tensor must be 3 for the 3 channels.")

        linearized_data = np.empty(shape=data.shape, dtype=self.curves.dtype)

        for channel in range(3):
            # Use pixel values as indices into the response curve for linearization
            linearized_data[..., channel] = self.curves[channel, data[..., channel]]

        return linearized_data

    def hdr_fusion(self, raw_data: np.ndarray, exposures: np.ndarray,
                   exponent: float = 2.0, saturation_threshold: float = 0.05) -> (np.ndarray, int, int):
        """
        Fuse multiple exposures of an image into a single HDR image using the sensor's response curves and
        a custom weight vector computed from the provided exponent.

        Parameters
        ----------
        raw_data : np.ndarray
            The raw image data as a 3D array (height, width, number of exposures).
        exposures : np.ndarray
            The exposure times for each image, represented as a 1D array.
        exponent : float, optional
            The exponent used to compute the weight vector, affecting how pixel weights decrease with distance
            from the median intensity. The higher the value the less confidence it will give to extreme measurements
            Defaults to 1.0.
        saturation_threshold: float
            If the total confidence in a pixel is below this threshold we rely only
            on the most extreme value and do not attempt fusion at all. Defaults to 0.05

        Returns
        -------
        np.ndarray
            The fused HDR image as a 3D array (height, width, color channels).
        int
            The number of under-saturated pixels in the HDR image.
        int
            The number of over-saturated pixels in the HDR image.

        Raises
        ------
        ValueError
            If the data type doesn't correspond to the bit depth of this sensor
        """
        z_max = self.curves.shape[1]

        if z_max != np.iinfo(raw_data.dtype).max + 1:
            raise ValueError("Please pass raw data not linearized")

        weights = compute_weight_vector(z_max, exponent).numpy()

        fused, under_saturated_pixels, over_saturated_pixels = fuse_hdr(
            raw_data, self.curves, exposures, weights, z_max, saturation_threshold
        )
        return fused, under_saturated_pixels, over_saturated_pixels


@nb.njit()
def fuse_hdr( raw_data: np.ndarray, sensor_curves: np.ndarray, exposures: np.ndarray,
              weights: np.ndarray, z_max: int, saturation_threshold: float = 0.05) -> (np.ndarray, int, int):
    """
    Fuse multiple exposures of an image into a single HDR image using Debevec's method.

    Parameters
    ----------
    raw_data : np.ndarray
        The raw image data as a 4D array (height, width, number of exposures, color channels).
    sensor_curves : np.ndarray
        The camera response curves for each color channel as a 2D array (color channels, sensor value).
    exposures : np.ndarray
        The exposure times for each image, represented as a 1D array.
    weights : np.ndarray
        The weights to give to a given measurement, indicating the confidence in their contribution to the HDR image.
    z_max : int
        The maximum number of values the sensor can measure
    saturation_threshold: float
        If the total confidence in a pixel is below this threshold we rely only
        on the most extreme value and do not attempt fusion at all.

    Returns
    -------
    np.ndarray
        The fused HDR image as a 3D array (height, width, color channels).
    int
        The number of under-saturated pixels in the HDR image.
    int
        The number of over-saturated pixels in the HDR image.

    References
    ----------
    Debevec, P.E., Malik, J.: Recovering high dynamic range radiance maps from photographs.
    In: Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques,
    SIGGRAPH 1997, pp. 369–378. ACM Press/Addison-Wesley Publishing Co. (1997)
    """
    # Find the indices of the lowest and highest exposures
    lowest_exposure = exposures.argmin()
    highest_exposure = exposures.argmax()
    # Compute the logarithm (base 2) of the exposure times
    log_exposures = np.log2(exposures)
    # Initialize counters for over and under-saturated pixels
    over_saturated_pixels = 0
    under_saturated_pixels = 0
    # Determine the number of shots (exposures)
    num_shots = raw_data.shape[2]
    # Initialize the result HDR image array with zeros
    result = np.zeros((raw_data.shape[0], raw_data.shape[1], 3), dtype=np.float32)

    # Iterate over each pixel in the image
    for i in range(raw_data.shape[0]):
        for j in range(raw_data.shape[1]):
            for c in range(3):  # For each color channel
                measurements = raw_data[i, j, :, c]
                coefficients = weights[measurements]
                total = coefficients.sum()
                if total <= saturation_threshold:  # Handle cases with negligible total weight
                    coefficients[:] = 0
                    if measurements.mean() > z_max / 2:  # Over-saturated pixel
                        over_saturated_pixels += 1
                        coefficients[lowest_exposure] = 1
                    else:  # Under-saturated pixel
                        under_saturated_pixels += 1
                        coefficients[highest_exposure] = 1
                else:
                    coefficients /= total  # Normalize coefficients

                pixel_value = 0
                # Compute the HDR value for the pixel
                for k in range(num_shots):
                    pixel_value += (np.log2(sensor_curves[c, measurements[k]]) - log_exposures[k]) * coefficients[k]
                pixel_value = 2 ** pixel_value  # Convert log value back
                result[i, j, c] = pixel_value

    return result, under_saturated_pixels, over_saturated_pixels
