import json
from typing import List, Tuple

import numpy as np
import cv2

from .calibration import calibrate_lens
from ...aruco import ArucoCalibrationBoard


class Lens:
    """
    A class representing the lens of a camera, including its intrinsic matrix and distortion coefficients.

    This class encapsulates the properties of a camera lens that are essential for correcting lens distortion,
    including the intrinsic camera matrix and the distortion coefficients calculated through calibration.

    Parameters
    ----------
    intrinsic_matrix : np.ndarray
        The intrinsic camera matrix of the lens.
    distortions_coefs : np.ndarray
        The distortion coefficients of the lens.

    Attributes
    ----------
    intrinsic : np.ndarray
        Stores the intrinsic camera matrix.
    distortions : np.ndarray
        Stores the distortion coefficients.
    """

    def __init__(self, intrinsic_matrix: np.ndarray, distortions_coefs: np.ndarray):
        """
        Initializes the Lens object with the intrinsic matrix and distortion coefficients.
        """
        self.intrinsic = intrinsic_matrix
        self.distortions = distortions_coefs

    def to_disk(self, file_name: str):
        """
        Saves the intrinsic matrix and distortion coefficients to a JSON file.

        Parameters
        ----------
        file_name : str
            The path where the lens data should be saved.
        """
        to_write = {
            'intrinsic': self.intrinsic.tolist(),
            'distortions': self.distortions.tolist()
        }

        with open(file_name, 'w') as handle:
            json.dump(to_write, handle)

    def undistort(self, image: np.ndarray) -> np.ndarray:
       """
       Corrects distortion in the provided image using the camera intrinsic parameters and distortion coefficients.

       This method uses OpenCV's `cv2.undistort` function to remove distortion from an image. The intrinsic camera matrix
       and the distortion coefficients, which should be predefined properties of the class, are used in this process.
       The corrected image is computed using the same intrinsic matrix for the new  camera matrix.

       Parameters
       ----------
       image : np.ndarray
           The distorted image to be corrected. This should be a NumPy array, typically of dtype `np.uint8` and with shape
           (height, width, channels) for a color image or (height, width) for a grayscale image.

       Returns
       -------
       np.ndarray
           The undistorted image as a NumPy array of the same shape and dtype as the input image.

       """
       return cv2.undistort(image, self.intrinsic, self.distortions, None, self.intrinsic)

    @staticmethod
    def from_disk(file_name: str) -> 'Lens':
        """
        Loads the lens data (intrinsic matrix and distortion coefficients) from a JSON file.

        Parameters
        ----------
        file_name : str
            The path of the file from which to load the lens data.

        Returns
        -------
        Lens
            An instance of Lens initialized with the data loaded from the file.
        """
        with open(file_name, 'r') as handle:
            data = json.load(handle)
        intrinsic_matrix = np.array(data['intrinsic'])
        distortions_coefs = np.array(data['distortions'])
        return Lens(intrinsic_matrix, distortions_coefs)

    @staticmethod
    def calibrate_from_measurements(image_filenames: List[str],
                                    board: ArucoCalibrationBoard,
                                    pixel_size_mm: float,
                                    image_clip_percentiles: List[int] = [15, 85]) -> Tuple['Lens', np.ndarray]:
        """
        Calibrates the camera lens using images of an ArUco calibration board.

        This static method processes a series of images containing an ArUco calibration board, detects
        the markers, and uses them to calibrate the camera lens. The calibration process determines
        the camera's intrinsic matrix and distortion coefficients, which are essential for correcting lens
        distortion in other images captured with the same camera.

        Parameters
        ----------
        image_filenames : List[str]
            A list of paths to the images used for calibration.
        board : ArucoCalibrationBoard
            An instance of ArucoCalibrationBoard that provides functionalities for marker detection and
            generating corresponding 3D points for calibration.
        pixel_size_mm : float
            The size of one pixel in millimeters. This is used to scale the physical dimensions of the
            calibration board to the image dimensions.
        image_clip_percentiles : List[int], optional
            Percentiles used for clipping the image intensities before ArUco detection. This can improve
            marker detection in images with varying lighting conditions. Defaults to [15, 85].

        Returns
        -------
        Lens
            An instance of the Lens class initialized with the calibrated intrinsic matrix and distortion coefficients.
        np.ndarray
            An array of calibration errors for each image used in the calibration process.
        """
        intrinsic, dist, errors = calibrate_lens(image_filenames,
                                                 board, pixel_size_mm, image_clip_percentiles)
        return Lens(intrinsic, dist), errors
