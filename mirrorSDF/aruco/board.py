from typing import Tuple
from cv2 import aruco
import cv2
import numpy as np


class ArucoCalibrationBoard:
    """
    A class to create and manage an ArUco marker calibration board.

    This class allows for the generation of a custom ArUco marker board image based on given
    parameters such as resolution and marker distribution. It supports detection of markers
    in images and can be used to generate a PNG image of the board.

    Parameters
    ----------
    resolution_width : int
        The desired width of the generated calibration board image in pixels.
    resolution_height : int
        The desired height of the generated calibration board image in pixels.
    aruco_dict_id : int, optional
        The dictionary ID of the ArUco markers to use. Defaults to `aruco.DICT_ARUCO_ORIGINAL`.
    num_horizontal_markers : int, optional
        The number of markers to place horizontally on the board. Defaults to 16.
    target_padding_ratio : float, optional
        The ratio of padding to marker size around each marker. Defaults to 0.3.
    lowest_marker_id : int, optional
        The first marker id index to include in the board. Defaults to 0.

    Attributes
    ----------
    image : np.ndarray
        The generated calibration board image. Shape: (resolution_height, resolution_width)
    objp : np.ndarray
        The array of object points in the calibration board.
    detector : aruco_ArucoDetector
        An ArUco detector configured for the dictionary used by this board.

    """

    def __init__(self, resolution_width: int, resolution_height: int,
                 aruco_dict_id: int = aruco.DICT_ARUCO_ORIGINAL,
                 num_horizontal_markers: int = 16, target_padding_ratio: float = 0.3,
                 lowest_marker_id: int = 0):

        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.n_markers_hz = num_horizontal_markers
        self.padding_ratio = target_padding_ratio

        self._init_grid_sizes()

        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_dict.bytesList = self.aruco_dict.bytesList[lowest_marker_id:]

        self.grid = aruco.CharucoBoard(self.grid_dims, self.marker_size,
                                       self.padding, self.aruco_dict)

        generated_resolution = (
            int(self.resolution_width - np.sum(self.leftover_hz)),
            int(self.resolution_height - np.sum(self.leftover_vt)),
        )

        self.image = self.grid.generateImage(generated_resolution, generated_resolution, 0)
        self.image = np.pad(self.image, (self.leftover_vt, self.leftover_hz), constant_values=255)

        self.objp = np.array(self.grid.getChessboardCorners() + np.array([
            self.leftover_hz[0],
            self.leftover_vt[0],
            0
        ])[None], dtype=np.float64)

        self.detector = aruco.CharucoDetector(self.grid)

    def to_png(self, file_path: str):
        """
        Saves the generated calibration board image as a PNG file.

        Parameters
        ----------
        file_path : str
            The path where the PNG image will be saved. Must end with '.png'.

        Raises
        ------
        ValueError
            If the file path does not end with '.png'.

        """
        if not file_path.endswith('.png'):
            raise ValueError('File extension must be png')

        if not cv2.imwrite(file_path, self.image):
            raise OSError("Writing Failed")

    def _compute_leftover(self, resolution: int, n_markers: int) -> Tuple[int, int]:
        """
        Computes the leftover space after placing markers and padding in one dimension.

        Parameters
        ----------
        resolution : int
            The size of the dimension (width or height) in pixels.
        n_markers : int
            The number of markers along the dimension.

        Returns
        -------
        tuple[int, int]
            The leftover space at the beginning and end of the dimension.

        """
        leftover = resolution - n_markers * self.marker_size
        left_x = leftover // 2
        left_y = leftover - left_x
        return left_x, left_y

    def _init_grid_sizes(self):
        """
        Initializes grid sizes, padding, and calculates leftover space for padding around the board.
        """
        self.marker_size = int(np.floor(self.resolution_width / self.n_markers_hz))
        self.padding = (1 - self.padding_ratio) * self.marker_size
        self.n_markers_vt = int(np.floor(self.resolution_height / self.marker_size))

        self.leftover_hz = self._compute_leftover(self.resolution_width,
                                                  self.n_markers_hz)
        self.leftover_vt = self._compute_leftover(self.resolution_height,
                                                  self.n_markers_vt)

        self.grid_dims = (self.n_markers_hz, self.n_markers_vt)

    def detect(self, image: np.ndarray, pixel_size_mm: float = None, marker_size_mm: float = None) -> tuple:
        """
        Detects ArUco markers in the provided image and returns their corner coordinates, IDs, and targets.

        Parameters
        ----------
        image : np.ndarray
            The image to detect ArUco markers in.
        pixel_size_mm : float, optional
            The physical size of one pixel in millimeters. If not provided, `marker_size_mm` must be given.
        marker_size_mm : float, optional
            The physical size of a single marker in millimeters. If not provided, `pixel_size_mm` must be given.

        Returns
        -------
        tuple
            A tuple containing corner coordinates, corner IDs, and targets of the detected markers.

        Raises
        ------
        ValueError
            If neither `pixel_size_mm` nor `marker_size_mm` is specified.

        """
        image = cv2.convertScaleAbs(image, alpha=1)
        if pixel_size_mm is None and marker_size_mm is None:
            raise ValueError("You need to specify the physical size of the grid one way or another")

        if pixel_size_mm is None:
            pixel_size_mm = marker_size_mm / self.marker_size

        objp = self.objp * pixel_size_mm

        corner_coords, corner_ids = self.detector.detectBoard(image)[:2]
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        corner_coords = cv2.cornerSubPix(image, corner_coords, (3, 3), (-1, -1), term)
        corner_ids = corner_ids.squeeze()
        corner_coords = np.concatenate(corner_coords, axis=0)

        targets = objp[corner_ids]

        return corner_coords, corner_ids, targets
