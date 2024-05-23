from typing import Optional, Tuple
import numpy as np
import cv2
from cv2 import aruco


class AruCrosshair:
    """
    Represents a single crosshair composed of two ArUco markers.

    Parameters
    ----------
    id_1 : int
        The ID of the first marker.
    id_2 : int
        The ID of the second marker.
    aruco_dict : cv2.aruco.Dictionary
        The ArUco dictionary object for marker generation.

    Attributes
    ----------
    id_1 : int
        The ID of the first marker.
    id_2 : int
        The ID of the second marker.
    aruco_dict : cv2.aruco.Dictionary
        The ArUco dictionary used for marker generation.
    """

    def __init__(self, id_1, id_2, aruco_dict):
        self.id_1 = id_1
        self.id_2 = id_2
        self.aruco_dict = aruco_dict

    def compute_objp(self, diagonal_mm: float) -> np.ndarray:
        """
        Computes the 3D object points of each corner of the arucos that the crosshair consists in.

        Parameters
        ----------
        diagonal_mm : float
            The diagonal size of the crosshair in millimeters.

        Returns
        -------
        np.ndarray
            The 3D object points of the crosshair.
        """
        measured = diagonal_mm / np.sqrt(2)  # Calculate the side length of the square from the diagonal
        marker_size, padding = self._get_dims(1)  # Get marker and padding dimensions for a unit size
        expected = 2 * (padding + marker_size)  # Calculate the expected total size
        full_size = measured / expected  # Scale factor to apply to dimensions

        m, p = self._get_dims(full_size)  # Get actual marker and padding dimensions

        # Define the coordinates of the marker corners in the plane
        points_plane = np.array([
            # first marker
            [
                [-p, p],
                [-p, m + p],
                [-m - p, m + p],
                [-m - p, p],
            ],
            # second marker
            [
                [m + p, - m - p],
                [m + p, -p],
                [p, -p],
                [p, - m - p],
            ]
        ])
        # Convert to 3D points by adding a zero z-coordinate
        points_3d = np.pad(points_plane[:, :, : None], ((0, 0), (0, 0), (0, 1)), constant_values=0)
        return points_3d

    @staticmethod
    def _get_dims(total_size: float) -> Tuple[float, float]:
        """
        Calculates the dimensions of the markers and padding based on the total size.

        Parameters
        ----------
        total_size : float
            The total size available for the marker and padding.

        Returns
        -------
        Tuple[float, float]
            A tuple containing the marker size and padding size.
        """
        ratio = 0.7
        marker_size = total_size / 2 * ratio
        padding = total_size / 4 * (1 - ratio)
        return marker_size, padding

    def detect(self, image: np.ndarray, detected_centers: np.ndarray,
               detected_aruco_ids: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects this crosshair in the image and computes its center.

        Parameters
        ----------
        image : np.ndarray
            The image in which to detect the crosshair.
        detected_centers : np.ndarray
            The centers of detected ArUco markers.
        detected_aruco_ids : np.ndarray
            The IDs of detected ArUco markers.

        Returns
        -------
        Optional[np.ndarray]
            The center of the crosshair, or None if the crosshair is not detected.
        """
        # Find indices of detected markers that match this crosshair's IDs
        ix_1 = np.where(detected_aruco_ids == self.id_1)[0]
        ix_2 = np.where(detected_aruco_ids == self.id_2)[0]
        if max(len(ix_1), len(ix_2)) > 1:
            return None  # Ensure only one instance of each marker is present

        try:
            # Extract and squeeze the centers of the detected markers
            centers = detected_centers[[ix_1[0], ix_2[0]]].squeeze()
        except IndexError:
            return None  # Handle cases where markers are not found

        # Compute the displacement and direction between the two markers
        displacement = centers[1] - centers[0]
        length = np.linalg.norm(displacement)
        direction = displacement / length
        factor = 0.5 - np.sum(self._get_dims(1)) / 4  # Compute adjustment factor based on marker dimensions

        # Calculate the points for refining the center estimate
        p1 = centers[0] + length * factor * direction
        p2 = centers[1] - length * factor * direction
        guess = 0.5 * (p1 + p2)

        # Determine window size for corner refinement based on the markers' distance
        window_size = int(np.round(np.linalg.norm(p1 - p2) / np.sqrt(2) / 2))
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        refined = cv2.cornerSubPix(image, guess[None], (window_size, window_size), (-1, -1), term)

        return refined.reshape(2)

    def draw(self, resolution: int = 400) -> np.ndarray:
        """
        Draws the crosshair with its ArUco markers at the specified resolution.

        Parameters
        ----------
        resolution : int, optional
            The resolution for the crosshair image, by default 400 pixels.

        Returns
        -------
        np.ndarray
            The generated crosshair image as a NumPy array.

        Raises
        ------
        ValueError
            If the resolution is not divisible by 10.
        """
        if resolution % 10 != 0:
            raise ValueError("Resolution must be divisible by 10")
        result = np.full((resolution, resolution), 255, dtype=np.uint8)  # Start with a white background

        half_res = resolution // 2
        # Create a basic pattern for visualization (not accurate for actual detection)
        result[:half_res, :half_res] = 0
        result[half_res:, half_res:] = 0

        marker_size, padding = self._get_dims(resolution)  # Calculate marker and padding sizes
        marker_size = int(marker_size)
        padding = int(padding)

        # Generate marker images and place them in the result image
        marker_1 = aruco.generateImageMarker(self.aruco_dict, self.id_1, marker_size)
        marker_2 = aruco.generateImageMarker(self.aruco_dict, self.id_2, marker_size)
        result[half_res + padding:, padding:][:marker_size, :marker_size] = marker_1
        result[padding:, half_res + padding:][:marker_size, :marker_size] = marker_2

        # Optional: Add marker ID text for easier identification (if space allows)
        if self.id_1 < 100:
            font_scale = int(resolution / 50)
            cv2.putText(result, str(self.id_1), (padding // 2, 2 * padding + half_res // 2), cv2.FONT_HERSHEY_PLAIN,
                        font_scale, (255, 255, 255), font_scale)

        return result
