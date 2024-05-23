from typing import Tuple

from cv2 import aruco
import cv2
import numpy as np

from ..utils import compute_square_centers
from .crosshair import AruCrosshair


class AruCrosshairField:
    """
    A class to create and manage a field of ArUco marker crosshairs.

    Parameters
    ----------
    num_crosshairs : int
        The number of crosshair pairs to generate.
    lowest_marker_id : int, optional
        The starting ID for the ArUco markers, by default 0.
    aruco_dict_id
        The dictionary ID of the ArUco markers, by default aruco.DICT_APRILTAG_36h11.

    Attributes
    ----------
    aruco_dict_id
        The dictionary ID of the ArUco markers used.
    aruco_dict : cv2.aruco.Dictionary
        The ArUco dictionary object for marker generation.
    num_crosshairs : int
        The total number of crosshair pairs.
    cross_hairs : list[AruCrosshair]
        A list of `AruCrosshair` objects representing each crosshair pair.
    """

    def __init__(self, num_crosshairs, lowest_marker_id: int = 0, aruco_dict_id=aruco.DICT_APRILTAG_36h11):
        # Initialize the ArUco dictionary based on provided ID
        self.aruco_dict_id = aruco_dict_id
        self.aruco_dict = aruco.getPredefinedDictionary(self.aruco_dict_id)

        self.num_crosshairs = num_crosshairs
        self.cross_hairs = []
        # Generate crosshair objects with consecutive marker IDs, skipping one to form pairs
        for i in range(lowest_marker_id, lowest_marker_id + 2 * num_crosshairs, 2):
            self.cross_hairs.append(AruCrosshair(i, i + 1, self.aruco_dict))

    def draw(self, num_markers_wide: int = 8, crosshair_res: int = 400, padding: int = 20) -> np.ndarray:
        """
        Draws the complete field of crosshairs as a single image.

        Parameters
        ----------
        num_markers_wide : int, optional
            The number of crosshairs to draw horizontally, by default 8.
        crosshair_res : int, optional
            The resolution of each crosshair, by default 400 pixels.
        padding : int, optional
            The padding around each crosshair, by default 20 pixels.

        Returns
        -------
        np.ndarray
            The generated image of the crosshair field as a NumPy array.
        """
        w = crosshair_res + padding  # Calculate total width of each crosshair including padding
        num_markers_height = int(np.ceil(self.num_crosshairs / num_markers_wide))  # Calculate the number of rows needed

        # Initialize the image with white background
        final_image = np.full((num_markers_height * w, num_markers_wide * w), 255, dtype=np.uint8)

        # Place each crosshair on the image
        for i, crosshair in enumerate(self.cross_hairs):
            x = i % num_markers_wide  # Calculate x position based on index
            y = i // num_markers_wide  # Calculate y position based on index

            # Insert the crosshair image into the final image
            final_image[y * w:, x * w:][:w - padding, :w - padding] = crosshair.draw(w - padding)

        return final_image

    def detect_marker_centers(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects the centers of all ArUco markers in the given image.

        Parameters
        ----------
        image : np.ndarray
            The image in which to detect the markers.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing an array of center coordinates and an array of corresponding marker IDs.
        """
        # Detect markers in the image
        corners, marker_ids, _ = aruco.detectMarkers(image, self.aruco_dict)
        corners = np.array(corners)[:, 0]  # Simplify corners array shape
        marker_ids = marker_ids.ravel()  # Flatten marker IDs array

        # Compute centers of detected markers
        corner_centers = compute_square_centers(corners)

        # Filter out invalid or duplicate markers based on predefined IDs and count
        all_ids = np.array([[x.id_1, x.id_2] for x in self.cross_hairs]).ravel()
        counts = np.bincount(marker_ids)
        valid = np.isin(marker_ids, all_ids) & (counts[marker_ids] == 1)

        # Return valid centers and their IDs
        corner_centers = corner_centers[valid]
        marker_ids = marker_ids[valid]
        return corner_centers, marker_ids

    def detect_crosshairs(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detects and computes the central coordinates of crosshairs in the image, their corresponding marker IDs,
        and the coordinates of the markers forming the crosshairs.

        Parameters
        ----------
        image : np.ndarray
            The image in which to detect crosshairs.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing three numpy arrays:
            - The first array contains the coordinates of detected crosshairs.
            - The second array contains the IDs of markers associated with each detected crosshair.
            - The third array contains the coordinates of the markers that form each detected crosshair.

        Notes
        -----
        This method first detects marker centers and their IDs within the given image.
        It then identifies crosshairs based on predefined criteria, calculating the central
        coordinate for each detected crosshair and recording the coordinates
        of the markers forming these crosshairs.
        """
        # Detect marker centers and IDs first
        marker_centers, marker_ids = self.detect_marker_centers(image)
        cross_hair_ids = []
        cross_hair_coordinates = []
        cross_hair_fiducial_centers = []

        # For each crosshair, detect its presence and calculate its center
        for crosshair in self.cross_hairs:
            current_coordinate = crosshair.detect(image, marker_centers, marker_ids)
            if current_coordinate is not None:
                cross_hair_ids.append(crosshair.id_1)
                cross_hair_coordinates.append(current_coordinate)
                cross_hair_fiducial_centers.append(np.stack([
                    marker_centers[np.where(marker_ids == crosshair.id_1)[0][0]],
                    marker_centers[np.where(marker_ids == crosshair.id_2)[0][0]],
                ]))

        # Convert lists to numpy arrays for consistent API
        cross_hair_ids = np.array(cross_hair_ids)
        cross_hair_coordinates = np.stack(cross_hair_coordinates) if cross_hair_coordinates else np.array([])
        cross_hair_fiducial_centers = (np.array(cross_hair_fiducial_centers) if cross_hair_fiducial_centers
                                       else np.array([]))
        return cross_hair_coordinates, cross_hair_ids, cross_hair_fiducial_centers
