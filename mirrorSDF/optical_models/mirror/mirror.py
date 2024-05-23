import json
from typing import List, Tuple

import numpy as np

from .calibration import fit_crosshair_locations
from .. import Lens, Camera
from ...aruco import AruCrosshairField, ArucoCalibrationBoard


class Mirror:
    """
    Represents a calibrated mirror object with properties derived from ArUco marker detections.

    Attributes
    ----------
    thickness_mm : float
        The thickness of the mirror in millimeters.
    ior : float
        The index of refraction of the mirror material. Default is 1.5, typical for glass.
    crosshair_ids : np.ndarray
        An array of IDs for the crosshairs detected on the mirror.
    crosshair_centers : np.ndarray
        An array containing the 3D coordinates of the centers of the crosshairs on the mirror.
    crosshair_corners : np.ndarray
        An array containing the 3D coordinates of the corners of the
        crosshairs, used for precise geometric calculations.
    mirror_corners_clockwise : np.ndarray
        An array of 3D coordinates defining the corners of the mirror, ordered clockwise.
    aruco_dict_id : int
        The dictionary ID of the ArUco markers used.
        This specifies which ArUco marker dictionary was used for the detection.

    Methods
    -------
    __init__(self, thickness_mm: float, crosshairs_ids: np.ndarray, crosshair_center: np.ndarray,
             crosshair_corners: np.ndarray, mirror_corners_clockwise: np.ndarray, aruco_dict_id: int, ior: float = 1.5):
        Initializes the Mirror object with the given parameters.
    """

    def __init__(self, thickness_mm: float,
                 crosshairs_ids: np.ndarray,
                 crosshair_center: np.ndarray,
                 crosshair_corners: np.ndarray,
                 mirror_corners_clockwise: np.ndarray,
                 aruco_dict_id: int,
                 ior: float = 1.5):

        self.thickness_mm = thickness_mm
        self.ior = ior
        self.crosshair_ids = crosshairs_ids
        self.crosshair_centers = crosshair_center
        self.crosshair_corners = crosshair_corners
        self.mirror_corners_clockwise = mirror_corners_clockwise
        self.aruco_dict_id = aruco_dict_id
        
    def to_disk(self, filepath: str):
        """
        Serialize the Mirror object to a JSON file.

        Parameters
        ----------
        filepath : str
            The path to the file where the object's data will be saved.

        """
        data = {
            'thickness_mm': self.thickness_mm,
            'ior': self.ior,
            'crosshair_ids': self.crosshair_ids.tolist(),  # Convert np.ndarray to list
            'crosshair_centers': self.crosshair_centers.tolist(),
            'crosshair_corners': self.crosshair_corners.tolist(),
            'mirror_corners_clockwise': self.mirror_corners_clockwise.tolist(),
            'aruco_dict_id': self.aruco_dict_id
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @property
    def crosshair_field(self) -> AruCrosshairField:
        """
        Construct an AruCrosshairField to detect the markers present on this mirror

        Returns
        -------
        field: AruCrosshairField
            The field to detect the markers on this mirror

        """
        lowest_id = self.crosshair_ids.min()
        largest_id = self.crosshair_ids.max()
        field = AruCrosshairField((largest_id - lowest_id) // 2, lowest_id, self.aruco_dict_id)
        return field

    @classmethod
    def from_disk(cls, filepath: str):
        """
        Deserialize a JSON file to a Mirror object.

        Parameters
        ----------
        filepath : str
            The path to the file from which the object's data will be loaded.

        Returns
        -------
        instance : Mirror
            A Mirror instance with properties loaded from the file.

        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            thickness_mm=float(data['thickness_mm']),
            crosshairs_ids=np.array(data['crosshair_ids']),
            crosshair_center=np.array(data['crosshair_centers']),
            crosshair_corners=np.array(data['crosshair_corners']),
            mirror_corners_clockwise=np.array(data['mirror_corners_clockwise']),
            aruco_dict_id=data['aruco_dict_id'],
            ior=float(data['ior'])
        )

    @classmethod
    def calibrate_from_measurements(cls, image_filenames: List[str], field: AruCrosshairField,
                                    board: ArucoCalibrationBoard, lens: Lens,
                                    checkerboard_tile_width_mm: float,
                                    thickness_mm: float, ior: float,
                                    marker_corners_id_clockwise: List[int],
                                    center_defining_marker_ids: List[int], marker_paper_thickness_mm: float
                                    ) -> Tuple['Mirror', List[Camera], np.ndarray, np.ndarray]:
        """
        Calibrates the mirror setup from measurements obtained from images.

        Parameters
        ----------
        image_filenames : List[str]
            A list of image filenames used for calibration.
        field : AruCrosshairField
            The field object responsible for detecting crosshairs.
        board : ArucoCalibrationBoard
            The calibration board object used for initial pose estimation.
        lens : Lens
            The lens object used for image undistortion.
        checkerboard_tile_width_mm : float
            The width of a single checkerboard tile in millimeters. (from the ArucoCalibrationBoard)
        thickness_mm : float
            The thickness of the mirror in millimeters.
        ior : float
            The index of refraction for the mirror material.
        marker_corners_id_clockwise : List[int]
            A list of marker IDs defining the corners of the mirror, ordered clockwise.
        center_defining_marker_ids : List[int]
            The IDs of the markers used to define the center of the mirror. The center will be the midpoint.
        marker_paper_thickness_mm : float
            The thickness of the paper on which the markers are printed, in millimeters.

        Returns
        -------
        Tuple[Mirror, List[Camera], np.ndarray, np.ndarray]
            A tuple containing:
            - The calibrated Mirror object.
            - A list of Camera objects with their poses.
            - RMSE by image: The root-mean-square error of the calibration by image.
            - RMSE by marker: The root-mean-square error of the calibration by marker.
        """
        # Fit crosshair locations from the images
        (crosshair_ids, crosshair_3d_locations, cameras_poses, crosshair_corners,
         rmse_by_image, rmse_by_marker) = fit_crosshair_locations(image_filenames, field, board,
                                                                  lens, checkerboard_tile_width_mm)
        
        # Calculate the offset from the center-defining markers
        def coords_by_index(index):
            return crosshair_3d_locations[np.where(crosshair_ids == index)[0][0]]

        m1 = coords_by_index(center_defining_marker_ids[0])
        m2 = coords_by_index(center_defining_marker_ids[1])

        # Calculate the coordinates for the mirror corners
        mirror_corner_coords = np.array([coords_by_index(x) for x in marker_corners_id_clockwise])
        offset = - (m1 + m2) / 2
        offset_paper = np.array([0, 0, marker_paper_thickness_mm])

        # Apply the offset to the crosshair locations and corners
        offset_crosshair_3d_locations = crosshair_3d_locations + offset + offset_paper
        offset_crosshair_corners = crosshair_corners + offset[None, None] + offset_paper[None, None]
        offset_mirror_corner_coords = mirror_corner_coords + offset

        # Create the Mirror object with the calculated parameters
        mirror = cls(
            thickness_mm,
            crosshair_ids,
            offset_crosshair_3d_locations,
            offset_crosshair_corners,
            offset_mirror_corner_coords,
            field.aruco_dict_id,
            ior
        )

        return mirror, cameras_poses, rmse_by_image, rmse_by_marker

    def estimate_camera_pose(self, pre_processed_image: np.ndarray, lens: Lens,
                             disable_distortions: bool = False) -> Tuple[Camera, float]:
        """
        Estimates the camera pose by matching 2D coordinates of detected crosshairs in an image
        with their corresponding 3D world coordinates and using these matches to recover the
        camera's position and orientation.

        Parameters
        ----------
        pre_processed_image : np.ndarray
            The pre-processed image where crosshairs are detected.
        lens : Lens
            The lens object used for the projection, which includes intrinsic
            parameters and optionally distortion coefficients.
        disable_distortions : bool, optional
            If True, ignores the lens distortions. Default is False.
            Useful if the images has been undistorted.

        Returns
        -------
        Camera
            An instance of the Camera class with estimated rotation and translation vectors.
        float
            The reprojection error, quantifying the discrepancy between the projected points using
            the estimated camera parameters and the original 2D coordinates in the image.
        """
        crosshairs_coords_2d, crosshair_ids, _ = self.crosshair_field.detect_crosshairs(pre_processed_image)
        coords_2d = dict(zip(crosshair_ids, crosshairs_coords_2d))  # Map detected 2D crosshair IDs to their coordinates
        coords_3d = dict(
            zip(self.crosshair_ids, self.crosshair_centers))  # Map predefined 3D crosshair IDs to their coordinates
        in_common = set(coords_2d) & set(coords_3d)  # Find crosshair IDs present in both 2D and 3D mappings

        selected_2d = np.zeros((len(in_common), 2))  # Array to hold 2D coordinates for matching IDs
        selected_3d = np.zeros((len(in_common), 3))  # Array to hold 3D coordinates for matching IDs

        for i, current_id in enumerate(in_common):  # Populate arrays with coordinates from matching IDs
            selected_2d[i] = coords_2d[current_id]
            selected_3d[i] = coords_3d[current_id]

        camera, reprojection_error = Camera.recover_from_coordinates(selected_2d, selected_3d,
                                                                     lens, disable_distortions)  # Estimate camera pose
        return camera, reprojection_error
