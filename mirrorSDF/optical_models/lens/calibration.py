from typing import List, Tuple

import cv2
import numpy as np
from tqdm.autonotebook import tqdm

from mirrorSDF.aruco import ArucoCalibrationBoard, prepare_image_for_aruco
from mirrorSDF.utils.image import imread


def calibrate_lens(image_filenames: List[str],
                   board: ArucoCalibrationBoard,
                   pixel_size_mm: float,
                   image_clip_percentiles: Tuple[int, int] = (15, 85)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibrate camera lens using images of an ArUco calibration board.

    This function processes a series of images containing an ArUco calibration board, detects
    the markers, and uses them to calibrate the camera lens. The calibration process determines
    the camera matrix and distortion coefficients which can be used to correct lens distortion
    in other images captured with the same camera at the same focus distance.

    Parameters
    ----------
    image_filenames : List[str]
        A list of paths to the images used for calibration.
    board : ArucoCalibrationBoard
        An instance of the ArucoCalibrationBoard class that provides functionalities to detect
        ArUco markers and generate corresponding 3D points for calibration.
    pixel_size_mm : float
        The size of one pixel in millimeters, used to know the size of the board in the physical world
    image_clip_percentiles : Tuple[int, int], optional
        Percentiles used for clipping the image intensities before ArUco detection. Defaults to [15, 85].

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the camera matrix, distortion coefficients, and re-projection errors
        as NumPy arrays.
    """
    all_coords = []
    all_targets = []

    if not image_filenames:
        raise ValueError("No Image given")
    adjusted_image = np.zeros((0, 0))

    for file_name in tqdm(image_filenames, desc='loading and analyzing images'):
        image, _ = imread(file_name)
        adjusted_image = prepare_image_for_aruco(image, image_clip_percentiles)
        coords, corner_ids, targets = board.detect(adjusted_image,
                                                   pixel_size_mm=pixel_size_mm)
        all_coords.append(coords)
        all_targets.append(targets)

    all_targets = [x.reshape(-1, 3).astype(np.float32) for x in all_targets]
    all_coords = [x.reshape(-1, 2) for x in all_coords]

    with tqdm(total=None, desc='Fitting parameters'):

        _, cam, dist, r_vecs, t_vecs = cv2.calibrateCamera(all_targets, all_coords,
                                                           adjusted_image.shape[:2], None, None)
        re_projection_errors = []
        for i in range(len(all_targets)):
            reprojected = cv2.projectPoints(all_targets[i], r_vecs[i], t_vecs[i], cam, dist)[0][:, 0]
            re_projection_errors.append(np.linalg.norm(reprojected - all_coords[i], axis=-1).mean())

        re_projection_errors = np.array(re_projection_errors)

    return cam, dist, re_projection_errors


def visualize_lens_distortion(camera_matrix: np.ndarray, distortion_coefs: np.ndarray) -> np.ndarray:
    """
    Visualize the effect of lens distortion based on the camera matrix and distortion coefficients.

    This function generates a grid of points in the normalized camera coordinates, applies the
    distortion based on the provided coefficients, and projects the points using the camera matrix.
    It's useful for understanding how distortion affects imaging across the field of view.

    Parameters
    ----------
    camera_matrix : np.ndarray
        The camera matrix obtained from lens calibration. Shape: (3, 3)
    distortion_coefs : np.ndarray
        The distortion coefficients obtained from lens calibration.

    Returns
    -------
    np.ndarray
        The projected points after applying lens distortion, useful for visualization purposes.
    """

    xes = np.linspace(-1, 1, 50)
    points = np.stack(np.meshgrid(xes, xes), 2)
    scale = np.abs(np.linalg.inv(camera_matrix) @ np.array([0, 0, 1]))[:2]
    points = points.reshape(-1, 2) * scale[None, :]
    points = np.pad(points, ((0, 0), (0, 1)), constant_values=0)
    projected_points = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                         np.eye(3), distortion_coefs)[0]
    return projected_points
