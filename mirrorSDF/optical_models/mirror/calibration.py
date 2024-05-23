import sys
from typing import List, Tuple
from collections import defaultdict

import cv2
import numpy as np
from tqdm.autonotebook import tqdm
from scipy.optimize import least_squares

from ...aruco import (AruCrosshairField,
                      ArucoCalibrationBoard,
                      prepare_image_for_aruco)
from ..lens import Lens
from ..camera import Camera
from ...utils.image import imread


def fit_crosshair_locations(image_filenames: List[str], field: AruCrosshairField,
                            bootstrap_board: ArucoCalibrationBoard, lens: Lens,
                            marker_size_mm: float) -> Tuple[np.ndarray, np.ndarray,
                            List[Camera], np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits crosshair locations across multiple images, optimizing their 3D positions and camera poses.

    Parameters
    ----------
    image_filenames : List[str]
        List of filenames for the images to process.
    field : AruCrosshairField
        The field object responsible for detecting crosshairs.
    bootstrap_board : ArucoCalibrationBoard
        The calibration board object used for initial camera pose estimation.
    lens : Lens
        The lens object used for image undistortion and camera parameter recovery.
    marker_size_mm : float
        The physical width of a checkerboard tile on the bootstrapping pattern.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[Camera], np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - detected_markers: Array of marker IDs that were detected and optimized.
        - optimized_centers: Optimized 3D locations of the crosshair centers.
        - optimized_cameras: List of Camera objects with optimized poses.
        - crosshair_corners: 3D locations of the estimated corners of each crosshair.
        - errors_by_image: RMS error of the reprojection by image.
        - errors_by_marker: RMS error of the reprojection by marker.
    """
    # Process images to detect markers and crosshairs, and recover camera poses
    data = process_images(bootstrap_board, field, image_filenames, lens, marker_size_mm)

    all_cameras, all_crosshair_coords, all_crosshair_ids, all_crosshair_marker_centers = data

    num_images = len(image_filenames)
    # Extract relevant markers based on the field setup
    relevant_markers = np.array([x.id_1 for x in field.cross_hairs])
    num_fiducials = max(relevant_markers) + 1

    # Initialize matrices for optimization
    marker_present = np.zeros((num_images, num_fiducials))
    guess_3d = np.zeros((num_fiducials, 3))
    all_poses = np.zeros((num_images, 2, 3))
    targets_matrix = np.zeros((num_images, num_fiducials, 2))

    # Prepare data for optimization
    for i, (camera, coords, ids) in enumerate(zip(all_cameras, all_crosshair_coords, all_crosshair_ids)):
        current_guess = camera.pixel_to_plane(coords, disable_distortions=True)
        all_poses[i, 0] = camera.r_vec
        all_poses[i, 1] = camera.t_vec

        for cx, cid, tc in zip(current_guess, ids, coords):
            targets_matrix[i, cid] = tc
            marker_present[i, cid] = 1
            guess_3d[cid] += cx

    num_cross_hair_visible = marker_present.sum(0)
    detected_markers = np.where(num_cross_hair_visible)[0]
    # Warn about missing markers
    warn_user_missing_markers(num_cross_hair_visible, relevant_markers)

    # Average the initial guess for crosshair locations
    guess_3d /= np.maximum(num_cross_hair_visible[:, None], 1)
    guess_3d = np.nan_to_num(guess_3d, 0.0)

    # Fine-tune guess for crosshair locations and camera poses
    optimized_centers, optimized_poses, residuals = fine_tune_guess(all_poses, guess_3d,
                                                                    num_fiducials, num_images,
                                                                    targets_matrix,
                                                                    marker_present, lens)
    
    # Filter the results for detected markers
    optimized_centers = optimized_centers[detected_markers]
    errors_by_marker = ((residuals ** 2).sum(-1).sum(0) / np.maximum(1, marker_present.sum(0))) ** 0.5
    errors_by_marker = errors_by_marker[detected_markers]
    errors_by_image = ((residuals ** 2).sum(-1).sum(1) / marker_present.sum(1)) ** 0.5

    # Create Camera objects with optimized poses
    optimized_cameras = [Camera(pose[0], pose[1], lens) for pose in optimized_poses]

    # Estimate crosshair corners based on optimized camera poses and centers
    crosshair_corners = estimate_crosshair_corners(
        optimized_cameras,
        detected_markers,
        optimized_centers,
        all_crosshair_ids,
        all_crosshair_marker_centers
    )

    return (detected_markers, optimized_centers, optimized_cameras,
            crosshair_corners, errors_by_image, errors_by_marker)


def warn_user_missing_markers(num_cross_hair_visible: dict, relevant_markers: np.ndarray):
    """
    Warns the user if any relevant crosshairs are not visible in the processed images.

    This function iterates through a list of relevant crosshair IDs and checks if they have been visible in any
    of the processed images. If a crosshair has not been visible at all (i.e., it has a visibility count of 0),
    a warning message is printed to stderr.

    Parameters
    ----------
    num_cross_hair_visible : dict
        A dictionary mapping crosshair IDs to their visibility count across all processed images.
    relevant_markers : np.ndarray
        A list of crosshair IDs that are relevant for the current analysis or processing task.
    """

    for cid in relevant_markers:
        if num_cross_hair_visible[cid] == 0:
            print(f"Warning: Crosshair {cid} not found, if it's on your mirror, make sure you take pictures of it!!",
                  file=sys.stderr)


def process_images(bootstrap_board: ArucoCalibrationBoard, field: AruCrosshairField,
                   image_filenames: List[str], lens: Lens,
                   marker_size_mm: float) -> Tuple[List[Camera], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Processes a list of image filenames to detect ArUco markers and crosshairs,
    recovering camera parameters and extracting relevant coordinates.

    Parameters
    ----------
    bootstrap_board : ArucoCalibrationBoard
        The bootstrap board that was printed for the purpose of calibration
    field : AruCrosshairField
        The field that was used to generate the crosshairs
    image_filenames : List[str]
        A list of filenames for the images to be processed.
    lens : object
        The lens object, which must be capable of undistorting images based on its distortion parameters.
    marker_size_mm : float
        The size of the markers in millimeters, used for scale in detection.

    Returns
    -------
    Tuple[List[Camera], List[np.ndarray], List[np.ndarray], List[np.ndarray]]
        A tuple containing:
        - A list of Camera objects recovered from each image.
        - A list of 2D numpy arrays with the coordinates of the detected crosshairs in each image.
        - A list of numpy arrays with the IDs of the detected crosshairs in each image.
        - A list of 2D numpy arrays with the centers of markers associated with each detected crosshair in each image.
    """
    all_cameras = []
    all_crosshair_coords = []
    all_crosshair_ids = []
    all_marker_centers = []
    all_crosshair_marker_centers = []

    for image_filename in tqdm(image_filenames, desc="Load and process images"):
        image, _ = imread(image_filename)  # Load image
        processed = prepare_image_for_aruco(image, percentiles=(1, 99))  # Pre-process image for ArUco detection
        processed = lens.undistort(processed)  # Undistort image based on lens parameters

        # Detect ArUco markers to use as bootstrap for camera recovery
        corner_coords, corner_ids, targets = bootstrap_board.detect(processed, marker_size_mm=marker_size_mm)

        # Adjust targets for camera coordinate system
        targets[:, 0] *= -1  # Flip x-coordinate

        # Recover camera parameters from detected corners and known targets
        camera, _ = Camera.recover_from_coordinates(corner_coords, targets, lens, disable_distortions=True)

        # Detect markers in the field
        all_marker_centers.append(field.detect_marker_centers(processed))

        # Detect crosshairs in the field
        crosshair_coords, crosshair_ids, crosshair_marker_centers = field.detect_crosshairs(processed)

        # Collect results
        all_cameras.append(camera)
        all_crosshair_coords.append(crosshair_coords)
        all_crosshair_ids.append(crosshair_ids)
        all_crosshair_marker_centers.append(crosshair_marker_centers)

    return all_cameras, all_crosshair_coords, all_crosshair_ids, all_crosshair_marker_centers


def estimate_crosshair_corners(cameras: List, crosshair_ids: np.ndarray, crosshairs_3d_location: np.ndarray,
                               aruco_ids_by_image: List[np.ndarray],
                               aruco_centers_by_image: List[np.ndarray]) -> np.ndarray:
    """
    Estimates the corners of crosshairs in 3D space based on the locations of the AruCo markers that they consist of.

    Parameters
    ----------
    cameras : List
        A list of Camera objects, one for each picture taken
    crosshair_ids : np.ndarray
        An array of unique identifiers for each crosshair whose corners are to be estimated.
    crosshairs_3d_location : np.ndarray
        The 3D locations of the center of each crosshair.
    aruco_ids_by_image : List[np.ndarray]
        A list of arrays, each containing the IDs of the ArUco markers detected in a single image.
    aruco_centers_by_image : List[np.ndarray]
        A list of arrays, each containing the pixel coordinates of
        the centers of the ArUco markers detected in a single image.

    Returns
    -------
    np.ndarray
        An array containing the estimated 3D locations of the corners of each crosshair. The shape of the array is
        (len(crosshair_ids), 4, 3), corresponding to the number of crosshairs, four corners per crosshair, and three
        coordinates (x, y, z) per corner.

    """
    center_estimates = defaultdict(list)
    for cam, ids, locations in zip(cameras, aruco_ids_by_image, aruco_centers_by_image):
        # Convert pixel coordinates to 3D points using the camera model
        test = cam.pixel_to_plane(locations.reshape(-1, 2))
        test = test.reshape(-1, 2, 3)  # Reshape for processing
        for cid, c3d in zip(ids, test):
            center_estimates[cid].append(c3d)  # Group 3D points by their ArUco marker ID

    cross_hair_corners = np.zeros((len(crosshair_ids), 4, 3))  # Initialize array for corner coordinates

    for i, (cid, middle) in enumerate(zip(crosshair_ids, crosshairs_3d_location)):
        centers = center_estimates[cid]  # Get 3D points associated with current crosshair ID
        direction = np.diff(np.stack(centers, 0).mean(0), 1, axis=0)[0]  # Calculate average direction vector
        a, b, _ = direction  # Decompose direction vector
        # Calculate outside corners based on direction and add to the middle position
        outside_corners = np.array([
            [a, b, 0],
            [b, -a, 0],
            [-a, -b, 0],
            [-b, a, 0]
        ]) + middle
        cross_hair_corners[i] = outside_corners  # Assign calculated corners to the output array

    return cross_hair_corners


def fine_tune_guess(all_poses: np.ndarray, guess_3d: np.ndarray, num_fiducials: int, num_images: int, targets_matrix: np.ndarray,
                    marker_present: np.ndarray, lens) -> tuple:
    """
    Fine-tunes the initial guess of fiducial marker locations and
    camera poses using non-linear least squares optimization.

    Parameters
    ----------
    all_poses : np.ndarray
        The initial guesses for the camera poses for each image, shaped (num_images, 2, 3).
    guess_3d : np.ndarray
        The initial guesses for the 3D locations of the fiducial markers, shaped (num_fiducials, 3).
    num_fiducials : int
        The total number of fiducial markers.
    num_images : int
        The total number of images.
    targets_matrix : np.ndarray
        The target 2D locations of the fiducial markers in the image plane, shaped (num_images, num_fiducials, 2).
    marker_present : np.ndarray
        A binary matrix indicating the presence (1) or absence (0) of a fiducial marker in each image, shaped (num_images, num_fiducials).
    lens
        The lens object, which must have distortion parameters and an intrinsic matrix for projection.

    Returns
    -------
    tuple
        A tuple containing the optimized 3D locations of fiducial markers (with added zero z-coordinates),
        optimized camera poses, and residuals of the optimization process.

    Raises
    ------
    ValueError
        If the optimization process fails, indicating potential issues with the provided images or initial guesses.
    """
    with tqdm(desc='Optimizing locations', total=None) as progress:
        def split_params(params: np.ndarray) -> tuple:
            """
            Splits the flattened parameter array back into separate arrays for the fiducial centers and camera poses.

            Parameters
            ----------
            params : np.ndarray
                A flattened array of all parameters being optimized.

            Returns
            -------
            tuple
                A tuple containing the reshaped fiducial centers (with a padded zero z-coordinate) and camera poses.
            """
            split_at = 2 * num_fiducials
            fiducials_centers = params[:split_at].reshape(-1, 2)
            fiducials_centers = np.pad(fiducials_centers, ((0, 0), (0, 1)), 'constant', constant_values=0)
            poses = params[split_at:].reshape(-1, 2, 3)
            return fiducials_centers, poses

        def loss(params: np.ndarray) -> np.ndarray:
            """
            Calculates the loss (residuals) for the current parameter estimates.

            Parameters
            ----------
            params : np.ndarray
                The current estimates for the parameters being optimized.

            Returns
            -------
            np.ndarray
                The flattened array of residuals for each marker in each image.
            """
            progress.update(1)
            fiducials_centers, poses = split_params(params)
            reprojected = project_points_many_cam(fiducials_centers, poses, lens, disable_distortion=True)
            current_residuals = (reprojected - targets_matrix) * marker_present[..., None]
            return current_residuals.ravel()

        # Concatenate the initial guesses for fiducial centers and camera poses into a single flat array
        x0 = np.concatenate([guess_3d[:, :2].ravel(), all_poses.ravel()])
        problem = least_squares(loss, x0)

        if not problem.success:
            raise ValueError("Optimization failed, check your images")

        optimized_centers, optimized_poses = split_params(problem.x)
        residuals = loss(problem.x).reshape(num_images, num_fiducials, 2)

    return optimized_centers, optimized_poses, residuals


def project_points_many_cam(points_3d: np.ndarray, poses: np.ndarray,
                            lens: Lens, disable_distortion: bool = False) -> np.ndarray:
    """
    Projects 3D points onto multiple camera image planes given their poses and a common lens.

    Parameters
    ----------
    points_3d : np.ndarray
        The 3D points to be projected, of shape (n_points, 3), where n_points is the number of points.
    poses : np.ndarray
        The poses of the cameras, of shape (n_scenes, 2, 3), where n_scenes is the number of camera scenes,
        the first dimension within 2 corresponds to rotation vectors and the second to translation vectors.
    lens : Lens
        The lens object associated with the cameras, containing intrinsic parameters and distortions.
    disable_distortion : bool, optional
        If True, lens distortions are ignored during the projection process. Default is False.

    Returns
    -------
    np.ndarray
        The projected 2D points on the camera image planes, of shape (n_scenes, n_points, 2).

    Notes
    -----
    - The function utilizes the camera poses (rotation and translation vectors) to first transform the 3D points
      into each camera's coordinate system.
    - It then projects these points onto the 2D image plane using the camera intrinsic parameters and optional distortion.
    """
    distortion = None if disable_distortion else lens.distortions
    zero = np.zeros(3, dtype=np.float32)  # Zero rotation and translation for projectPoints since transformation is pre-applied
    n_scenes = poses.shape[0]
    n_points = points_3d.shape[0]

    r_vecs = poses[:, 0, :]
    t_vecs = poses[:, 1, :]

    # Pre-compute rotation matrices from rotation vectors for all scenes
    rot_matrices = np.array([cv2.Rodrigues(r_vec)[0] for r_vec in r_vecs])

    # Apply extrinsic transformations (rotation and translation) to 3D points
    after_extrinsic = np.einsum('sij,mj->smi', rot_matrices, points_3d) + t_vecs[:, None, :]

    # Project the transformed 3D points to 2D using the camera's intrinsic parameters and optional distortion
    projected = cv2.projectPoints(after_extrinsic.reshape(-1, 1, 3), zero, zero, lens.intrinsic, distortion)[0]

    # Reshape the projected points to separate scenes and points
    projected = projected.reshape(n_scenes, n_points, 2)

    return projected
