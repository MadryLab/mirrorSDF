from typing import Tuple
import numpy as np
import cv2

from ..lens import Lens


class Camera:
    """
    A class representing a camera with its rotation vector, translation vector, and associated lens.

    Attributes
    ----------
    r_vec : np.ndarray
        The rotation vector of the camera.
    t_vec : np.ndarray
        The translation vector of the camera.
    lens : Lens
        The lens object associated with the camera, containing intrinsic parameters and distortions.
    """

    def __init__(self, r_vec: np.ndarray, t_vec: np.ndarray, lens: Lens):
        """
        Initializes the Camera object with rotation vector, translation vector, and lens.

        Parameters
        ----------
        r_vec : np.ndarray
            The rotation vector of the camera. Shape: (3, )
        t_vec : np.ndarray
            The translation vector of the camera. Shape: (3, )
        lens : Lens
            The lens object associated with the camera.
        """
        self.r_vec = r_vec
        self.t_vec = t_vec
        self.lens = lens

    @staticmethod
    def recover_from_coordinates(coords: np.ndarray, targets: np.ndarray, lens: Lens,
                                 disable_distortions: bool = False) -> Tuple['Camera', float]:
        """
        Recover the camera's rotation and translation vectors from given coordinates
        and targets, also returning the mean re-projection error.

        Parameters
        ----------
        coords : np.ndarray
            2D coordinates in the image plane. Shape (N, 2).
        targets : np.ndarray
            Corresponding 3D target points in the world coordinates. Shape (N, 3).
        lens : Lens
            The lens object used for projection.
        disable_distortions : bool, optional
            If True, ignores the lens distortions. Default is False.
            Useful if the images have been undistorted.

        Returns
        -------
        Camera
            A new Camera instance with the recovered rotation and translation vectors.
        float
            The mean re-projection error, quantifying the discrepancy between the projected points
            using the recovered camera parameters and the original 2D coordinates.

        """
        distortions = lens.distortions if not disable_distortions else None
        _, r_vec, t_vec = cv2.solvePnP(targets.astype(np.float32), coords.astype(np.float32),
                                       lens.intrinsic, distortions)
        r_vec = r_vec.ravel().astype(np.float32)  # Flatten the rotation vector for ease of use
        t_vec = t_vec.ravel().astype(np.float32)  # Flatten the translation vector for ease of use
        camera = Camera(r_vec, t_vec, lens)
        reprojected = camera.project(targets)  # Project the 3D points back to 2D using the recovered camera parameters
        error = np.linalg.norm(reprojected - coords, axis=-1).mean()  # Calculate the mean re-projection error
        return camera, error

    def project(self, points_world: np.ndarray, disable_distortions: bool = False) -> np.ndarray:
        """
        Project 3D points in world coordinates to 2D points in the image plane.

        Parameters
        ----------
        points_world : np.ndarray
            3D points in world coordinates. Shape: (N, 3)
        disable_distortions : bool, optional
            If True, ignores the lens distortions. Default is False.
            Useful if the images has been undistorted.

        Returns
        -------
        np.ndarray
            2D points in the image plane.

        Raises
        ------
        AssertionError
            If the shape of `points_world` is not 2D or does not have three columns.
        """
        assert len(points_world.shape) == 2 and points_world.shape[1] == 3, "Input points must be in Nx3 format."
        distortions = self.lens.distortions if not disable_distortions else None
        return cv2.projectPoints(points_world, self.r_vec, self.t_vec,
                                 self.lens.intrinsic, distortions)[0][:, 0]

    def pixel_to_plane(self, points_2d: np.ndarray, disable_distortions: bool = False) -> np.ndarray:
        """
        Projects back points in the camera image plane (in pixel coordinates) into 3D under the constraint that
        their lie on the Z=0 plane.

        Parameters
        ----------
        points_2d : np.ndarray
            2D points in the image plane.
        disable_distortions : bool, optional
            If True, ignores the lens distortions. Default is False.

        Returns
        -------
        np.ndarray
            3D points in world coordinates laying ont the Z=0 plane.
        """
        distortions = self.lens.distortions if not disable_distortions else None
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = cv2.Rodrigues(self.r_vec)[0]  # Convert rotation vector to rotation matrix
        extrinsic[:3, 3] = self.t_vec

        camera_to_world = np.linalg.inv(extrinsic)  # Inverse the matrix to get the camera to world transformation

        points_2d = points_2d.astype(np.float64)
        points_on_camera_plane = cv2.undistortPoints(points_2d, self.lens.intrinsic, distortions, None)[:, 0]

        origin_point = np.array([0, 0, 0, 1.0])  # Define the camera origin in homogenous coordinates
        origin = (camera_to_world @ origin_point)[:3]  # Calculate the world coordinates of the camera origin

        # Calculate directions in camera space, adding a z=1 for depth and a
        # placeholder for w in homogeneous coordinates
        directions_cam_space = np.concatenate([
            points_on_camera_plane,
            np.ones((points_on_camera_plane.shape[0], 1)),
            np.zeros((points_on_camera_plane.shape[0], 1)),
        ], axis=1)

        # Convert directions to world space
        directions_world_space = (camera_to_world @ directions_cam_space.T).T[:, :3]

        # Calculate the scale factor to reach the plane z=0 in world space
        t = -origin[2] / directions_world_space[:, 2]

        # Scale and translate the directions to get final coordinates
        final_coords = origin + directions_world_space * t[:, None]

        return final_coords

    def generate_rays_from_pixels(self, pixel_coords: np.ndarray,
                                  disable_distortions: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates rays from camera origin through pixel coordinates in world space.

        This method computes the origin and direction vectors of rays passing through
        specified pixel coordinates. It accounts for lens distortions unless explicitly
        disabled.

        Parameters
        ----------
        pixel_coords : np.ndarray
            A 2D numpy array of shape (N, 2), where each row represents the (x, y)
            coordinates of a pixel through which a ray is to be generated.
        disable_distortions : bool
            If True, lens distortions are ignored in the computation. Otherwise, lens
            distortions are considered. Defaults to False

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - The origin of the rays in world space, a 1D numpy array of shape (3,).
            - Directions of the rays in world space, a 2D numpy array of shape (N, 3),
              where each row is a unit vector representing the direction of a ray.
        """
        # Handle lens distortions based on disable_distortions flag
        distortions = None if disable_distortions else self.lens.distortions

        # Convert rotation vector to rotation matrix
        rotation = cv2.Rodrigues(self.r_vec)[0]
        # Compute the camera origin in world space
        origin = -rotation.T @ self.t_vec

        # Initialize directions in camera space; assume z=1 for all directions
        directions_cam_space = np.ones((pixel_coords.shape[0], 3))
        # Update x, y coordinates based on undistorted pixel positions
        directions_cam_space[:, :2] = cv2.undistortPoints(pixel_coords.reshape(-1, 1, 2), self.lens.intrinsic,
                                                          distortions, None)[:, 0, :]

        # Transform directions to world space and normalize
        direction_world_space = np.einsum('cw,bc->bw', rotation, directions_cam_space)
        direction_world_space /= np.linalg.norm(direction_world_space, axis=1)[:, None]

        return origin, direction_world_space
