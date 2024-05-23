import json
from os import path

import numpy as np


class ShotMetadata:
    """
    A class for storing and managing metadata associated with a specific shot.

    Attributes
    ----------
    shot_id : int
        Identifier for the shot.
    env_id : int
        Identifier for the environment where the shot was taken.
    r_vec : np.ndarray
        Rotation vector (Rodrigues' rotation formula) representing the shot's orientation.
    t_vec : np.ndarray
        Translation vector representing the shot's position.
    pose_est_rmse : float
        Root Mean Square Error (RMSE) of the pose estimation.
    num_markers_available : int
        Number of markers detected in this picture (and used for pose estimation)

    Methods
    -------
    to_folder(folder_path: str) -> None
        Saves the shot metadata to a JSON file in the specified folder.

    _get_filename(folder_path: str, shot_id: int) -> str
        Generates a file path for saving the shot metadata JSON file.

    from_file(folder_path: str, shot_id: int) -> 'ShotMetadata'
        Class method to load shot metadata from a JSON file.
    """

    def __init__(self, shot_id: int, env_id: int, r_vec: np.ndarray, t_vec: np.ndarray, pose_est_rmse: float, num_markers_available: int) -> None:
        """
        Initializes the ShotMetadata object with the provided parameters.

        Parameters
        ----------
        shot_id : int
            Identifier for the shot.
        env_id : int
            Identifier for the environment where the shot was taken.
        r_vec : np.ndarray
            Rotation vector (Rodrigues' rotation formula) representing the shot's orientation.
        t_vec : np.ndarray
            Translation vector representing the shot's position.
        pose_est_rmse : float
            Root Mean Square Error (RMSE) of the pose estimation.
        num_markers_available : int
            Number of markers detected in this picture (and used for pose estimation)
        """
        self.shot_id = shot_id
        self.env_id = env_id
        self.r_vec = r_vec
        self.t_vec = t_vec
        self.pose_est_rmse = pose_est_rmse
        self.num_markers_available = num_markers_available

    def to_folder(self, folder_path: str) -> None:
        """
        Saves the shot metadata as a JSON file in the specified folder.

        Parameters
        ----------
        folder_path : str
            The path to the folder where the metadata JSON file will be saved.
        """
        to_write = {
            'shot_id': self.shot_id,
            'env_id': self.env_id,
            'rotation_rodrigues': self.r_vec.tolist(),  # Convert numpy array to list for JSON serialization
            'translation': self.t_vec.tolist(),  # Convert numpy array to list for JSON serialization
            'pose_estimation_rmse': self.pose_est_rmse,
            'num_markers_detected': self.num_markers_available
        }

        with open(ShotMetadata._get_filename(folder_path, self.shot_id), 'w+') as handle:
            json.dump(to_write, handle)

    @staticmethod
    def _get_filename(folder_path: str, shot_id: int) -> str:
        """
        Generates the filename for saving the shot metadata JSON file.

        Parameters
        ----------
        folder_path : str
            The path to the folder where the metadata file will be saved.
        shot_id : int
            The identifier of the shot.

        Returns
        -------
        str
            The full file path for the metadata JSON file.
        """
        return path.join(folder_path, f'shot_metadata_{shot_id}.json')

    @classmethod
    def from_file(cls, folder_path: str, shot_id: int) -> 'ShotMetadata':
        """
        Loads shot metadata from a JSON file.

        Parameters
        ----------
        folder_path : str
            The path to the folder containing the metadata file.
        shot_id : int
            The identifier of the shot.

        Returns
        -------
        ShotMetadata
            An instance of ShotMetadata initialized with data loaded from the file.
        """
        with open(cls._get_filename(folder_path, shot_id), 'r') as handle:
            to_parse = json.load(handle)

        return ShotMetadata(
            to_parse['shot_id'],
            to_parse['env_id'],
            np.array(to_parse['rotation_rodrigues']).ravel(),  # Convert list back to numpy array
            np.array(to_parse['translation']).ravel(),  # Convert list back to numpy array
            to_parse['pose_estimation_rmse'],
            to_parse['num_markers_detected'],
        )
