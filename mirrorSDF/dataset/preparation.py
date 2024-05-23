import os
from glob import glob
from os import path
from typing import Tuple, List

import numpy as np
import torch as ch

from .dataset import MirrorSDFDataset
from .shot_metadata import ShotMetadata
from ..aruco.utils import prepare_image_for_aruco
from ..optical_models import RGBImageSensor, Lens, Mirror
from ..utils.geometry import generate_cube_triangles, quads_to_tris, expand_rectangles
from ..utils.image import read_hdr_stack, generate_pixel_grid
from ..utils.light_simulation import mirror_bounce_if_intersects, ray_triangles_intersection_pytorch


def precompute_intersection_data(mirror_corners: np.ndarray, marker_locations: np.ndarray,
                                 object_center: np.ndarray, object_bounding_box: np.ndarray,
                                 mirror_safety_padding: float,
                                 marker_safety_padding: float) -> tuple:
    """
    Precomputes geometric data for intersection tests within a scene, including
    the normal vector of a mirror and triangles representing objects, mirrors,
    and markers with applied safety padding.

    Parameters
    ----------
    mirror_corners : np.ndarray
        Array of shape (N, 4, 3) representing the corners of N mirrors, each defined by four vertices in 3D space.
    marker_locations : np.ndarray
        Array of shape (M, 4, 3) representing the corners of M markers, each defined by four vertices in 3D space.
    object_center : np.ndarray
        The 3D center point of the object of interest.
    object_bounding_box : np.ndarray
        Dimensions of the object's bounding box as np.ndarray (width, height, depth).
    mirror_safety_padding : float
        Safety padding to expand the mirror's bounding area.
    marker_safety_padding : float
        Safety padding to expand the marker's bounding area.

    Returns
    -------
    tuple
        A tuple containing the mirror normal vector, an array of all triangles
        of interest, and an array of triangles representing the mirror surface.

    Notes
    -----
    This function assumes that the mirror and markers are rectangular
    """
    # Generate triangles for the object's bounding box and shift them to the object's center
    bounding_box_triangles = generate_cube_triangles(*object_bounding_box) + object_center[None]

    # Convert mirror corners to triangles directly
    mirror_triangles = quads_to_tris(mirror_corners)

    # Expand mirror corners for safety padding and convert to triangles
    expanded_mirror_corners = expand_rectangles(mirror_corners, mirror_safety_padding)
    expanded_mirror_triangles = quads_to_tris(expanded_mirror_corners)

    # Expand marker locations for safety padding and convert to triangles
    expanded_markers = expand_rectangles(marker_locations, marker_safety_padding)
    marker_triangles = quads_to_tris(expanded_markers).reshape(-1, 3, 3)

    # Aggregate all triangles for intersection tests
    all_triangles = np.concatenate([mirror_triangles, expanded_mirror_triangles,
                                    bounding_box_triangles, marker_triangles])

    # Compute the mirror normal
    d1 = mirror_corners[0, 1] - mirror_corners[0, 0]
    d2 = mirror_corners[0, -1] - mirror_corners[0, 0]
    mirror_normal = np.cross(d1, d2)
    mirror_normal /= np.linalg.norm(mirror_normal)
    # Ensure the normal is oriented positively in the z-direction
    if mirror_normal[-1] < 0:
        mirror_normal *= -1

    return mirror_normal, all_triangles, mirror_triangles


def ray_segmentation(triangles: ch.Tensor, origins: ch.Tensor, directions: ch.Tensor,
                     num_fiducials: int) -> tuple[ch.Tensor, ch.Tensor]:
    """
    Segments ray intersections into categories based on their interactions with
    various scene components, including the mirror, bounding boxes, and markers.

    Parameters
    ----------
    triangles : ch.Tensor
        The vertices of triangles representing all objects in the scene,
        shape (M, 3, 3), where M is the number of triangles.
    origins : ch.Tensor
        The origin points of the rays, shape (N, 3), where N is the number of rays.
    directions : ch.Tensor
        The direction vectors of the rays, shape (N, 3).
    num_fiducials : int
        The number of fiducial markers present in the scene.

    Returns
    -------
    tuple[ch.Tensor, ch.Tensor]
        A tuple containing two tensors:
        - intersects_bb: A boolean tensor indicating which rays intersect with the bounding box, shape (N,).
        - avoid: A boolean tensor indicating which rays should be avoided based on hitting
          markers or only the expanded mirror area, shape (N,).

    Notes
    -----
    The function performs ray-triangle intersection tests to determine whether rays intersect
    with specific scene components. It then uses these intersections to categorize rays into
    those that intersect the bounding box of an object and those that should be avoided for
    reasons such as intersecting with markers or only intersecting with an expanded safety
    area around the mirror without hitting the mirror itself.
    """
    # Perform ray-triangle intersection tests for all triangles
    remaining_intersections, _ = ray_triangles_intersection_pytorch(triangles, origins, directions)

    # Segment the intersection results into different categories based on the scene components
    # See a couple lines below what each block of triangles is
    split = ch.split(remaining_intersections, [2, 2, 12, 2 * num_fiducials], dim=-1)
    split = [x.any(-1) for x in split]

    # Unpack the segmented intersection results into specific categories
    intersects_mirror, intersects_expanded_mirror, intersects_bb, intersects_markers = split

    # Determine which rays should be avoided based on hitting markers or only the expanded mirror
    avoid = intersects_markers | (intersects_expanded_mirror & ~intersects_mirror)

    return intersects_bb, avoid


def segment_image(
        origin: np.ndarray, directions: np.ndarray, mirror_corners: np.ndarray,
        marker_locations: np.ndarray, object_center: np.ndarray, object_bounding_box: np.ndarray,
        mirror_thickness: float, mirror_safety_padding: float, marker_safety_padding: float,
        eta: float, batch_size: int = 16_384, device: ch.device = ch.device('cuda:0')) -> (ch.Tensor, ch.Tensor):
    """
    Segments an image into regions based on markers, object bounding_boxes, and mirror reflections.

    Parameters
    ----------
    origin : np.ndarray
        The origin of the rays used for segmentation.
    directions : np.ndarray
        The directions of the rays emitted from the origin.
    mirror_corners : np.ndarray
        The coordinates of the corners of the mirror.
    marker_locations : np.ndarray
        The locations of the markers in the scene.
    object_center : np.ndarray
        The center of the object of interest.
    object_bounding_box : np.ndarray
        The bounding box of the object.
    mirror_thickness : float
        The thickness of the mirror.
    mirror_safety_padding : float
        Additional padding added to the mirror to ensure rays intersect.
    marker_safety_padding : float
        Additional padding added to the markers to ensure rays intersect.
    eta : float
        The refractive index ratio used for calculating the reflection of rays off the mirror.
    batch_size : int, optional
        The number of rays processed in each batch. Default is 16,384.
    device : torch.device, optional
        The device (CPU/GPU) on which to perform computations. Default is 'cuda:0'.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        A tuple containing two tensors:
        - The first tensor indicates whether each ray intersects with the object shape.
        - The second tensor indicates whether each ray is valid for environment mapping.
    """

    num_fiducials = marker_locations.shape[0]

    mirror_normal, all_triangles, mirror_triangles = precompute_intersection_data(
        mirror_corners, marker_locations,
        object_center, object_bounding_box,
        mirror_safety_padding,
        marker_safety_padding)

    # Convert numpy arrays to PyTorch tensors and transfer them to the specified device
    mirror_normal_p = ch.from_numpy(mirror_normal).to(device)
    all_triangles_p = ch.from_numpy(all_triangles).to(device)
    mir_triangles_p = ch.from_numpy(mirror_triangles).to(device)
    origin_p = ch.from_numpy(origin).to(device)
    direction_p = ch.from_numpy(directions).to(device)

    all_valid_train_shape = []  # To store rays intersecting with the object shape
    all_valid_train_env = []  # To store rays valid for environment mapping

    # Process directions in batches to manage memory usage on the device
    directions_batches = ch.split(direction_p, batch_size, dim=0)
    for directions_batch in directions_batches:

        # Compute mirror bounce for intersecting rays and determine their subsequent path
        bounce_coords, bounce_direction, _ = mirror_bounce_if_intersects(origin_p, directions_batch, mirror_normal_p,
                                                                         mir_triangles_p,
                                                                         mirror_thickness, eta)

        # Determine ray intersections before and after the bounce, and compute valid training regions
        intersects_bb_primary, avoid_primary = ray_segmentation(all_triangles_p,
                                                                origin_p,
                                                                directions_batch,
                                                                num_fiducials)
        intersects_bb_secondary, avoid_secondary = ray_segmentation(all_triangles_p,
                                                                    bounce_coords,
                                                                    bounce_direction,
                                                                    num_fiducials)
        
        # Combine results to determine valid rays for training on shape and environment
        ray_ok = ~(avoid_primary | avoid_secondary)
        valid_train_shape = (intersects_bb_primary | intersects_bb_secondary) & ray_ok
        valid_train_env = ray_ok & (~valid_train_shape)

        # Collect results, converting them back to CPU for further processing or analysis
        all_valid_train_env.append(valid_train_env.cpu())
        all_valid_train_shape.append(valid_train_shape.cpu())

    # Concatenate results from all batches
    all_valid_train_env = ch.cat(all_valid_train_env)
    all_valid_train_shape = ch.cat(all_valid_train_shape)
    return all_valid_train_shape, all_valid_train_env


def process_image_first_stage(image_index: int, image_file_names: List[str], sensor: RGBImageSensor,
                              lens: Lens, mirror: Mirror, black_point: float, white_point: float
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ShotMetadata, any]:
    """
    Processes the first stage of image analysis for a given set of images, 
    focusing on environment detection, camera pose estimation, and ray generation.

    Parameters
    ----------
    image_index : int
        Index of the image being processed.
    image_file_names : List[str]
        List of file names for the images to be processed.
    sensor : RGBImageSensor
        The sensor used to capture the images.
    lens : Lens
        The lens used with the sensor.
    mirror : Mirror
        The mirror component used in the imaging system.
    black_point : float
        The black point value for image normalization.
    white_point : float
        The white point value for image normalization.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ShotMetadata, any]
        A tuple containing:
        - Pixel grid generated for the processed image.
        - The loaded image after HDR stacking.
        - Origin points for ray generation.
        - Directions of the generated rays.
        - Metadata of the processed shot.
        - The camera object after pose estimation.

    Raises
    ------
    ValueError
        If the images span across multiple environments, indicating an inconsistency in the dataset.

    """
    environments = []
    for file_name in image_file_names:
        environments.append(path.normpath(file_name).split(os.sep)[-2])
    if len(set(environments)) != 1:
        raise ValueError("Some of groups of picture span multiple environment, your data (and the value ot T_N)")

    environment = int(environments[0])
    loaded_image = read_hdr_stack(image_file_names, sensor)

    processed = prepare_image_for_aruco(loaded_image, low=black_point, high=white_point)
    crosshairs = mirror.crosshair_field.detect_crosshairs(processed)[0].reshape(-1, 2)
    camera, pose_rmse = mirror.estimate_camera_pose(processed, lens)

    shot_metadata = ShotMetadata(image_index, environment, camera.r_vec,
                                 camera.t_vec, pose_rmse, len(crosshairs))

    pixels = generate_pixel_grid(*processed.shape[:2])

    origin, directions = camera.generate_rays_from_pixels(pixels)

    return pixels, loaded_image, origin, directions, shot_metadata, camera


def write_partial_dataset(mask: np.ndarray, image: np.ndarray, pixel_coords: np.ndarray,
                          shot_id: int, folder: str, prefix: str) -> None:
    """
    Saves a filtered subset of image data points to a NumPy file, based on the provided mask.

    Parameters
    ----------
    mask : np.ndarray
        A boolean array where True values indicate the data points to be saved.
    image : np.ndarray
        The image data array, from which measurements will be extracted based on the mask.
        Expected to be in a shape that supports indexing by the mask (e.g., (H*W, C) for an image).
    pixel_coords : np.ndarray
        The coordinates of each pixel/data point in the image, aligned with the `image` array's first dimension.
    shot_id : int
        An identifier for the current shot or image processing instance, to be saved with each data point.
    folder : str
        The path to the directory where the partial dataset file will be saved.
    prefix : str
        A prefix for the filename, allowing for organization or categorization of saved files.
    """
    dataset = np.zeros(mask.sum().item(), dtype=MirrorSDFDataset.storage_dtype)
    dataset['shot_id'] = shot_id
    dataset['coord'] = pixel_coords[mask]
    dataset['measurement'] = image.reshape(-1, 3)[mask]

    np.save(path.join(folder, f'partial_{prefix}_{shot_id}.npy'), dataset)


def process_image_second_stage(shot_metadata: ShotMetadata, loaded_image: np.ndarray, destination_folder: str,
                               pixels: np.ndarray, origin: np.ndarray, directions: np.ndarray, mirror: Mirror,
                               object_center: np.ndarray, object_bounding_box: np.ndarray,
                               mirror_safety_padding: float, marker_safety_padding: float,
                               device: ch.device = ch.device('cuda:0')) -> None:
    """
    Processes the second stage of image analysis, focusing on segmenting the image for object and environment training,
    and then saving these datasets along with the shot metadata.

    Parameters
    ----------
    shot_metadata : ShotMetadata
        Metadata about the shot, including identifiers and pose estimation data.
    loaded_image : np.ndarray
        The image data loaded in the first stage of processing.
    destination_folder : str
        The path to the folder where segmented datasets and metadata will be saved.
    pixels : np.ndarray
        The coordinates of pixels in the loaded image.
    origin : np.ndarray
        The origin point(s) from which rays were generated for segmentation.
    directions : np.ndarray
        The direction vectors of rays used for segmentation.
    mirror : Mirror
        The mirror object corresponding to the scene
    object_center : np.ndarray
        The center point of the object of interest in the scene.
    object_bounding_box : np.ndarray
        The bounding box defining the extents of the object.
    mirror_safety_padding : float
        Safety padding added around the mirror for segmentation purposes.
    marker_safety_padding : float
        Safety padding added around markers for segmentation purposes.
    device : torch.device, optional
        The computing device (e.g., CPU, GPU) to use for processing. Defaults to 'cuda:0'.
    """
    train_object, train_env = segment_image(origin[None], directions, mirror.mirror_corners_clockwise[None],
                                            mirror.crosshair_corners.copy(), object_center, object_bounding_box,
                                            mirror.thickness_mm, mirror_safety_padding, marker_safety_padding,
                                            1 / mirror.ior, device=device)
    write_partial_dataset(train_object, loaded_image, pixels, shot_metadata.shot_id, destination_folder, 'shape')
    write_partial_dataset(train_env, loaded_image, pixels, shot_metadata.shot_id, destination_folder, 'env')
    shot_metadata.to_folder(destination_folder)


def get_image_list_sorted(folder: str, extension: str, num_bracketed_shots: int) -> np.ndarray:
    """
    Retrieve a sorted list of image file paths from a specified folder, filtering by file extension,
    and organize these paths into groups representing bracketed shots.
    
    Assumes the following pattern folder/LIGHT_SOURCE_LOCATION/IMAGE_ID.extension

    Parameters
    ----------
    folder : str
        The directory path from which to search for image files.
    extension : str
        The file extension to filter by (e.g., 'CR3', 'JPG') without the leading dot.
    num_bracketed_shots : int
        The number of images in each bracketed set. This parameter determines how the returned
        array is shaped; each sub-array contains paths to `num_bracketed_shots` images.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each row contains file paths to images that are part of the same
        bracketed shot sequence.
    """
    # Generate a sorted list of file paths for images with the specified extension
    all_file_names = list(sorted(glob(path.join(folder, '*/', f"*.{extension}"))))

    # Reshape the list into a 2D numpy array where each row corresponds to a bracketed shot sequence
    all_file_names = np.array(all_file_names).reshape(-1, num_bracketed_shots)

    return all_file_names
