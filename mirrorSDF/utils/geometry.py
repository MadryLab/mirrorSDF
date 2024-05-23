from typing import Tuple

import numpy as np
import torch as ch


def generate_cube_triangles(dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Generates the vertices for triangles composing the faces of a cube.

    This function calculates the vertices of triangles that make up each face of a cube,
    given the dimensions of the cube along the x, y, and z axes. Each face of the cube
    is composed of two triangles for efficient rendering in graphics applications.

    Parameters
    ----------
    dx : float
        The length of the cube along the x-axis.
    dy : float
        The length of the cube along the y-axis.
    dz : float
        The height of the cube along the z-axis.

    Returns
    -------
    np.ndarray
        An array of shape (12, 3, 3), representing the vertices of 12 triangles. Each
        triangle is defined by 3 vertices, and each vertex is a 3D point in space.

    Examples
    --------
    >>> generate_cube_triangles(2, 2, 2)
    array([[[...]]])  # Output is truncated for brevity

    Notes
    -----
    The cube is centered at the origin (0, 0, 0) of the coordinate system, with each
    face parallel to a pair of axes. The triangles are defined in a way to ensure the
    cube can be fully represented with minimal redundancy.
    """
    # Define the vertices of the cube
    vertices = np.array([
        [-dx/2, -dy/2, 0],    # V1
        [dx/2, -dy/2, 0],     # V2
        [dx/2, dy/2, 0],      # V3
        [-dx/2, dy/2, 0],     # V4
        [-dx/2, -dy/2, dz],   # V5
        [dx/2, -dy/2, dz],    # V6
        [dx/2, dy/2, dz],     # V7
        [-dx/2, dy/2, dz]     # V8
    ])

    # Define triangles for each face (two triangles per face)
    # Each triangle is represented by indices of its three vertices
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 6, 5], [4, 7, 6],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [3, 2, 6], [3, 6, 7],  # Back face
        [0, 3, 7], [0, 7, 4],  # Left face
        [1, 2, 6], [1, 6, 5]   # Right face
    ])

    # Return the vertices for each triangle
    return vertices[triangles]


def expand_rectangles(rectangles: np.ndarray, offset: float) -> np.ndarray:
    """
    Expands rectangles by a specified offset from their centers.

    This function calculates the expanded vertices of each rectangle based on a given
    offset. Each vertex is moved outward from the center of the rectangle by the offset
    distance, effectively resizing the rectangle.

    Parameters
    ----------
    rectangles : np.ndarray
        An array of shape (N, 4, 2) representing N rectangles. Each rectangle is defined
        by 4 vertices with 2D coordinates (x, y).
    offset : float
        The distance by which to expand each rectangle from its center. Positive values
        expand the rectangle, negative values contract it.

    Returns
    -------
    np.ndarray
        An array of shape (N, 4, 2) representing the expanded rectangles.
    """
    # Calculate the center of each rectangle
    rectangles_centers = rectangles.mean(axis=-2, keepdims=True)

    # Calculate the direction from each vertex to the center
    vertices_directions = rectangles - rectangles_centers

    # Calculate the distance of each vertex from the center
    distances_to_center = np.linalg.norm(vertices_directions, axis=-1, keepdims=True)

    # Normalize the directions
    vertices_directions /= distances_to_center

    # Calculate the expanded positions of the vertices
    expanded_rectangles = rectangles_centers + (distances_to_center + offset) * vertices_directions

    return expanded_rectangles


def quads_to_tris(quads: np.ndarray) -> np.ndarray:
    """
    Converts quadrilaterals to triangles by splitting each quad into two triangles.

    Parameters
    ----------
    quads : np.ndarray
        An array of shape (N, 4, 3) representing N quadrilaterals, each defined by four vertices
        in 3D space. The second dimension should have size 4 for the four vertices, and the
        last dimension should have size 3 for the (x, y, z) coordinates of each vertex.

    Returns
    -------
    np.ndarray
        An array of shape (N*2, 3, 3) representing the converted triangles. Each quadrilateral
        is split into two triangles, thus doubling the number of elements from the input array.
        Each triangle is defined by three vertices in 3D space.

    Examples
    --------
    >>> quads = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]])
    >>> quads_to_tris(quads)
    array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
           [[1, 1, 0], [0, 1, 0], [1, 0, 0]]])

    Note
    ----
    The conversion assumes that the vertices of each quadrilateral are specified in a
    consistent order, either clockwise or counterclockwise. The split is performed by
    connecting the first and third vertices, dividing the quad into two triangles.
    """
    # Unpack the vertices of the quads
    c1, c2, c3, c4 = quads.transpose(1, 0, 2)

    # Create two triangles from each quad
    tris = np.array([
        [c1, c2, c4],  # First triangle: vertices 1, 2, and 4
        [c3, c4, c2],  # Second triangle: vertices 3, 4, and 2
    ])

    # Reshape to create a (N*2, 3, 3) array for triangles
    return tris.transpose(2, 0, 1, 3).reshape(-1, 3, 3)


def intersect_with_sphere(origin: ch.Tensor, direction: ch.Tensor, radius: float = 1.0
                          ) -> Tuple[ch.Tensor, ch.Tensor]:
    """
    Calculates the distances from the ray origin to the points of intersection with a sphere.

    The sphere is centered at the coordinate system's origin with a given radius. The function
    returns two distances for each ray: one to the near intersection point and one to the far
    intersection point. If a ray does not intersect with the sphere, the corresponding entries
    in the result tensors will not represent valid distances.

    Parameters
    ----------
    origin : torch.Tensor
        The origin points of the rays. This tensor should have a shape of [..., 3], where the last
        dimension represents the XYZ coordinates.
    direction : torch.Tensor
        The direction vectors of the rays. This tensor should have the same shape as `origin`.
    radius : float, optional
        The radius of the sphere. Default is 1.0.

    Returns
    -------
    dist_near : torch.Tensor
        A tensor containing the distances from the ray origins to the near intersection points
        with the sphere. The shape of this tensor matches the shape of `origin` but with the last
        dimension being 1.
    dist_far : torch.Tensor
        A tensor containing the distances from the ray origins to the far intersection points
        with the sphere. The shape of this tensor matches that of `dist_near`.

    References
    ----------
    Based on the intersection calculation method the repository: https://github.com/NVlabs/neuralangelo
    """
    # Calculate the coefficients for the quadratic formula
    ctc = (origin * origin).sum(dim=-1, keepdim=True)  # Sum of squares of origin coordinates
    ctv = (origin * direction).sum(dim=-1, keepdim=True)  # Dot product of origin and direction vectors
    b2_minus_4ac = ctv ** 2 - (ctc - radius ** 2)  # Discriminant

    # Calculate distances to the near and far intersection points
    dist_near = -ctv - b2_minus_4ac.sqrt()
    dist_far = -ctv + b2_minus_4ac.sqrt()

    return dist_near, dist_far

def compute_mirror_normal(mirror_corners: np.ndarray) -> np.ndarray:
    """
    Computes the normal vector to the plane defined by the mirror corners.

    Parameters:
    -----------
    mirror_corners : np.ndarray
        An array of corner points of the mirror. It is expected to be of shape (4, 3),
        and each corner is a 3D point in space (x, y, z).

    Returns:
    --------
    np.ndarray
        A normalized 3D vector (np.ndarray of shape (3,)) representing the mirror's normal.
    """
    # Assuming mirror_corners is of shape (N, 3) and N >= 3
    d1 = mirror_corners[1] - mirror_corners[0]  # Vector from first to second corner
    d2 = mirror_corners[-1] - mirror_corners[0]  # Vector from first to last corner

    # Compute the cross product to find the normal vector
    mirror_normal = np.cross(d1, d2)

    # Normalize the normal vector to unit length
    mirror_normal /= np.linalg.norm(mirror_normal)

    # Ensure the normal vector points in the desired direction
    if mirror_normal[-1] < 0:
        mirror_normal *= -1

    return mirror_normal


def cartesian_to_spherical(coords):
    x, y, z = ch.unbind(coords, dim=-1)
    r = coords.norm(2, dim=-1).clamp(min=1e-9)
    theta = ch.acos(z / r)
    phi = ch.atan2(y, x)
    return ch.stack((theta, phi), dim=-1)


def spherical_to_cartesian(coords):
    # Spherical to Cartesian coordinates conversion
    theta, phi = ch.unbind(coords, -1)
    sin_theta = ch.sin(theta)
    cos_theta = ch.cos(theta)
    x = sin_theta * ch.cos(phi)
    y = sin_theta * ch.sin(phi)
    z = cos_theta

    return ch.stack([x, y, z], -1)
