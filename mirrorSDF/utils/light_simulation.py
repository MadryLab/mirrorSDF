from typing import Tuple, TYPE_CHECKING

import numpy as np
import torch as ch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..dataset.batch import Batch


def batch_dot(tensor_a: ch.Tensor, tensor_b: ch.Tensor) -> ch.Tensor:
    return (tensor_a * tensor_b).sum(-1)


def ray_triangles_intersection_pytorch(triangles: ch.Tensor, origins: ch.Tensor, directions: ch.Tensor,
                                       epsilon: float = 1e-6):
    """
    Compute ray triangles intersection for many rays and triangles.
    Uses the Moller-Trumbore algorithm:
    https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

    Parameters:
    - triangles: 3D coordinates of each triangle to try to intersect. Shape: (T, 3, 3)
    - origins: Origins of the rays. Shape: (N, 3)
    - directions: Direction of the rays: Shape: (N, 3)
    - epsilon: floating point tolerance to detect parallelism of rays with the triangle plane

    Returns:
    - intersects: whether this rays intersect a particular triangle. Shape (N, T)
    - t: When intersecting, the parametric location where it happened: Shape (N, T)
    """
    # We add the batch dimension
    triangles = triangles[None]
    origins = origins[:, None]
    directions = directions[:, None]

    a = triangles[:, :, 0]
    b = triangles[:, :, 1]
    c = triangles[:, :, 2]

    edge1 = (b - a)
    edge2 = (c - a)

    ray_cross_e2 = ch.cross(directions, edge2, dim=2)
    determinant = batch_dot(edge1, ray_cross_e2)

    # noinspection PyTypeChecker
    not_parallel: ch.Tensor = ch.abs(determinant) >= epsilon

    # We replace bad determinants to avoids nans and bad gradients
    determinant = ch.where(not_parallel, determinant, 1.0)

    inv_det = 1.0 / determinant
    s = origins - a
    s_cross_e1 = ch.cross(s, edge1, dim=2)

    u = inv_det * batch_dot(s, ray_cross_e2)
    v = inv_det * batch_dot(directions, s_cross_e1)
    t = inv_det * batch_dot(edge2, s_cross_e1)

    intersects = (
            not_parallel
            & (u >= 0) & (u < 1)
            & (v >= 0) & (u + v < 1)
            & (t >= epsilon)
    )

    return intersects, t


def compute_reflection_world_space(ray: ch.Tensor, normal: ch.Tensor) -> ch.Tensor:
    """
    Computes the reflection of a ray off a surface in world space using PyTorch tensors.

    This function calculates the direction of a reflected ray given the incident ray's
    direction and the normal vector of the reflecting surface, based on the reflection
    formula: R = 2(N·V)N - V, where R is the reflected ray, N is the normal, and V is
    the incident ray vector (inverted in this calculation).

    Parameters
    ----------
    ray : torch.Tensor
        The incident ray's direction vectors, expected to be a tensor of shape (N, 3),
        where N is the number of rays and each ray is a 3D vector.
    normal : torch.Tensor
        The normal vectors of the reflecting surface, expected to have the same shape as
        `ray`, (N, 3).

    Returns
    -------
    torch.Tensor
        The reflected ray's direction vectors, having the same shape as `ray` and `normal`,
        (N, 3).

    Examples
    --------
    >>> ray = ch.tensor([[0., -1., 0.]])  # A ray pointing downwards
    >>> normal = ch.tensor([[0., 1., 0.]])  # A normal pointing upwards (flat surface)
    >>> compute_reflection_world_space(ray, normal)
    tensor([[0., 1., 0.]])  # Reflected ray points upwards

    Note
    ----
    This function assumes that both `ray` and `normal` are normalized vectors.
    """
    # Invert the direction of the incident ray
    ray = -ray

    # Compute the dot product between each normal and the inverted ray
    # and maintain the last dimension for broadcasting
    dot_product = (normal * ray).sum(dim=-1, keepdim=True)

    # Compute the reflection vector using the formula
    reflection = 2 * dot_product * normal - ray

    # Normalize the reflected ray vectors
    reflection_normalized = F.normalize(reflection, dim=-1)

    return reflection_normalized


def simulate_ray_bounce_on_mirror(w_i: ch.Tensor, normal: ch.Tensor, location: ch.Tensor,
                                  eta: float, thickness: float) -> Tuple[ch.Tensor, ch.Tensor]:
    """
    Computes exit location and direction of a light ray hitting a second surface mirror,
    incorporating refraction effects based on custom derived equations. This method
    simplifies the calculation by not explicitly modeling all medium transitions.

    Parameters
    ----------
    w_i : ch.Tensor
        The incident ray vectors, with shape (N, 3), where N is the number of rays.
    normal : ch.Tensor
        The normal vectors at the bounce points on the second surface, with shape (N, 3).
    location : ch.Tensor
        The entering location of the ray into the mirror, with shape (N, 3).
    eta : float
        The ratio of the indices of refraction (n2/n1) at the interface.
    thickness : float
        The thickness of the medium through which the ray travels before bouncing.

    Returns
    -------
    Tuple[ch.Tensor, ch.Tensor]
        A tuple containing two ch.Tensor objects:
        - The first tensor represents the bounce origins, with shape (N, 3).
        - The second tensor represents the bounce directions, with shape (N, 3).
    """
    cos_theta_i = (w_i * normal).sum(-1)
    w_i_perpendicular = w_i - normal * cos_theta_i[:, None]
    w_i_perpendicular_normalized = F.normalize(w_i_perpendicular, dim=-1)

    sin2_theta_i = 1 - cos_theta_i ** 2
    sin2_theta_o = eta ** 2 * sin2_theta_i

    offset = w_i_perpendicular_normalized * thickness * ch.sqrt(sin2_theta_o / (1 - sin2_theta_o))[:, None]

    bounce_origin = location + 2 * offset
    bounce_direction = w_i_perpendicular - normal * cos_theta_i[:, None]
    return bounce_origin, bounce_direction


def mirror_bounce_if_intersects(origin: ch.Tensor, directions_batch: ch.Tensor,
                                mirror_normal: ch.Tensor, mirror_triangles: ch.Tensor,
                                mirror_thickness: float, eta: float) -> Tuple:
    """
    Determines the interaction of rays with a mirror surface and computes the
    refracted bounce locations and directions, considering potential intersections.
    If no hit occurs the ray is unchanged.

    Parameters
    ----------
    origin : ch.Tensor
        The origin points of the rays, shape (N, 3), where N is the number of rays.
    directions_batch : ch.Tensor
        The direction vectors of the rays, shape (N, 3).
    mirror_normal : ch.Tensor
        The normal vector of the mirror surface, shape (3,).
    mirror_triangles : ch.Tensor
        The vertices defining the triangles of the mirror surface, shape (M, 3, 3),
        where M is the number of triangles.
    mirror_thickness : float
        The thickness of the mirror, affecting the refraction calculation.
    eta : float, optional
        The ratio of the indices of refraction (n2/n1) at the interface, with a default value.

    Returns
    -------
    Tuple
        A tuple containing two ch.Tensor objects:
        - The first tensor represents the bounce locations of the rays after interaction,
          with shape (N, 3).
        - The second tensor represents the new directions of the rays after the bounce,
          with shape (N, 3).

    Notes
    -----
    - This function first checks for ray-mirror intersections using ray-triangle intersection tests.
    - For rays that intersect the mirror, it calculates the bounce location and direction
      considering refraction, based on the mirror's physical properties (thickness, eta).
    - Rays that do not hit the mirror are returned with their original direction and origin.
    """
    # Determine if and where the rays intersect with the mirror
    intersects_mirror, mirror_t = ray_triangles_intersection_pytorch(
        mirror_triangles,
        origin,
        directions_batch
    )
    # Find the closest intersection point for each ray
    mirror_t = mirror_t.min(-1).values
    intersects_mirror = intersects_mirror.any(-1)
    # Calculate the intersection location
    intersection_location = origin + mirror_t[:, None] * directions_batch

    # Compute the bounce coordinates and directions at the intersection points
    bounce_coords, bounce_direction = simulate_ray_bounce_on_mirror(directions_batch, mirror_normal,
                                                                    intersection_location,
                                                                    eta, mirror_thickness)
    # Update the bounce direction and coordinates only for rays that hit the mirror
    bounce_direction = ch.where(intersects_mirror[:, None], bounce_direction, directions_batch)
    bounce_coords = ch.where(intersects_mirror[:, None], bounce_coords, origin)

    return bounce_coords, bounce_direction, intersection_location


def evaluate_samples_ray(batch: 'Batch', normalized_t: ch.Tensor,
                         foreground: bool) -> Tuple[ch.Tensor, ch.Tensor, ch.Tensor]:
    """
    Evaluate the samples of rays at normalized time steps, taking the mirror into account.

    This function calculates the positions of rays at specified normalized time steps.
    It differentiates between the rays before and after the bounce, providing their respective positions.
    If the ray has bounced (reflected or refracted), the position is calculated based on the bounce parameters.

    Parameters
    ----------
    batch : Batch
        The batch object describing the rays we are processing
    normalized_t : ch.Tensor
        The normalized time steps at which to evaluate the rays, ranging from 0 to 1. Shape (BS, N)
    foreground : bool
        Flag indicating whether to sample within the foreground or background of the scene

    Returns
    -------
    ch.Tensor
        The positions of the rays at the specified normalized time steps. Shape: (BS, N, 3)
    ch.Tensor
        The direction of the rays at the specified normalized time steps. Shape: (BS, N, 3)
    ch.Tensor
        The time steps (un-normalized) where the samples were taken. Shape: (BS, N)

    """
    if foreground:
        start, end = batch['foreground_t_range']
    else:
        start, end = batch['background_t_range']

    span = end - start

    t = normalized_t * span + start
    primary_rays = batch['origin'][:, None] + batch['direction'][:, None] * t[:, :, None]
    secondary_rays = (batch['exit_bounce_coord'][:, None]
                      + batch['bounce_direction'][:, None] * (t[:, :, None] - batch['t_bounce'][:, None]))
    after_bounce = t >= batch['t_bounce']
    positions = ch.where(after_bounce[:, :, None], secondary_rays, primary_rays)
    directions = ch.where(after_bounce[:, :, None], batch['bounce_direction'][:, None], batch['direction'][:, None])
    return positions, directions, t


def generate_bounce_LUT(num_bins: int = 64, eta: float = 1 / 1.53, thickness: float = 3) -> ch.Tensor:
    """
    Generate a Look-Up Table (LUT) for solving the inverse problem of light bouncing on
    the mirror.

    The goal is to be able, given a bounce on a perfect mirror, where would be the actual
    incoming ray that would behave similarly on a second surface mirror.

    The idea behind this implementation is to simulate the behavior of for a bunch of incoming
    angles. measure the effective angle it would be on a perfect mirror, and the equivalent
    offset. We then store it in an inverted table so that we don't have to solve for
    the equation in real time.

    The final table maps the cos of the angle (on a perfect mirror) to the offset (in mm) that
    the equivalent ray on a second surface mirror.

    Parameters
    ----------
    num_bins : int, optional
        The number of pre-computed values.
    eta : float, optional
        The refractive index of the medium, by default 1/1.53.
    thickness : float, optional
        The thickness of the medium in some units, by default 3.

    Returns
    -------
    ch.Tensor
        A PyTorch tensor containing the computed offsets for equispaced cosine incident angles.
    """
    print("MT", thickness)

    def simulate(cos_theta_i: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the change in angle and compute the offset for a given cosine of the incident angle.

        This is essentially a scalar version of the function `simulate_ray_bounce_on_mirror` that
        also computes the equivalent angle.

        Parameters
        ----------
        cos_theta_i : np.ndarray
            An array of cosines of the incident angles to simulate.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing arrays of the modified cosines of the incident angles and their corresponding offsets.
        """
        sin2_theta_i = 1 - cos_theta_i ** 2
        sin2_theta_o = eta ** 2 * sin2_theta_i
        current_offset = thickness * np.sqrt(sin2_theta_o / (1 - sin2_theta_o))
        ratio = np.sqrt(current_offset ** 2 + 1 + 2 * current_offset * np.sqrt(sin2_theta_i))
        return cos_theta_i / ratio, current_offset

    original_cos_theta = np.linspace(0, 1, 10000)
    new_cos_theta, offset = simulate(original_cos_theta)
    equispaced_new_cost_theta = np.linspace(0, 1, num_bins)
    values = np.interp(equispaced_new_cost_theta, new_cos_theta, offset)
    return ch.from_numpy(values)


def solve_bounce_location(source: ch.Tensor, destination: ch.Tensor, mirror_point: ch.Tensor,
                          mirror_normal: ch.Tensor, lut: ch.Tensor) -> (ch.Tensor, ch.Tensor, ch.Tensor):
    """
    Finds the location on an infinite flat mirror where a ray coming from source would bounce and eventually
    reach destination.

    It simultaneously solves for the case of a perfect first surface mirror and approximate the solution
    for a second surface mirror using the LUT passed as an argument and precomputed using `generate_bounce_LUT`
    with the physical attributes of the mirror.

    It assumes the mirror is only reflective on one side (the one with positive dot product with the normal)

    Parameters
    ----------
    source : torch.Tensor
        The starting point(s) of the ray(s), shape (N, 3).
    destination : torch.Tensor
        The ending point(s) of the ray(s), shape (N, 3).
    mirror_point : torch.Tensor
        A point belonging to the mirror, shape (N, 3) or (3,)
    mirror_normal : torch.Tensor
        The normal of the mirror, shape (N, 3) or (3,)
    lut : torch.Tensor
        The lookup table for computing offsets based on the cosine of the angle, shape (M,).
        This shall be precomputed using `generate_bounce_LUT`

    Returns
    -------
    true_bounce_location : torch.Tensor
        The point on the mirror that where the bounce would happen. Assuming the mirror is a second surface mirror
    perfect_bounce_location : torch.Tensor
        The point on the mirror that where the bounce would happen. Assuming the mirror is a perfect
        first surface mirror.
    valid : torch.Tensor
        A boolean describing for each combination source, dest, whether a bounce on the mirror is possible
    """
    # Linear part solution
    source_cos = batch_dot(source - mirror_point, mirror_normal)  # 3 FLOPS + 3 MAD
    dest_cos = batch_dot(destination - mirror_point, mirror_normal)  # 3FLOPS + 3 MAD
    valid = (source_cos > 0) & (dest_cos > 0)  # 3 bool
    source_p = source - source_cos[:, None] * mirror_normal  # 3 MAD
    dest_p = destination - dest_cos[:, None] * mirror_normal  # 3 MAD
    ratio_source = (dest_cos / (source_cos + dest_cos))[:, None]  # 1 div, 1 ADD
    perfect_bounce_location = ch.lerp(dest_p, source_p, ratio_source)  # 3 LERP

    # Non-linear correction
    w_i = source - perfect_bounce_location  # 3 FLOPS
    cos_theta_i = batch_dot(w_i, mirror_normal) / w_i.norm(dim=-1, p=2)  # 3 MAD 1DIV 1 SQRT
    offset = lut[ch.round(cos_theta_i * (lut.shape[0] - 1)).int()]  # 1 IDX, 1 FLOPS, 1 CONV
    dir_offset = source_p - perfect_bounce_location  # 3 FLOPS
    # 3 MAD 1 SQRT 6 FLOPS
    true_bounce_location = perfect_bounce_location + (offset / dir_offset.norm(dim=-1, p=2))[:, None] * dir_offset

    return true_bounce_location, perfect_bounce_location, valid
