import torch as ch
import torch.nn.functional as F

from ...utils.light_simulation import batch_dot
from ...utils.materials import impute_L, impute_H

sobol_engine_2d = ch.quasirandom.SobolEngine(2, scramble=True)


class GGXSampler:
    """
    A class for sampling directions based on the GGX (Trowbridge-Reitz) microfacet
    distribution model in computer graphics.

    This class uses Sobol sequences to generate quasi-random sample points for importance sampling the GGX distribution,
    which models the distribution of microfacets on a surface. The samples are used to generate directions for light
    reflection based on the surface's roughness.

    Attributes
    ----------
    _random_numbers : torch.Tensor or None
        A tensor holding the precomputed Sobol sequence samples. It is initialized as None and populated on the first
        call to `_get_samples`. The shape is dynamically based on the number of samples requested.
    """

    _random_numbers = None

    @classmethod
    def _get_samples(cls, num_samples: int, device: ch.device) -> ch.Tensor:
        """
        Generates or retrieves a set of quasi-random samples for GGX sampling.

        This method checks if precomputed samples exist and match the requested size. If not, it generates new samples
        using a Sobol sequence. The samples are then transferred to the specified device.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate or retrieve.
        device : torch.device
            The device on which the samples should be stored.

        Returns
        -------
        torch.Tensor
            The quasi-random samples for GGX sampling.
        """
        if cls._random_numbers is None or cls._random_numbers.shape[0] != num_samples:
            cls._random_numbers = sobol_engine_2d.draw(num_samples)

        cls._random_numbers = cls._random_numbers.to(device)
        return cls._random_numbers

    @classmethod
    def clear_samples(cls):
        """
        Clears the cached Sobol sequence samples.

        This method resets the `_random_numbers` class variable to None, forcing new samples
        to be generated on the next call to `_get_samples`.
        """
        cls._random_numbers = None

    @staticmethod
    def sample(batch_size: int, num_samples: int, N: ch.Tensor, V: ch.Tensor, roughness_squared: ch.Tensor,
               shading_to_normal_rotation: ch.Tensor) -> ch.Tensor:
        """
        Samples microfacet normal vectors based on the GGX distribution and computes corresponding light directions.

        Parameters
        ----------
        batch_size : int
            The number of separate instances for which to generate samples.
        num_samples : int
            The number of samples to generate per instance.
        N : torch.Tensor
            The normal vectors at the shading points.
            Unused for GGX. relies on the shading_to_normal rotation param instead
        V : torch.Tensor
            The view vectors from the shading points.
        roughness_squared : torch.Tensor
            The squared roughness parameter of the surface at each shading point.
        shading_to_normal_rotation : torch.Tensor
            The rotation matrices that align shading normals to the global coordinate system.

        Returns
        -------
        torch.Tensor
            The samples in Cartesian coordinates as light directions.
        """
        samples_uniform = GGXSampler._get_samples(num_samples, device=V.device)[None]
        samples_uniform = samples_uniform.expand(batch_size, -1, -1)
        return ggx_sample_kernel(V, ch.tensor(batch_size), samples_uniform, roughness_squared,
                                 shading_to_normal_rotation)

    def pdf(self, N: ch.Tensor, V: ch.Tensor, L: ch.Tensor, roughness_squared: ch.Tensor, *args,
            **kwargs) -> ch.Tensor:
        """
        Computes the probability density function (PDF) for already
        sampled directions based on the GGX distribution.

        Parameters
        ----------
        N : torch.Tensor
            The normal vectors at the shading points.
        V : torch.Tensor
            The view vectors from the shading points.
        L : torch.Tensor
            The light directions at the shading points.
        roughness_squared : torch.Tensor
            The squared roughness parameter of the surface at each shading point.
        *args
            Variable length argument list for future compatibility.
        **kwargs
            Arbitrary keyword arguments for future compatibility.

        Returns
        -------
        torch.Tensor
            The PDF values for each sampled direction.
        """
        return ggx_pdf_kernel(N, V, L, roughness_squared)


@ch.jit.script
def get_ggx_sample(uv: ch.Tensor, roughness_squared: ch.Tensor) -> ch.Tensor:
    """
    Generates microfacet normals based on GGX distribution using quasi-random samples.

    Parameters
    ----------
    uv : torch.Tensor
        A tensor of quasi-random numbers in the range [0, 1] with shape `(N, 2)`, where `n` is the number of samples.
        These numbers are used as inputs to the importance sampling formula for the GGX distribution.
    roughness_squared : torch.Tensor
        A tensor with shape `(N,)` representing the squared roughness parameter of the surface. This parameter controls
        the spread of the microfacet normals.

    Returns
    -------
    torch.Tensor
        A tensor of microfacet normals in Cartesian coordinates with shape `(N, 3)`.

    Notes
    -----
    The function computes the angles theta and phi for each sample point to convert from spherical to Cartesian
    coordinates. Theta is calculated using the importance sampling formula for the GGX distribution, determining the
    angle from the normal. Phi represents the azimuthal angle around the normal. The function returns the microfacet
    normal vectors (M) detached from the computation graph to prevent gradients from flowing through the sampling
    process.
    """
    u1, u2 = ch.unbind(uv, dim=-1)
    roughness_squared = roughness_squared.unsqueeze(1).expand(-1, u1.shape[1])

    # Theta is the angle from the normal
    # This uses the importance sampling formula for GGX
    tan_theta2 = roughness_squared * u1 / (1.0 - u1)
    cos_theta = 1.0 / ch.sqrt(1.0 + tan_theta2)
    sin_theta = ch.sqrt(1.0 - cos_theta ** 2)

    # Phi is the azimuth angle
    phi = 2.0 * ch.pi * u2

    # Spherical to Cartesian coordinates conversion
    x = sin_theta * ch.cos(phi)
    y = sin_theta * ch.sin(phi)
    z = cos_theta

    # Create the micro-facet normal
    M = ch.stack([x, y, z], dim=2)

    return M.detach()  # We don't usually want to backprop through the sampling


@ch.jit.script
def ggx_sample_kernel(V: ch.Tensor, batch_size: int, samples_uniform: ch.Tensor,
                      roughness_squared: ch.Tensor,
                      shading_to_normal_rotation: ch.Tensor) -> ch.Tensor:
    """
    Samples directions based on the GGX distribution for a batch of shading points.

    Parameters
    ----------
    V : torch.Tensor
        The view vectors at the shading points.
    batch_size : int
        The number of instances in the batch.
    samples_uniform : torch.Tensor
        Quasi-random numbers for sampling, expanded to match the batch size.
    roughness_squared : torch.Tensor
        The squared roughness parameters for the shading points.
    shading_to_normal_rotation : torch.Tensor
        Rotation matrices to align shading normals with the global coordinate system.

    Returns
    -------
    torch.Tensor
        The sampled light directions in Cartesian coordinates.
    """
    H = get_ggx_sample(samples_uniform, roughness_squared)
    H = H.expand(batch_size, -1, -1)
    H = F.normalize(batch_dot(H[:, :, None], shading_to_normal_rotation[:, None]), dim=-1)
    L = impute_L(V, H)
    L = F.normalize(L, dim=-1)

    return L


@ch.jit.script
def ggx_distribution(NdotH: ch.Tensor, roughness_squared: ch.Tensor) -> ch.Tensor:
    """
    Computes the GGX distribution function D for given dot products of normals and half-vectors.

    Parameters
    ----------
    NdotH : torch.Tensor
        The dot product between the surface normals and the half-vector H, representing the microfacet orientation.
    roughness_squared : torch.Tensor
        The squared roughness parameter of the surface, controlling the width of the GGX distribution.

    Returns
    -------
    torch.Tensor
        The values of the GGX distribution function D for each pair of normal and half-vector.

    Notes
    -----
    The GGX distribution function describes the proportion of microfacets oriented in the direction of the
    half-vector H for a given roughness value. This function is crucial for computing the reflectance properties of
    surfaces in physically based rendering models.
    """
    # Source: http://cwyman.org/code/dxrTutors/tutors/Tutor14/tutorial14.md.html
    denominator = NdotH.clip(0.001, 1) ** 2 * (roughness_squared - 1) + 1
    val = roughness_squared / (ch.pi * denominator ** 2 + 1e-8)
    return val


@ch.jit.script
def ggx_pdf_kernel(N: ch.Tensor, V: ch.Tensor, L: ch.Tensor, roughness_squared: ch.Tensor) -> ch.Tensor:
    """
    Computes the probability density function (PDF) of the GGX distribution for sampled directions.

    Parameters
    ----------
    N : torch.Tensor
        The normal vectors at the shading points.
    V : torch.Tensor
        The view vectors at the shading points.
    L : torch.Tensor
        The light directions at the shading points.
    roughness_squared : torch.Tensor
        The squared roughness parameters of the surface at each shading point.

    Returns
    -------
    torch.Tensor
        The PDF values for each sampled direction.

    Notes
    -----
    The function computes the half-vector (H) between the view (V) and light (L) directions, calculates the dot product
    between the normals (N) and H (NdotH), and the dot product between L and H (LdotH). It then uses these values to compute
    the GGX distribution function (D) and returns the PDF value for each direction based on the GGX distribution.
    """
    H = impute_H(V, L)
    NdotH = batch_dot(N.expand_as(H), H).clip(0, 1)
    LdotH = batch_dot(L.expand_as(H), H).clip(1e-4, 1)
    D = ggx_distribution(NdotH, roughness_squared)
    return D * NdotH / (4 * LdotH)
