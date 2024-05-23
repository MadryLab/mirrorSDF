import torch as ch

sobol_engine_2d = ch.quasirandom.SobolEngine(2, scramble=True)


class GGXSampler:
    @classmethod
    def _get_samples(cls, batch_size: int, num_samples: int, device: ch.device) -> ch.Tensor:
        return ch.rand(batch_size, num_samples, 2, device=device)

    @staticmethod
    def sample(roughness_squared: ch.Tensor, num_samples: int) -> ch.Tensor:
        uv = GGXSampler._get_samples(roughness_squared.shape[0], num_samples,
                                     device=roughness_squared.device)

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

    def pdf(self, roughness_squared: ch.Tensor, NdotH: ch.Tensor, VdotH: ch.Tensor) -> ch.Tensor:
        return ggx_distribution(NdotH, roughness_squared) / 4 / VdotH


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
    denominator = ((NdotH * roughness_squared - NdotH) * NdotH + 1)
    val = roughness_squared / (ch.pi * denominator ** 2)
    return val


