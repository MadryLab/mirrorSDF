from dataclasses import dataclass


@dataclass
class RenderingConfig:
    """
    Configuration settings for the rendering processes

    Parameters
    ----------
    background_spp : int, default=32
        The number of samples per pixel (SPP) taken along the ray
        used for background rendering.
    """

    background_spp: int = 32
    num_bsdf_samples: int = 16
