from dataclasses import dataclass


@dataclass
class DiffuseIrradianceConfig:
    """
    Configuration for generation of the diffuse irradiance dataset using Monte Carlo methods.

    Attributes
    ----------
    num_viewpoints_per_env : int
        The number of viewpoints per environment from which to sample irradiance.
        Environment will be randomized so there might not necessarily be exactly
        the same amount of sampler per environment
    num_normals_per_viewpoint : int
        The total number of surface normals sampled per viewpoint we simulate
        diffuse irradiance for
    random_normals_per_round : int
        The number of random normals sampled in each round of simulation.
    num_sampling_rounds : int
        The total number of rounds to perform the Monte Carlo simulation.
    """

    num_viewpoints_per_env: int = 2048
    num_normals_per_viewpoint: int = 2048
    random_normals_per_round: int = 2048
    num_sampling_rounds: int = 256
