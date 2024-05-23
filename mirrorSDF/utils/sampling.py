import torch as ch
import torch.nn.functional as F
from ..dataset.batch import Batch


def torch_interp(x, xp, fp):
    """
    Batched one-dimensional linear interpolation in PyTorch using torch.lerp.

    Arguments:
    - x: 2D tensor of shape (B, M) where M is the number of x-coordinates at which to evaluate the interpolated values.
    - xp: 2D tensor of shape (B, N) where N is the number of x-coordinates of the data points.
    - fp: 2D tensor of shape (B, N) where N is the number of y-coordinates of the data points.

    Returns:
    - 2D tensor of interpolated values of shape (B, M).
    """

    import torch
    _, N = xp.shape

    # Since xp is sorted within each batch, we can use torch.searchsorted directly
    bin_indices = torch.searchsorted(xp, x, right=True) - 1

    # Clip bin_indices to the range [0, N-2]
    bin_indices = torch.clamp(bin_indices, 0, N - 2)

    # Compute the weight for lerp
    x_low = torch.gather(xp, 1, bin_indices)
    x_high = torch.gather(xp, 1, bin_indices + 1)
    weight = (x - x_low) / (x_high - x_low).clamp_min(1e-9)

    # Compute interpolated values using lerp
    y_low = torch.gather(fp, 1, bin_indices)
    y_high = torch.gather(fp, 1, bin_indices + 1)

    return torch.lerp(y_low, y_high, weight)


def equi_spaced_samples(batch_size: int, num_samples: int, device: ch.device, randomized: bool):
    """
    Generate a tensor of equally spaced samples with an optional random perturbation.

    This function creates a batch of sample tensors, each containing equally spaced values
    within the interval [0, 1]. An optional random perturbation can be added to each sample to 
    avoid sampling at precisely the same points every time, which is useful for stochastic processes.

    Parameters
    ----------
    batch_size : int
        The number of sample tensors to generate, corresponding to the batch size.
    num_samples : int
        The number of samples per tensor.
    device : ch.device
        The PyTorch device on which to create the tensor (e.g., CPU or CUDA GPU).
    randomized : bool
        If True, add a small random perturbation to each sample. If False, samples are evenly spaced.

    Returns
    -------
    ch.Tensor
        A tensor of shape `(batch_size, num_samples)` containing the generated samples
        for each item in the batch. Values lie in ]0, 1[

    Examples
    --------
    >>> equi_spaced_samples(batch_size=2, num_samples=4, device=ch.device('cpu'), randomized=False)
    tensor([[0.2000, 0.4000, 0.6000, 0.8000],
            [0.2000, 0.4000, 0.6000, 0.8000]])
    """
    offset = 1 / num_samples
    t_s = ch.linspace(0, 1 - offset, num_samples, device=device)[None]
    t_s = t_s.expand(batch_size, -1)

    if randomized:
        t_s = t_s + ch.rand_like(t_s) * offset
    else:
        t_s = t_s + offset / 2

    return t_s


def sample_according_to_density(xes: ch.Tensor, density: ch.Tensor, num_samples: int, randomized=False):
    batch_size = density.shape[0]
    t_normalized = equi_spaced_samples(batch_size, num_samples, device=density.device, randomized=randomized)
    cdf = ch.cumsum(density, dim=1).clip(0, 1)  # Total probability can't be more than 1
    total_prob = cdf[:, -1]
    total_prob = ch.where(total_prob == 0, 1, total_prob)  # Avoid division by 0
    cdf = cdf / total_prob[:, None]
    cdf = F.pad(cdf, (1, 0, 0, 0), mode='constant', value=0)
    xes = F.pad(xes, (1, 0, 0, 0), mode='constant', value=0)

    result = torch_interp(t_normalized, cdf, xes)

    return result


def sample_nerf_pp(batch: Batch, n_samples: int, randomized: bool, eps: float = 1e-10):
    """
    Generate points where to sample according to the nerf++ inverted sphere parameterization

    Parameters
    ----------
    batch : Batch
        The batch object containing all the ray information
    n_samples : int
        The number of samples to generate per ray.
    randomized : bool
        If True, samples we randomize the sampling using the hierarchical sampling described in the
        original NeRF paper
    eps: float, defaults to 1e-10
        Value that controls how close we can get from the maximum and minimum values of 1/Z

    Returns
    -------
    torch.Tensor
        A tensor of points in 3D space where one should evaluate the NeRF model, shaped as (N, n_samples, 3),
        where N is the number of samples per Ray.

    Notes
    -----
    This achieves the same thing as Section 4 (See fig8) but more efficiently using the equation of
    intersection between a sphere and a ray.

    References
    ----------
    
    See: https://arxiv.org/abs/2010.07492
    """
    min_bg, max_bg = batch['background_radius_range']
    t_n = equi_spaced_samples(batch['size'], n_samples, batch['origin'].device, randomized=randomized)
    t_n = t_n.clip(eps, 1 - eps)
    t_n = ch.flip(t_n, dims=(-1,))
    t = min_bg / t_n
    c_origins = batch['exit_bounce_coord']
    directions = batch['bounce_direction']
    ctc = (c_origins * c_origins).sum(dim=-1, keepdim=True)
    ctv = (c_origins * directions).sum(dim=-1, keepdim=True)
    b2_minus_4ac = ctv ** 2 - (ctc - t ** 2)
    dist_far = -ctv + b2_minus_4ac.sqrt()
    points = c_origins[:, None] + dist_far[:, :, None] * directions[:, None]
    return points
