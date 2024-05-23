import torch as ch


def rgb_loss(prediction: ch.Tensor, ground_truth: ch.Tensor) -> ch.Tensor:
    return ch.nn.functional.mse_loss(prediction, ground_truth)


def eikonal_loss(not_normalized_normals: ch.Tensor) -> ch.Tensor:
    """
    Calculate the Eikonal loss for a given set of normals.

    The Eikonal loss enforces the gradient norm of a scalar field to be close to 1 everywhere.

    Parameters
    ----------
    not_normalized_normals : torch.Tensor
        A tensor of shape `(...,  3)` representing the **Unormalized** normals in a 3D space.

    Returns
    -------
    torch.Tensor
        The mean Eikonal loss computed over all input normals.
        
    References
    ----------
    https://arxiv.org/pdf/2002.10099.pdf

    """
    grad_norm = (not_normalized_normals.norm(dim=-1) - 1.0) ** 2
    grad_norm = grad_norm.nan_to_num(0.0)
    return grad_norm.mean()


def curvature_loss(hessian: ch.Tensor) -> ch.Tensor:
    """
    Calculate the curvature loss for a given set of Hessians.

    The curvature loss is intended to penalize high curvature regions in a surface by computing the absolute sum of the
    Hessian matrices' eigenvalues (approximated by the trace, or sum of diagonal elements, here) for each point, aiming
    to flatten the surface. This function computes the curvature loss as the mean of these absolute sums, handling NaN
    values by replacing them with zeros.

    Parameters
    ----------
    hessian : torch.Tensor
        representing the Hessian matrices

    Returns
    -------
    torch.Tensor
        The mean curvature loss computed over all input Hessians.

    References
    ----------
    https://arxiv.org/pdf/2306.03092.pdf
    """

    laplacian = hessian.sum(dim=-1).abs()
    laplacian = laplacian.nan_to_num(0.0)
    return laplacian.mean()
