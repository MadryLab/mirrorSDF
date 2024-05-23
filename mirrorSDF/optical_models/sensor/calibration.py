import numpy as np
import torch as ch
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm


def calibrate_single_channel(images: np.ndarray, exposures: np.ndarray,
                             batch_size: int, n_iterations: int, weights_exponent: float = 1.0) -> np.ndarray:
    """
    Calibrates a single channel of images to compute the camera response function.

    This function processes a set of images taken with different exposure times to
    calibrate the camera response function for a single color channel. It uses an
    iterative optimization approach to minimize the difference between the predicted
    and observed pixel values across all images, adjusting for the exposure times.

    This implementation is different from the paper (see References), as for high
    bit-depth the linear system is too complex to solve. Instead, we use gradient
    descent and a reparamterization to ensure monotonicity

    For 8-bit images the original algorithm (available in python-opencv), is probably
    a better choice, and more battle-tested

    Parameters
    ----------
    images : np.ndarray
        A 3D numpy array of shape (height, width, num_images) containing the images
        used for calibration. Images must be of the same size and aligned.
    exposures : np.ndarray
        A 1D numpy array containing relative exposure factors for each image.
    batch_size : int
        The size of batches to use for processing images during optimization.
    n_iterations : int
        The number of iterations to run the optimization process.
    weights_exponent : float, optional
        The exponent applied to the weighting function used in optimization.
        Default is 1.0.

    Returns
    -------
    np.ndarray
        A 1D numpy array of shape (Z_max), where `Z_max` is determined by
        the bit depth of the input images. It corresponds to the calibrated
        response curve for the channel.

    Raises
    ------
    ValueError
        If the bit depth of the images is not supported. Supported formats are
        8-bit (np.int8) and 16-bit (np.uint16).

    References
    ----------
    Debevec, P.E., Malik, J.: Recovering high dynamic range radiance maps from photographs.
    In: Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques,
    SIGGRAPH 1997, pp. 369–378. ACM Press/Addison-Wesley Publishing Co. (1997)

    Examples
    --------
    >>> images = np.random.randint(0, 256, size=(100, 100, 5), dtype=np.uint8)
    >>> exposures = np.array([1/60, 1/30, 1/15, 1/8, 1/4])
    >>> calibrate_single_channel(images, exposures, batch_size=10, n_iterations=100)
    """
    # Determine the maximum pixel value based on the image dtype
    if images.dtype == np.int8:
        Z_max = 256
    elif images.dtype == np.uint16:
        Z_max = 65536
    else:
        raise ValueError("Bit depth for image not supported")

    n_images = images.shape[2]

    # Using all pixels
    n_pixels = 1_000_000_000_000

    # Prepare the dataset and DataLoader
    dataset = PixelDataset(images, n_pixels)
    log_Dt = ch.from_numpy(np.log(exposures))
    weights = compute_weight_vector(Z_max, weights_exponent)
    parameters = ch.nn.Parameter(ch.diff(ch.log1p(ch.arange(Z_max, dtype=ch.float32))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup the optimizer
    optimizer = ch.optim.AdamW([parameters], weight_decay=10)

    # Optimization loop
    progress = tqdm(total=n_iterations, desc="Calibrating")
    iteration = 0
    while True:
        for i, Zes in enumerate(loader):
            rhs = log_Dt[None]

            # Compute weighted curve
            cw = weights[Zes.ravel()].reshape(Zes.shape)
            cw = cw / (ch.sum(cw, dim=1, keepdim=True) + 1e-8)

            curve = ch.nn.functional.pad(ch.cumsum(parameters, 0), (1, 0), mode='constant')
            prediction = curve[Zes]
            diff = prediction - rhs
            best_est = (diff * cw).sum(-1)
            loss = (prediction - best_est[:, None] - rhs) * cw
            loss = (loss ** 2).mean()

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            parameters.data.clip_(0, ch.inf)  # Ensure parameters remain non-negative
            progress.update(1)
            progress.set_postfix({
                'loss': loss.item()
            })
            iteration += 1
            if iteration >= n_iterations:
                break
        if iteration >= n_iterations:
            break

    # Normalize the response curve
    curve = normalize_response(parameters)
    progress.close()  # Ensure to close the progress bar after completion
    return curve.data.cpu().numpy()


class PixelDataset(Dataset):
    """
    A dataset class for loading and processing a set stack of images
    taken at different exposures.

    It supports taking a subset of the dataset based on their amount
    of standard deviation.

    Parameters
    ----------
    loaded_images : np.ndarray
        A 3D numpy array of images with shape (height, width, num_images), where
        each image is expected to be in a consistent pixel format.
    n_pixels : int
        The number of pixels to select based on the highest standard deviations
        across the images.

    Attributes
    ----------
    num_shots : int
        The number of images or "shots" loaded into the dataset.
    loaded_images : torch.Tensor
        A 2D tensor of shape (n_pixels, num_shots) containing the selected pixels
        after reshaping and sorting by standard deviation.

    Methods
    -------
    __len__():
        Returns the total number of selected pixels in the dataset.
    __getitem__(ix):
        Returns the pixel values across all images for the given index `ix`.

    Examples
    --------
    >>> loaded_images = np.random.randint(0, 256, size=(100, 100, 10))
    >>> dataset = PixelDataset(loaded_images, n_pixels=500)
    >>> len(dataset)
    500
    >>> dataset[0].shape
    torch.Size([10])
    """

    def __init__(self, loaded_images: np.ndarray, n_pixels: int):
        self.num_shots = loaded_images.shape[2]  # Number of images
        # Reshape and convert images to a 2D tensor where each row is a pixel across all images
        self.loaded_images = ch.from_numpy(loaded_images.reshape(-1, self.num_shots).astype(np.int32))
        # Compute standard deviation and sort pixels by it in descending order
        std = ch.argsort(-self.loaded_images.float().std(dim=1))
        # Select top `n_pixels` based on their standard deviation
        self.loaded_images = self.loaded_images[std[:n_pixels]]

    def __len__(self) -> int:
        """
        Returns the total number of pixels in the dataset.

        Returns
        -------
        int
            The number of pixels selected for the dataset.
        """
        return self.loaded_images.shape[0]

    def __getitem__(self, ix: int) -> ch.Tensor:
        """
        Retrieves the pixel values across all images for the specified index.

        Parameters
        ----------
        ix : int
            The index of the pixel to retrieve.

        Returns
        -------
        torch.Tensor
            The pixel values across all images at the given index.
        """
        return self.loaded_images[ix]


def compute_weight_vector(Z_max: int, exponent: float = 1.0) -> ch.Tensor:
    """
    Computes a weight vector for pixel values as defined in Debevec and Malik's HDR imaging paper.

    This function calculates a weight for each possible pixel value based on its distance
    from the midpoint of the possible range. Pixel values closer to the midpoint are given higher
    weights. This is used in HDR imaging to give more importance to well-exposed pixel values.

    Parameters
    ----------
    Z_max : int
        The maximum pixel value in the images. Typically, this would be 65536 for 16-bit images.
    exponent : float, optional
        The exponent to apply to the weights, allowing for adjustment of the weighting curve.
        The default value is 1.0, which applies no additional curve adjustment.

    Returns
    -------
    torch.Tensor
        A 1D tensor of weights for each pixel value, normalized to the range [0, 1].

    References
    ----------
    Debevec, P.E., Malik, J.: Recovering high dynamic range radiance maps from photographs.
    In: Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques,
    SIGGRAPH 1997, pp. 369–378. ACM Press/Addison-Wesley Publishing Co. (1997)

    Examples
    --------
    >>> weights = compute_weight_vector(256)
    >>> weights.shape
    torch.Size([256])
    >>> weights[128]  # Weight at the midpoint
    tensor(0.5000)
    """
    z = np.arange(Z_max)  # Create an array of all possible pixel values
    weights = z.copy()  # Initialize weights as a copy of the pixel values
    cond = z > Z_max // 2  # Condition for pixel values greater than the midpoint
    weights[cond] = Z_max - z[cond]  # Adjust weights for pixel values beyond the midpoint
    weights = ch.from_numpy(weights).float() / Z_max  # Normalize weights and convert to PyTorch tensor
    return weights ** exponent  # Apply exponent and return the weight vector


def normalize_response(parameters: ch.Tensor) -> ch.Tensor:
    """
    Take the log-space sensor response paramterized by their cummulative differences (to ensure
    monotonicity) and transorm to a standart Lookup-table in linear space. The response
    is normalized so that it's midpoint is at 0.5.

    Parameters
    ----------
    parameters : torch.Tensor
        A 1D tensor of parameters to be normalized. It's expected that the parameters
        are in a log-space and parametrized as a cummulative sum.

    Returns
    -------
    torch.Tensor
        The normalized parameters as a 1D tensor, where the midpoint value is adjusted to 0.5,
        and the sequence is scaled accordingly.

    Examples
    --------
    >>> parameters = ch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> normalized_params = normalize_response(parameters)
    >>> normalized_params[len(normalized_params) // 2]
    tensor(1.)
    """
    # Cumulatively sum the parameters and prepend a zero (padding)
    p = ch.nn.functional.pad(ch.cumsum(parameters, 0), (1, 0), mode='constant')
    # Exponentiate the cumulative sum to adjust the scale
    p = ch.exp(p)
    # Find the midpoint value and normalize the sequence so the midpoint becomes 1
    x_mid = p[len(p) // 2].item()
    p = p / x_mid / 2  # Adjust so the midpoint is 1, and scale down by a factor of 2
    return p
