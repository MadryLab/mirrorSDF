import numpy as np
import torch as ch
from torch.utils.data import Sampler


# noinspection PyTypeChecker
class SamplerWithReplacement(Sampler):
    """
    A fast random sampler for large datasets where computing a
    permutation is computationally prohibitive and sampling with replacement
    is the only alternative.

    This sampler generates random indices for selecting batches from a dataset, 
    intended to be used with a DataLoader for efficient data loading in stochastic
    training processes.

    Attributes
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to sample from.
    batch_size : int
        The size of each batch to sample.

    Methods
    -------
    __len__() -> int:
        Returns the number of batches available in the dataset.
    __iter__() -> Iterator[torch.Tensor]:
        An iterator that yields indices of the dataset elements to form a batch.
    """

    def __init__(self, dataset: ch.utils.data.Dataset, batch_size: int) -> None:
        """
        Initializes the FastRandomSampler with a dataset and batch size.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset from which to draw samples.
        batch_size : int
            The number of samples to include in each batch.
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self) -> int:
        """
        Returns the number of batches that can be sampled from the dataset.

        Returns
        -------
        int
            The number of available batches.
        """
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        """
        Provides an iterator that yields random indices for batch sampling.

        Yields
        ------
        torch.Tensor
            A tensor containing randomly selected indices for a batch.
        """
        total_samples = len(self.dataset)
        for _ in range(len(self)):
            yield ch.from_numpy(np.random.choice(total_samples, size=self.batch_size, replace=True)).to(dtype=ch.long)
