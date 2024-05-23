from typing import TYPE_CHECKING, Optional

import numpy as np
import torch as ch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from ..config import MirrorSDFConfig

    pass


class PrecomputedBackgroundDataset(Dataset):
    """
    A dataset class for handling precomputed background values stored in a numpy array.

    This class facilitates the loading and handling of datasets containing precomputed
    background values, predicted from our background models. Since we optimize the
    background and the foreground separately, it makes sense to pre-compute the background
    ahead of time

    Attributes
    ----------
    storage_dtype : np.dtype
        The dtype of the numpy array used to store the dataset. It defines the structure
        of the stored data, including field names and their data types. This field
        is meant to be static and shall not be overwritten by users.
    data : np.ndarray
        The actual data of the dataset stored as a numpy array following the `storage_dtype`.

    Parameters
    ----------
    data : np.ndarray
        The precomputed background data to be loaded into the dataset.

    Raises
    ------
    ValueError
        If the provided data does not match the expected `storage_dtype` or if its shape is incorrect.

    Methods
    -------
    __len__():
        Returns the number of items in the dataset.
    __getitem__(index: int) -> ch.Tensor:
        Retrieves an item from the dataset by index and returns it as a PyTorch tensor.
    from_config(config: 'MirrorSDFConfig', create: bool = False, num_rows: Optional[int] = None) -> 'PrecomputedBackgroundDataset':
        A factory method to create an instance of `PrecomputedBackgroundDataset` from a configuration object,
        optionally creating a new dataset file.
    """

    storage_dtype = np.dtype([
        ('predicted_linear', (np.float32, 3)),
    ])

    def __init__(self, data: np.ndarray):
        """
        Initializes the dataset with precomputed background data.

        Parameters
        ----------
        data : np.ndarray
            The precomputed background data, expected to be a structured numpy array
            that matches `storage_dtype`.

        Raises
        ------
        ValueError
            If `data` does not have the expected `storage_dtype` or if its shape is not 1-dimensional.
        """
        super().__init__()
        if data.dtype != self.storage_dtype or len(data.shape) != 1:
            raise ValueError("Invalid data type")

        self.data = data

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> ch.Tensor:
        """
        Retrieves an item from the dataset by index.

        Parameters
        ----------
        index : int
            The index of the item to retrieve.

        Returns
        -------
        ch.Tensor
            The dataset item at the specified index, converted to a PyTorch tensor.
        """
        return ch.from_numpy(self.data['predicted_linear'][index].copy())

    @staticmethod
    def from_config(config: 'MirrorSDFConfig', create: bool = False,
                    num_rows: Optional[int] = None) -> 'PrecomputedBackgroundDataset':
        """
        Creates a `PrecomputedBackgroundDataset` instance from a configuration object.

        This method allows for the dataset to be loaded based of configuration file, with
        the option to create a new dataset when we want to (re)-generate it

        Parameters
        ----------
        config : MirrorSDFConfig
            The configuration object containing dataset parameters.
        create : bool, optional
            Whether to create a new dataset file if it does not exist. Default is False.
        num_rows : int, optional
            The number of rows for the new dataset if `create` is True. Required if `create` is True.

        Returns
        -------
        PrecomputedBackgroundDataset
            An instance of `PrecomputedBackgroundDataset` loaded or created according to the provided configuration.

        Raises
        ------
        ValueError
            If `num_rows` is None when `create` is True, or if `num_rows` is provided but `create` is False.
        """
        log_cfg = config.logging
        file_name = log_cfg.get_full_path(log_cfg.precomputed_background_values)
        shape = None
        mode = 'r'

        if create:
            if num_rows is None:
                raise ValueError("Cannot create without the number of rows needed")
            shape = (num_rows,)
            mode = 'w+'

        if (num_rows is not None) and (not create):
            raise ValueError("num_rows argument should only be used when creating a new dataset")

        memmap = np.lib.format.open_memmap(file_name,
                                           dtype=PrecomputedBackgroundDataset.storage_dtype,
                                           shape=shape, mode=mode)
        return PrecomputedBackgroundDataset(memmap)
