from os import path
from typing import Tuple, TYPE_CHECKING

import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm

from ..dataset import MirrorSDFDataset, MirrorSDFDatasetSplit

if TYPE_CHECKING:
    from ..config import MirrorSDFConfig
    from ..config.training import TrainingConfig
    from ..ml_models import NerfBackground


class DiffuseIrradianceDataset(Dataset):
    """
    A dataset class for storing and accessing diffuse irradiance simulation data.

    This consists of 3D coordinates,
    surface normals, environment identifiers, and the logarithm of radiance values.
    It uses a memory-mapped file for efficient data storage and retrieval.

    Attributes
    ----------
    memmap_filename : str
        The file path of the memory-mapped file used to store the dataset.
    data : np.memmap
        The memory-mapped array holding the dataset.

    Methods
    -------
    __init__(memmap_filename: str)
        Initializes the dataset by loading the memory-mapped file.
    __len__()
        Returns the total number of entries in the dataset.
    __getitem__(ix: int)
        Retrieves the data at the specified index.

    create_memmap(config: MirrorSDFConfig, num_viewpoints: int) -> np.memmap
        Creates and returns a new empty memory-mapped file for the dataset.
    load_background_model(config: MirrorSDFConfig) -> ch.nn.Module
        Loads the reference background model from the specified checkpoint.
    generate_from_background_model(config: MirrorSDFConfig, device: ch.device)
        Generates the dataset using the background model and stores it in the memory-mapped file.
    """
    storage_dtype = np.dtype([
        ('coord', (np.float32, 3)),
        ('normal', (np.float32, 3)),
        ('env_id', np.uint16),
        ('log_radiance', (np.float32, 3)),
    ])

    def __init__(self, memmap_filename: str):
        """
        Initializes the dataset object by loading the specified memory-mapped file.

        Parameters
        ----------
        memmap_filename : str
            The file path of the memory-mapped file used to store the dataset.
        """
        super().__init__()
        self.memmap_filename = memmap_filename
        self.data = np.lib.format.open_memmap(self.memmap_filename,
                                              dtype=self.storage_dtype, mode='r')

    def __len__(self):
        """Returns the total number of entries in the dataset."""
        return len(self.data)

    def __getitem__(self, ix: int) -> Tuple[ch.Tensor, ch.Tensor, ch.IntTensor, ch.Tensor]:
        return (
            ch.from_numpy(self.data['coord'][ix].copy()),
            ch.from_numpy(self.data['normal'][ix].copy()),
            ch.tensor(int(self.data['env_id'][ix].copy())).int(),
            ch.from_numpy(self.data['log_radiance'][ix].copy())
        )

    @staticmethod
    def from_config(config: 'MirrorSDFConfig') -> 'DiffuseIrradianceDataset':
        filename = config.logging.get_full_path(config.logging.diffuse_irradiance_dataset)
        return DiffuseIrradianceDataset(filename)

    @staticmethod
    def create_memmap(config: 'MirrorSDFConfig', num_viewpoints: int) -> np.memmap:
        """
        Creates a memory-mapped file for storing the dataset.

        Parameters
        ----------
        config : MirrorSDFConfig
            The configuration object containing paths and settings for the simulation.
        num_viewpoints : int
            The total number of viewpoints to be stored in the dataset.

        Returns
        -------
        np.memmap
            The created memory-mapped file object.
        """
        dataset_path = config.logging.get_full_path(config.logging.diffuse_irradiance_dataset)
        gen_config = config.diffuse_irradiance

        dataset_shape = (num_viewpoints * gen_config.num_normals_per_viewpoint,)
        dataset_memmap = np.lib.format.open_memmap(dataset_path,
                                                   shape=dataset_shape,
                                                   dtype=DiffuseIrradianceDataset.storage_dtype,
                                                   mode='w+')
        return dataset_memmap

    @staticmethod
    def load_background_model(config: 'MirrorSDFConfig') -> 'NerfBackground':
        """
        Loads the background model from a checkpoint.

        Parameters
        ----------
        config : MirrorSDFConfig
            The configuration object specifying the checkpoint file path and architecture

        Returns
        -------
        NerfBackground
            The loaded background model.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        ValueError
            If the checkpoint does not match the keys expected from the current config file
        """
        # Dynamic import to avoid cyclic dependencies
        from ..ml_models import MirrorSDFModel

        full_model = MirrorSDFModel.from_config(config)

        bg_model_path = config.logging.get_full_path(config.logging.background_checkpoint_file)

        if not path.exists(bg_model_path):
            raise FileNotFoundError(f"Background model checkpoint not available: {bg_model_path}")

        try:
            full_model.background.load_state_dict(ch.load(bg_model_path))
        except KeyError:
            raise ValueError("The checkpoint for background model doesn't match the current configuration")

        return full_model.background

    def create_loader(self, config: 'TrainingConfig', shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=config.batch_size, shuffle=shuffle,
                          pin_memory=True,
                          num_workers=config.num_workers)

    @staticmethod
    def generate_from_background_model(config: 'MirrorSDFConfig', device: ch.device = ch.device('cuda:0')):
        """
        Generates dataset entries by sampling the background model and saves them to the memory-mapped file.

        Parameters
        ----------
        config : MirrorSDFConfig
            The configuration object containing simulation settings and paths.
        device : torch.device, optional
            The device on which to perform computations. Defaults to 'cuda:0'.
        """
        model = DiffuseIrradianceDataset.load_background_model(config)
        model = model.to(device)

        dataset_config = config.diffuse_irradiance
        num_viewpoints = config.dataset.num_environments * dataset_config.num_viewpoints_per_env

        dataset_mmap = DiffuseIrradianceDataset.create_memmap(config, num_viewpoints)
        dataset_mmap = dataset_mmap.reshape(num_viewpoints, dataset_config.num_normals_per_viewpoint)

        dataset = MirrorSDFDataset.from_config(config)
        environment_split = MirrorSDFDatasetSplit(dataset, config.dataset.env_memmap)
        train_data_loader = environment_split.create_loader(config.envnet_training, shuffle=False, device=device)
        batch = next(iter(train_data_loader))

        for i in tqdm(range(num_viewpoints)):
            with ch.inference_mode():
                model.eval()
                coords = ch.rand(1, 3, device=batch['device']) * 2 - 1
                # We do not want to be under or exactly on the mirror
                coords[:, -1].abs_().add_(1e-5)
                normals = ch.randn(1, dataset_config.num_normals_per_viewpoint, 3, device=batch['device'])
                normals = F.normalize(normals, p=2, dim=-1)
                env_ids = ch.randint(0, config.dataset.num_environments, size=(1,),
                                     device=batch['device'])
                result = None
                for k in range(dataset_config.num_sampling_rounds):
                    diffuse_linear = model.compute_diffuse_irradiance(coords, normals, env_ids, batch,
                                                                      dataset_config.random_normals_per_round,
                                                                      config.rendering_eval.background_spp,
                                                                      log_space=False)

                    if result is None:
                        result = diffuse_linear
                    else:
                        result += diffuse_linear

                result = result / dataset_config.num_sampling_rounds
                result = result.clamp(1e-10, ch.inf)
                # We want to train the model in log space
                result = ch.log(result)

                dataset_mmap['coord'][i] = coords.cpu().numpy()
                dataset_mmap['normal'][i] = normals.cpu().numpy()
                dataset_mmap['env_id'][i] = env_ids.cpu().numpy()
                dataset_mmap['log_radiance'][i] = result.cpu().numpy()
