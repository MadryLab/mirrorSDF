from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch as ch
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from tqdm.autonotebook import tqdm

from ..dataset.loader_wrapper import LoaderWrapper

if TYPE_CHECKING:
    from ..ml_models import MirrorSDFModel
    from ..config.training import TrainingConfig
    from ..config import MirrorSDFConfig
    from ..dataset.batch import Batch
    from ..dataset import MirrorSDFDataset
    from ..ml_models.background.nerf_background import NerfBackground


def discard_nan_grads(tensor):
    return tensor.nan_to_num(0, 0, 0)



def optimizer_from_config(model: 'MirrorSDFModel', config: 'TrainingConfig') -> ch.optim.Optimizer:
    optimizer_ctr = getattr(ch.optim, config.optimizer_name)

    optimizer: ch.optim.Optimizer = optimizer_ctr(model.parameters(),
                                                  lr=config.learning_rate,
                                                  weight_decay=config.weight_decay)

    return optimizer


def scheduler_from_config(optimizer, config: 'TrainingConfig'):
    return MultiStepLR(optimizer, milestones=config.drop_lr_at, gamma=config.drop_lr_factor)


def cycle(iterable):
    """

    from: https://github.com/pytorch/pytorch/issues/23900
    Parameters
    ----------
    iterable

    Returns
    -------

    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)

    def log(self, metric_name: str, value: ch.Tensor):
        self.metrics[metric_name].append(value)

    def report(self):
        to_report = {
            k: ch.cat([x.ravel() for x in v]).mean().item() for (k, v) in self.metrics.items()
        }

        wandb.log(to_report)
        self.metrics.clear()


def render_nerf_spherical_view(batch: 'Batch', config: 'MirrorSDFConfig',
                               model: 'NerfBackground', dataset: 'MirrorSDFDataset', env_id: int = 0,
                               height: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    origins = ch.zeros(1, 3, device=batch['origin'].device)
    origins[:, 2] = height
    render_resolution = config.validation.render_resolution
    phis = ch.linspace(-ch.pi, ch.pi, render_resolution)
    theta = ch.linspace(0, ch.pi / 2, render_resolution // 2)
    directions = ch.ones(render_resolution // 2, render_resolution, 3)
    phis, thetas = ch.meshgrid(phis, theta, indexing='xy')
    directions[:, :, 0] = np.sin(thetas) * np.cos(phis)
    directions[:, :, 1] = np.sin(thetas) * np.sin(phis)
    directions[:, :, 2] = np.cos(thetas)
    directions = directions.reshape(-1, 3)
    predictions_rgb = []
    predictions_depth = []

    for i, db in enumerate(tqdm(ch.split(directions, config.validation.batch_size), desc='Render Validation')):
        new_batch: Batch = {**batch}
        db = db.cuda()
        new_batch['size'] = db.shape[0]
        new_batch['origin'] = origins.expand(db.shape[0], -1)
        new_batch['direction'] = db
        new_batch['mirror_vertices'] = new_batch['mirror_vertices'] * 1e-7
        new_batch['env_id'] = ch.zeros(db.shape[0]).long().cuda() + env_id

        LoaderWrapper.precompute_useful_batch_quantities(new_batch)

        log_rgb, depth = model.render(new_batch, config.rendering_eval.background_spp)
        rgb = ch.exp(log_rgb)
        rgb = rgb ** config.validation.visualize_gamma[0] * config.validation.visualize_gamma[1]
        predictions_rgb.append(rgb.data.cpu().numpy())
        predictions_depth.append(depth.data.cpu().numpy())
    predictions_rgb = np.concatenate(predictions_rgb, 0)
    predictions_depth = np.concatenate(predictions_depth, 0)
    image = predictions_rgb.reshape(render_resolution // 2, render_resolution, 3)
    depth_image = predictions_depth.reshape(render_resolution // 2, render_resolution, 1)
    wnf = dataset.world_scale_normalization_factor
    depth_image -= config.dataset.minimum_distance_to_env * wnf
    depth_image /= (config.dataset.maximum_distance_to_env - config.dataset.minimum_distance_to_env) * wnf

    return image, depth_image


def check_for_nans(model):
    bad = False

    for param_name, param in model.named_parameters():
        if ch.any(~ch.isfinite(param.data)):
            bad = True
            print(f'{param_name} contained a non finite value')

        grad = param.grad
        if grad is None:
            continue

        if grad.is_sparse:
            grad = grad._values()

        if ch.any(~ch.isfinite(grad)):
            bad = True
            print(f'{param_name} gradients contained a non finite value')
    return bad
