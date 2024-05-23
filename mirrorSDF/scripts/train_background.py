import argparse
import warnings
from os import path
from pprint import pprint
from typing import Iterator

import torch as ch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from ..config import MirrorSDFConfig
from ..config.training import TrainingConfig
from ..dataset import MirrorSDFDataset, MirrorSDFDatasetSplit
from ..dataset.batch import Batch
from ..ml_models import MirrorSDFModel
from ..utils.cli import apply_cli_overrides
from ..utils.training import (optimizer_from_config, scheduler_from_config,
                              cycle, MetricsTracker, render_nerf_spherical_view)


def train_iter(model: MirrorSDFModel, iterator: Iterator[Batch], optimizer: ch.optim.Optimizer,
               scheduler: ch.optim.lr_scheduler.LRScheduler,
               metrics: MetricsTracker, config: MirrorSDFConfig, training_config: TrainingConfig):
    model.train()

    for i in range(training_config.gradient_accumulation):
        batch = next(iterator)
        prediction = model.background.render(batch, config.rendering_train.background_spp)[0]

        if ch.any(ch.isnan(prediction)):
            warnings.warn("NaN in prediction skipping batch")
            continue

        target = batch['log_measurement']

        loss = F.mse_loss(prediction, target)
        scaled_loss = loss

        if training_config.gradient_accumulation > 1:
            scaled_loss = scaled_loss / training_config.gradient_accumulation
        scaled_loss.backward()
        metrics.log('loss', loss)
        metrics.log('lr', ch.tensor(optimizer.param_groups[0]['lr']))

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


def validation_iter(model: MirrorSDFModel, dataset: MirrorSDFDataset, batch: Batch, config: MirrorSDFConfig):
    if not config.logging.use_wandb:
        return  # No need to render if we are not saving the render

    model.eval()

    image, depth_image = render_nerf_spherical_view(batch, config, model.background, dataset)

    wandb.log({
        'view_360_rgb': wandb.Image(image * 255),
        'view_360_depth': wandb.Image(depth_image * 255),
    })


def run(config: MirrorSDFConfig):
    dataset = MirrorSDFDataset.from_config(config)

    environment_split = MirrorSDFDatasetSplit(dataset, config.dataset.env_memmap)
    model = MirrorSDFModel.from_config(config).cuda()
    environment_training_config = config.background_training
    train_data_loader = environment_split.create_loader(environment_training_config,
                                                        shuffle=True,
                                                        device=ch.device('cuda:0'))
    optimizer = optimizer_from_config(model, environment_training_config)
    scheduler = scheduler_from_config(optimizer, environment_training_config)

    metrics = MetricsTracker()

    if config.logging.use_wandb:
        wandb.init(config=config.to_dict())

    num_iterations = environment_training_config.num_iterations

    iterator = iter(cycle(train_data_loader))
    for iteration in tqdm(range(num_iterations), desc='Train environment'):
        train_iter(model, iterator, optimizer, scheduler, metrics, config, environment_training_config)

        if config.logging.use_wandb and ((iteration + 1) % config.logging.log_every_iter == 0):
            metrics.report()

        if (iteration + 1) % config.validation.validate_every_iter == 0:
            validation_iter(model, dataset, next(iterator), config)
            if config.logging.output_folder:
                ch.save(model.background.state_dict(), path.join(config.logging.output_folder,
                                                                 config.logging.background_checkpoint_file))


def main():
    parser = argparse.ArgumentParser(description="Train an environment model")
    parser.add_argument("--config", "-c", type=str, help="Path to the configuration file.", required=True)
    parser.add_argument("--override", "-o", nargs='*', help="override the config with the syntax a.b.c=value")

    args = parser.parse_args()

    config = MirrorSDFConfig.from_disk(args.config)

    if args.override:
        apply_cli_overrides(args.override, config)

    pprint(config.to_dict())
    run(config)


if __name__ == "__main__":
    main()
