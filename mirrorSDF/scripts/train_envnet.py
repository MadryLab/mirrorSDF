import warnings
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch as ch
import torch.nn.functional as F
import wandb
from torch.utils.benchmark import Timer
from tqdm import tqdm

from ..config import MirrorSDFConfig
from ..config.training import TrainingConfig
from ..dataset import MirrorSDFDataset, MirrorSDFDatasetSplit
from ..dataset.batch import Batch
from ..dataset.diffuse_irradiance import DiffuseIrradianceDataset
from ..dataset.loader_wrapper import LoaderWrapper
from ..ml_models import MirrorSDFModel
from ..utils.cli import create_cli_and_parse
from ..utils.geometry import cartesian_to_spherical
from ..utils.training import (
    optimizer_from_config, scheduler_from_config, MetricsTracker, cycle, render_nerf_spherical_view)


def train_iter(model: MirrorSDFModel, batch: Batch, optimizer: ch.optim.Optimizer,
               scheduler: ch.optim.lr_scheduler.LRScheduler, dataset: MirrorSDFDataset,
               diffuse_irradiance_iterator, metrics: MetricsTracker, config: MirrorSDFConfig,
               training_config: TrainingConfig):
    model.train()

    for i in range(training_config.gradient_accumulation):

        # Part 1: Specular

        with ch.no_grad():
            model.background.eval()
            coords = ch.rand(batch['size'], 3, device=batch['device']) * 2 - 1
            # We do not want to be under or exactly on the mirror
            coords[:, -1].abs_().add_(1e-5)
            directions = F.normalize(ch.randn(batch['size'], 3, device=batch['device']), dim=-1)
            batch['origin'] = coords
            batch['direction'] = directions
            batch['env_id'].random_().remainder_(config.dataset.num_environments)
            LoaderWrapper.precompute_useful_batch_quantities(batch)
            log_radiance_specular, depth = model.background.render(batch, config.rendering_eval.background_spp)

            if ch.any(ch.isnan(log_radiance_specular)):
                warnings.warn("NaN in prediction skipping iteration")
                return

        model.train()

        log_prediction_specular, prediction_diffuse = model.envnet(batch['origin'],
                                                                   batch['env_id'],
                                                                   batch['direction'][:, None],
                                                                   None)

        loss_specular = F.mse_loss(log_prediction_specular, log_radiance_specular[:, None])

        # Part 1: Diffuse

        coords, normals, env_ids, log_radiance_diffuse = [x.to(batch['device'])
                                                          for x in next(diffuse_irradiance_iterator)]
        _, log_prediction_diffuse = model.envnet(coords, env_ids, None, normals[:, None])

        loss_diffuse = F.mse_loss(log_prediction_diffuse[:, 0], log_radiance_diffuse)
        loss = loss_diffuse + loss_specular

        scaled_loss = loss

        if training_config.gradient_accumulation > 1:
            scaled_loss = scaled_loss / training_config.gradient_accumulation

        scaled_loss.backward()
        metrics.log('loss', loss)
        metrics.log('loss_specular', loss_specular)
        metrics.log('loss_diffuse', loss_diffuse)
        metrics.log('lr', ch.tensor(optimizer.param_groups[0]['lr']))

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


def visualize_sampled_light_sources(batch: 'Batch', config: 'MirrorSDFConfig', model: 'MirrorSDFModel',
                                    dataset: 'MirrorSDFDataset', env_id: int, height: float = 0.5) -> np.ndarray:
    image, _ = render_nerf_spherical_view(batch, config, model.background, dataset, env_id=env_id,
                                          height=height)

    plt.ioff()
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)

    p = model.envnet.point_lights_loc.clone()[env_id]
    p[:, 2] -= height
    theta, phi = cartesian_to_spherical(p).unbind(dim=-1)
    plt.scatter(phi.data.cpu().numpy(), theta.data.cpu(), s=1, color='red', alpha=0.5)
    plt.imshow(image, extent=(-np.pi, np.pi, np.pi / 2, 0))
    ax.set_aspect(2)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    return image_from_plot


def validation_iter(model: MirrorSDFModel, dataset: MirrorSDFDataset, batch: Batch, config: MirrorSDFConfig):
    if not config.logging.use_wandb:
        return  # No need to render if we are not saving the render

    model.eval()

    origins = ch.zeros(1, 3, device=batch['origin'].device)
    origins[:, 2] = 0.5
    render_resolution = config.validation.render_resolution
    phis = ch.linspace(0, 2 * ch.pi, render_resolution)
    theta = ch.linspace(0, ch.pi, render_resolution)
    directions = ch.ones(render_resolution, render_resolution, 3)
    phis, thetas = ch.meshgrid(phis, theta, indexing='xy')
    directions[:, :, 0] = np.sin(thetas) * np.cos(phis)
    directions[:, :, 1] = np.sin(thetas) * np.sin(phis)
    directions[:, :, 2] = np.cos(thetas)
    directions = directions.reshape(-1, 3)

    to_log = {}

    for env_id in range(config.dataset.num_environments):

        predictions_rgb = []
        predictions_rgb_log = []
        predictions_rgb_diffuse = []
        for i, db in enumerate(tqdm(ch.split(directions, batch['size']), desc='Render Validation')):
            new_batch: Batch = {**batch}
            db = db.cuda()
            new_batch['size'] = db.shape[0]
            new_batch['origin'] = origins.expand(db.shape[0], -1)
            new_batch['direction'] = db
            new_batch['env_id'] = ch.ones(db.shape[0]).long().cuda() * 0 + env_id

            LoaderWrapper.precompute_useful_batch_quantities(new_batch)

            rgb_log_specular, rgb_log_diffuse = model.envnet(new_batch['origin'], new_batch['env_id'],
                                                             new_batch['direction'][:, None],
                                                             new_batch['direction'][:, None])
            rgb_linear_specular = ch.exp(rgb_log_specular)
            rgb_linear_specular = (rgb_linear_specular ** config.validation.visualize_gamma[0]
                                   * config.validation.visualize_gamma[1])
            rgb_diffuse = (ch.exp(rgb_log_diffuse) ** config.validation.visualize_gamma[0]
                           * config.validation.visualize_gamma[1])

            predictions_rgb.append(rgb_linear_specular.data.cpu().numpy())
            predictions_rgb_diffuse.append(rgb_diffuse.data.cpu().numpy())
            predictions_rgb_log.append(rgb_log_specular.data.cpu().numpy())

        predictions_rgb = np.concatenate(predictions_rgb, 0)
        predictions_rgb_diffuse = np.concatenate(predictions_rgb_diffuse, 0)
        image_diffuse = predictions_rgb_diffuse.reshape(render_resolution, render_resolution, 3)
        image_specular = predictions_rgb.reshape(render_resolution, render_resolution, 3)

        predictions_rgb_log = np.concatenate(predictions_rgb_log, 0)
        predictions_rgb_log -= predictions_rgb_log.min()
        predictions_rgb_log /= predictions_rgb_log.max()
        image_log = predictions_rgb_log.reshape(render_resolution, render_resolution, 3)

        to_log[f'view_env_{env_id}'] = wandb.Image(
            np.concatenate([image_specular, image_diffuse, image_log], axis=1) * 255)

    wandb.log(to_log)


def run(config: MirrorSDFConfig):
    dataset = MirrorSDFDataset.from_config(config)

    environment_split = MirrorSDFDatasetSplit(dataset, config.dataset.env_memmap)
    model = MirrorSDFModel.from_config(config).cuda()

    print("Loading weights from the background nerf model")
    model.background.load_state_dict(ch.load(path.join(config.logging.output_folder,
                                                       config.logging.background_checkpoint_file)))

    envnet_training_config = config.envnet_training

    device = ch.device('cuda:0')

    train_data_loader = environment_split.create_loader(envnet_training_config,
                                                        shuffle=False,
                                                        device=device)

    diffuse_irradiance_loader = DiffuseIrradianceDataset.from_config(config).create_loader(config.envnet_training,
                                                                                           shuffle=True)
    diffuse_irradiance_loader = cycle(diffuse_irradiance_loader)
    diffuse_irradiance_iterator = iter(diffuse_irradiance_loader)

    optimizer = optimizer_from_config(model.envnet, envnet_training_config)
    scheduler = scheduler_from_config(optimizer, envnet_training_config)

    metrics = MetricsTracker()

    if config.logging.use_wandb:
        wandb.init(config=config.to_dict())

    num_iterations = envnet_training_config.num_iterations

    base_batch = next(iter(train_data_loader))

    model.envnet.pre_compute_light_sources(model.background, base_batch)

    source_light_images = {}
    for env_id in range(config.dataset.num_environments):
        source_light_images[f'point_sources_env{env_id}'] = wandb.Image(visualize_sampled_light_sources(
            base_batch, config, model, dataset, env_id))

    wandb.log(source_light_images)

    if config.logging.use_wandb:
        wandb.run.summary[f"throughput_bs{base_batch['size']}"] = (
                base_batch['size'] * 1000 / benchmark_model(model, base_batch)
        )

    for iteration in tqdm(range(num_iterations)):
        train_iter(model, base_batch, optimizer, scheduler, dataset, diffuse_irradiance_iterator,
                   metrics, config, envnet_training_config)

        if config.logging.use_wandb and ((iteration + 1) % config.logging.log_every_iter == 0):
            metrics.report()

        if (iteration + 1) % config.validation.validate_every_iter == 0:
            validation_iter(model, dataset, base_batch, config)
            if config.logging.output_folder:
                ch.save(model.envnet.state_dict(), path.join(config.logging.output_folder,
                                                             config.logging.envnet_checkpoint_file))


def benchmark_model(model: MirrorSDFModel, batch: Batch) -> float:
    print("Benchmarking model")
    timer = Timer(
        stmt='with ch.inference_mode(): m(batch["origin"], batch["env_id"], batch["direction"][:, None], batch["direction"][:, None])',
        globals={'m': ch.compile(model.envnet),
                 'batch': batch,
                 'ch': ch})
    ## warump
    ch.backends.cudnn.benchmark = True
    timer.timeit(1000)
    return timer.timeit(10000).mean * 1e3


def main():
    config = create_cli_and_parse("Train the envnet approximation of an already trained background model")
    run(config)


if __name__ == "__main__":
    main()
