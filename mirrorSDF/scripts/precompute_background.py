import torch as ch
from tqdm.autonotebook import tqdm

from ..config import MirrorSDFConfig
from ..dataset import MirrorSDFDataset, MirrorSDFDatasetSplit
from ..dataset.precomputed_background import PrecomputedBackgroundDataset
from ..ml_models import MirrorSDFModel
from ..utils.cli import create_cli_and_parse


def run(config: 'MirrorSDFConfig'):
    model = MirrorSDFModel.from_config(config).cuda()
    background_weights = ch.load(config.logging.get_full_path(config.logging.background_checkpoint_file))
    model.background.load_state_dict(background_weights)
    dataset = MirrorSDFDataset.from_config(config)
    shape_split = MirrorSDFDatasetSplit(dataset, dataset.shape_memmap)

    # We set the proper batch size
    config.background_training.batch_size = config.validation.batch_size
    shape_loader = shape_split.create_loader(config.background_training,
                                             shuffle=False, device=ch.device('cuda:0'))
    precomputed_bg_dataset = PrecomputedBackgroundDataset.from_config(config, create=True,
                                                                      num_rows=len(shape_split))

    # Settings for max speed inference
    ch.backends.cudnn.benchmark = True
    ch.set_float32_matmul_precision('high')
    render = ch.compile(model.background.render)

    i = 0
    model.eval()
    with ch.inference_mode():
        for batch in tqdm(shape_loader):
            bg, _ = render(batch, config.rendering_eval.background_spp)
            bg = ch.exp(bg).data.cpu().numpy()
            precomputed_bg_dataset.data[i: i + batch['size']]['predicted_linear'] = bg
            i = i + batch['size']


def main():
    config = create_cli_and_parse(
        "Pre-compute high quality predictions of the background model on the shape sub-dataset"
    )
    run(config)


if __name__ == "__main__":
    main()
