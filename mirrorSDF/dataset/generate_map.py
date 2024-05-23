import argparse

import torch as ch
import numpy as np

from ..optical_models import RGBImageSensor, Lens, Mirror
from ..dataset.preparation import process_image_first_stage, process_image_second_stage, get_image_list_sorted
from ..config import MirrorSDFConfig


def main(args):
    # Print the parsed arguments
    print(f"Config file path: {args.config_file}")
    print(f"Image index: {args.image_ix}")
    print(f"PyTorch device: {args.device}")

    device = ch.device(args.device)

    config = MirrorSDFConfig.from_disk(args.config_file)

    sensor = RGBImageSensor.from_disk(config.calibration_files.sensor)
    lens = Lens.from_disk(config.calibration_files.lens)
    mirror = Mirror.from_disk(config.calibration_files.mirror)

    dataset_cfg = config.dataset

    all_file_names = get_image_list_sorted(dataset_cfg.original_source, dataset_cfg.source_extension,
                                           dataset_cfg.num_bracketed_shots)

    result = process_image_first_stage(args.image_ix,
                                       all_file_names[args.image_ix],
                                       sensor, lens, mirror,
                                       dataset_cfg.black_point,
                                       dataset_cfg.white_point)
    pixels, loaded_image, origin, directions, shot_metadata, camera = result

    process_image_second_stage(shot_metadata, loaded_image, dataset_cfg.root, pixels,
                               origin, directions, mirror, np.array(dataset_cfg.object_center),
                               np.array(dataset_cfg.object_bounding_box), dataset_cfg.mirror_safety_padding,
                               dataset_cfg.marker_safety_padding, device=device)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Script to generate the slice of dataset corresponding to a given image"
    )

    # Add arguments
    parser.add_argument("-c", "--config_file", required=True, help="Path to the configuration file.")
    parser.add_argument("-i", "--image_ix", type=int, required=True, help="Image index to process.")
    parser.add_argument("-d", "--device", default="cpu",
                        help="PyTorch device to use (e.g., 'cpu', 'cuda', 'cuda:0'). Default is 'cpu'.")

    # Call the main function
    main(parser.parse_args())
