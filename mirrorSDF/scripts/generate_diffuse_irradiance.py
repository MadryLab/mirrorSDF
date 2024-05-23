from ..dataset import DiffuseIrradianceDataset
from ..utils.cli import create_cli_and_parse


def main():
    config = create_cli_and_parse("Generate a a diffuse irradiance a dataset to train the EnvNet on")
    DiffuseIrradianceDataset.generate_from_background_model(config)


if __name__ == "__main__":
    main()
