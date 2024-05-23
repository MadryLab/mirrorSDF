import argparse
from pprint import pprint
from typing import List, get_type_hints

from ..config import MirrorSDFConfig


def apply_cli_overrides(overrides: List[str], config: MirrorSDFConfig):
    for item in overrides:
        key_value = item.split('=')
        if len(key_value) == 2:
            key, value = key_value
            parts = key.split('.')
            obj = config
            try:
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                field_name = parts[-1]
                field_type = get_type_hints(obj.__class__).get(field_name, str)
                if field_type == bool:
                    value = value in ["True", "1", "true"]
                else:
                    try:
                        value = field_type(value)
                    except:
                        pass
                setattr(obj, field_name, value)
            except AttributeError:
                raise ValueError("Could not set", item)
        else:
            print(f"Skipping invalid format for --set argument: {item}")


def create_cli_and_parse(cli_description: str) -> MirrorSDFConfig:
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument("--config", "-c", type=str, help="Path to the configuration file.", required=True)
    parser.add_argument("--override", "-o", nargs='*', help="override the config with the syntax a.b.c=value")

    args = parser.parse_args()

    config = MirrorSDFConfig.from_disk(args.config)

    if args.override:
        apply_cli_overrides(args.override, config)

    pprint(config.to_dict())

    return config
