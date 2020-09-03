import json
from dotmap import DotMap
import os
import time


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configuration file
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)

    # convert the configuration dictionary to a namespace
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.callbacks.tensorboard_log_dir = os.path.join(
        config.callbacks.tensorboard_log_dir,
        config.exp.name,
        time.strftime("%Y-%j-%-H/", time.localtime()),
    )
    config.callbacks.checkpoint_dir = os.path.join(
        config.callbacks.checkpoint_dir,
        config.exp.name,
        time.strftime("%Y-%j-%-H/", time.localtime()),
    )

    return config
