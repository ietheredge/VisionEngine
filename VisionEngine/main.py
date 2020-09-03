from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
from dotenv import load_dotenv
from pathlib import Path
import sys
import tensorflow as tf


def setup_env():
    env_path = Path(".") / ".env"
    load_dotenv(dotenv_path=env_path)


def main():

    # capture the config path from the run arguments
    # then process the json configuration fill
    try:
        args = get_args()
        config = process_config(args.config)

        # create the experiments dirs
        create_dirs(
            [
                config.callbacks.tensorboard_log_dir,
                config.callbacks.checkpoint_dir,
            ]
        )

        print("Create the data generator.")
        data_loader = factory.create(
            "VisionEngine.data_loaders." + config.data_loader.name
        )(config)

        print("Create the model.")
        model = factory.create("VisionEngine.models." + config.model.name)(config)

        if config.model.loadckpt:
            print("loading model checkpoint")
            model.load(config.model.ckpt_path)

        print("Create the trainer")
        trainer = factory.create("VisionEngine.trainers." + config.trainer.name)(
            model.model, data_loader.get_train_data(), config
        )

        print("Start training the model.")
        trainer.train()

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    setup_env()
    main()
