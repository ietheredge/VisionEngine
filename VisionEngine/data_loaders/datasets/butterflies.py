import numpy as np
import os

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


def load_data(path="butterflies.zip"):
    origin_folder = "https://owncloud.gwdg.de/index.php/s/ObTXnWN9ll45EoV/download"
    path = get_file(
        path,
        origin=origin_folder,
        extract=True,
        archive_format="zip",
        cache_dir=os.path.join(
            os.getenv("VISIONENGINE_HOME"), "VisionEngine/data_loaders"
        ),
    )
