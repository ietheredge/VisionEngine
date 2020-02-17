import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


@tf_export('butterflies.load_data')
def load_data(path='butterflies.npz'):
    origin_folder = 'https://owncloud.gwdg.de/index.php/s/7YvzorWb4QHCTMW/download'
    path = get_file(path,
                    origin=origin_folder,
                    cache_subdir='',
                    cache_dir='.',
                    file_hash='8a61469f7ea1b54cbae51d4f78837e458a61469f7ea1b54cbae51d4f78837e45')
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)
