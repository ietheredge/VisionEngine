import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


@tf_export('guppies.load_data')
def load_data(path='guppies.npz'):
    origin_folder = 'https://owncloud.gwdg.de/index.php/s/bffyCQiqV5DBGfY/download'
    path = get_file(path,
                    origin=origin_folder,
                    cache_subdir='datasets',
                    cache_dir='data_loaders',
                    file_hash='c41616920521fc3b2ee5ec314bf5bf8249a2ce44bcf81d4f4e66ec0413fa38ee')
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        # x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (None, None)
