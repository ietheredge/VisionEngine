import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


@tf_export('generated_butterflies.load_data')
def load_data(path='generated_butterflies.npz'):
    origin_folder = 'https://owncloud.gwdg.de/index.php/s/3jjfDkLEwKclIk4/download'
    path = get_file(path,
                    origin=origin_folder,
                    cache_subdir='datasets',
                    cache_dir='data_loaders',
                    file_hash='9ebe6f56fb4d2c06e36f4c737ea6d2d10b5b9be38303e1d2a140313609e39a45')
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        # x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (None, None)
