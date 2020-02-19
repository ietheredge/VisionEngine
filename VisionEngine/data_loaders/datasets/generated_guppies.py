import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


@tf_export('generated_guppies.load_data')
def load_data(path='generated_guppies.npz'):
    origin_folder = 'https://owncloud.gwdg.de/index.php/s/3jjfDkLEwKclIk4/download'
    path = get_file(path,
                    origin=origin_folder,
                    cache_subdir='datasets',
                    cache_dir='data_loaders',
                    file_hash='a0b0c9fb2b641b07cc4e69b264638fa1f855d6602436b119d12e9950144e9f3d')
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        # x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (None, None)
