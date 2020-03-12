import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


@tf_export('generated_guppies.load_data')
def load_data(path='generated_guppies.npz'):
    origin_folder = 'https://owncloud.gwdg.de/index.php/s/3jjfDkLEwKclIk4/download'
    path = get_file(path,
                    origin=origin_folder,
                    cache_subdir='datasets',
                    cache_dir='VisionEngine/data_loaders/',
                    hash_algorithm='sha256',
                    file_hash='b27e2fbbf67399aa2ba4b2dbaebce5a6318f9ec03bc023299ec95df82b7c0ab2')
    with np.load(path) as f:
        X, y = f['X'].astype('uint8'), f['y'].astype('str')

    return (X, y), (None, None)
