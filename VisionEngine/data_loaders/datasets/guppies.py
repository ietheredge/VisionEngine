import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


@tf_export('guppies.load_data')
def load_data(path='guppies.npz'):
    origin_folder = 'https://owncloud.gwdg.de/index.php/s/bffyCQiqV5DBGfY/download'
    path = get_file(path,
                    origin=origin_folder,
                    cache_subdir='datasets',
                    cache_dir='VisionEngine/data_loaders/',
                    hash_algorithm='sha256',
                    file_hash='b93e110a13b3489e0340637ffdf2c752049ab40eac1c50e80a6610e3b7be3199')
    with np.load(path) as f:
        X, y = f['X'].astype('uint8'), f['y'].astype('str')

    return (X, y), (None, None)
