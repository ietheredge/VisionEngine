import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


@tf_export('generated_guppies.load_data')
def load_data(path='generated_guppies.npz'):
    origin_folder = 'https://owncloud.gwdg.de/index.php/s/3jjfDkLEwKclIk4/download'
    path = get_file(path,
                    origin=origin_folder,
                    cache_subdir='',
                    cache_dir='.',
                    file_hash='e06beedcb735023e16829fe92c2daacdfe7174f2543157803b66d22b5a3308cd')
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)
