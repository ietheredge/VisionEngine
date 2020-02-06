import numpy as np

# from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


@tf_export('guppies.load_data')
def load_data(path='./raw/guppies.npz'):
    # eventually we need to host these datasets online somewhere, not locally,
    # e.g.:
    # origin_folder = 'https://storage.googledrive.com/ietheredge/datasets/'
    # path = get_file(path,
    #                 origin=origin_folder + 'guppies.npz',
    #                 file_hash='8a61469f7ea1b54cbae51d4f78837e45')
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)
