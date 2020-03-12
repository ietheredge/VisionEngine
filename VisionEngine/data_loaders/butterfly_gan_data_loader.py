"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_data_loader import BaseDataLoader

from VisionEngine.datasets import butterflies, generated_butterflies
import numpy as np
import pathlib

class ButterflyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super( ButterflyDataLoader, self).__init__(config)

        if self.config.data_loader.use_real is True:
            if self.config.data_loader.use_generated is True:
                self.data_dir = pathlib.Path('VisionEngine/data_loaders/datasets/butterflies')
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError


    def get_train_data(self):
        def get_label(path):
            parts = tf.strings.split(path, os.path.sep)
            return parts

        def preprocess_input(path):
            file = tf.io.read_file(path)
            img = tf.image.decode_png(file, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            return img, img

        def prepare_for_training(ds, cache=self.config.data_loader.cache, shuffle_buffer_size=1000):
            if cache:
                if isinstance(cache, str):
                    ds = ds.cache(cache)
                else:
                    ds = ds.cache()
            
            if self.config.trainer.shuffle:
                ds = ds.shuffle(buffer_size=shuffle_buffer_size)
            
            ds = ds.repeat()
            ds = ds.batch(self.config.trainer.batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return ds

        self.list_data = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))

        ds = list_data.map(preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = prepare_for_training(ds)

        return train_ds

    def get_test_data(self):
        def preprocess_input(path):
            img = tf.image.decode_png(path, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            return img, img, labels

        list_data = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))
        ds = list_data.map(preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds
