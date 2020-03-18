"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_data_loader import BaseDataLoader

import numpy as np
import tensorflow as tf
import pathlib
import os

class ButterflyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super( ButterflyDataLoader, self).__init__(config)
        self.data_dir = pathlib.Path('VisionEngine/data_loaders/datasets/butterflies')

    def get_train_data(self):
        def alpha_blend_decoded_png(file):
            # alpha blending with a white background 
            bg = tf.ones((256,256,3))  # if you want black change to tf.zeros
            r = ((1 - file[:, :, 3]) * bg[:, :, 0]) \
                    + (file[:, :, 3] * file[:, :, 0])
            g = ((1 - file[:, :, 3]) * bg[:, :, 1]) \
                    + (file[:, :, 3] * file[:, :, 1])
            b = ((1 - file[:, :, 3]) * bg[:, :, 2]) \
                    + (file[:, :, 3] * file[:, :, 2])
            rgb = tf.stack([r, g, b], axis=2)
            return rgb

        def preprocess_input(path):
            file = tf.io.read_file(path)
            label = tf.strings.split(path, os.path.sep)[-2]
            img = tf.image.decode_png(file, channels=0)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = alpha_blend_decoded_png(img)
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

        if self.config.data_loader.use_real is True:
            if self.config.data_loader.use_generated is True:
                raise NotImplementedError
            else:
                list_data = tf.data.Dataset.list_files(str(self.data_dir/'*/*'), shuffle=False, seed=42) 
        else:
            raise NotImplementedError

        ds = list_data.map(preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = prepare_for_training(ds)

        return train_ds

    def get_test_data(self):
        def alpha_blend_decoded_png(file):
            # alpha blending with a white background 
            bg = tf.ones((256,256,3))  # if you want black change to tf.zeros
            r = ((1 - file[:, :, 3]) * bg[:, :, 0]) \
                    + (file[:, :, 3] * file[:, :, 0])
            g = ((1 - file[:, :, 3]) * bg[:, :, 1]) \
                    + (file[:, :, 3] * file[:, :, 1])
            b = ((1 - file[:, :, 3]) * bg[:, :, 2]) \
                    + (file[:, :, 3] * file[:, :, 2])
            rgb = tf.stack([r, g, b], axis=2)
            return rgb

        def preprocess_input(path):
            file = tf.io.read_file(path)
            label = tf.strings.split(path, os.path.sep)[-2]
            img = tf.image.decode_png(file, channels=0)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = alpha_blend_decoded_png(img)
            label = tf.strings.split(path, os.path.sep)[-2]
            return img, label

        def prepare_for_testing(ds, cache=self.config.data_loader.cache, shuffle_buffer_size=1000):
            if cache:
                if isinstance(cache, str):
                    ds = ds.cache(cache)
                else:
                    ds = ds.cache()

            ds = ds.batch(self.config.trainer.batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return ds

        if self.config.data_loader.use_real is True:
            if self.config.data_loader.use_generated is True:
                list_data = tf.data.Dataset.list_files(str(self.data_dir/'*/*'), shuffle=False, seed=42)
            else:
                list_data = tf.data.Dataset.list_files(str(self.data_dir/'*_*/*_*'), shuffle=False, seed=42) 
        else:
            list_data = tf.data.Dataset.list_files(str(self.data_dir/'[!a-z]/[!a-z]*'), shuffle=False, seed=42)
        
        ds = list_data.map(preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = prepare_for_testing(ds)

        return test_ds

    def get_plot_data(self):
        
        def preprocess_input(path):
            file = tf.io.read_file(path)
            img = tf.image.decode_png(file)
            label = tf.strings.split(path, os.path.sep)[-2]
            return img, label

        def prepare_for_testing(ds, cache=self.config.data_loader.cache, shuffle_buffer_size=1000):
            if cache:
                if isinstance(cache, str):
                    ds = ds.cache(cache)
                else:
                    ds = ds.cache()

            return ds

        if self.config.data_loader.use_real is True:
            if self.config.data_loader.use_generated is True:
                list_data = tf.data.Dataset.list_files(str(self.data_dir/'*/*'), shuffle=False, seed=42)
            else:
                list_data = tf.data.Dataset.list_files(str(self.data_dir/'*_*/*_*'), shuffle=False, seed=42) 
        else:
            list_data = tf.data.Dataset.list_files(str(self.data_dir/'[!a-z]/[!a-z]*'), shuffle=False, seed=42)
        
        ds = list_data.map(preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        plot_ds = prepare_for_testing(ds)
        return plot_ds