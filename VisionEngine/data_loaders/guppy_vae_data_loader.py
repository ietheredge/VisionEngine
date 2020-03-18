"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_data_loader import BaseDataLoader

# from VisionEngine.datasets import guppies, generated_guppies
import numpy as np
import tensorflow as tf
import pathlib
import os


class GuppyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(GuppyDataLoader, self).__init__(config)
        self.data_dir = pathlib.Path('VisionEngine/data_loaders/datasets/guppies')

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
                list_data = tf.data.Dataset.list_files(str(self.data_dir/'*/*'), shuffle=False, seed=42)
            else:
                list_data = tf.data.Dataset.list_files(str(self.data_dir/'*_*/*'), shuffle=False, seed=42) 
        else:
            list_data = tf.data.Dataset.list_files(str(self.data_dir/'[*0-9][*0-9]/*'), shuffle=False, seed=42)

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


# class GuppyDataLoader(BaseDataLoader):
#     def __init__(self, config):
#         super(GuppyDataLoader, self).__init__(config)

#         if self.config.data_loader.use_real is True:
#             (self.X_train, self.y_train), (_, _) = \
#                 guppies.load_data()

#         if self.config.data_loader.use_generated is True:
#             (self.X_trainG, self.y_trainG), (_, _) = \
#                 generated_guppies.load_data()

#         if self.config.data_loader.use_real is True:
#             if self.config.data_loader.use_generated is True:
#                 self.X_train = np.concatenate((self.X_train, self.X_trainG))
#                 self.y_train = np.concatenate((self.y_train, self.y_trainG))

#         else:
#             self.X_train = self.X_trainG
#             self.y_train = self.y_trainG

#     def get_train_data(self):
#         if self.config.model.input_shape[-1] == 3:
#             if self.X_train.shape[-1] == 4:
#                 # assumes a white background
#                 bg = np.ones(self.X_train.shape)[:, :, :, :3]
#                 r = ((1 - self.X_train[:, :, :, 3]) * bg[:, :, :, 0]) \
#                     + (self.X_train[:, :, :, 3] * self.X_train[:, :, :, 0])
#                 g = ((1 - self.X_train[:, :, :, 3]) * bg[:, :, :, 1]) \
#                     + (self.X_train[:, :, :, 3] * self.X_train[:, :, :, 1])
#                 b = ((1 - self.X_train[:, :, :, 3]) * bg[:, :, :, 2]) \
#                     + (self.X_train[:, :, :, 3] * self.X_train[:, :, :, 2])
#                 return np.stack([r, g, b], axis=3), self.y_train
#             else:
#                 return self.X_train, self.y_train

#         else:
#             return self.X_train, self.y_train

#     def get_test_data(self):
#         if self.config.model.input_shape[-1] == 3:
#             if self.X_train.shape[-1] == 4:
#                 # assumes a white background
#                 bg = np.ones(self.X_train.shape)[:, :, :, :3]
#                 r = ((1 - self.X_train[:, :, :, 3]) * bg[:, :, :, 0]) \
#                     + (self.X_train[:, :, :, 3] * self.X_train[:, :, :, 0])
#                 g = ((1 - self.X_train[:, :, :, 3]) * bg[:, :, :, 1]) \
#                     + (self.X_train[:, :, :, 3] * self.X_train[:, :, :, 1])
#                 b = ((1 - self.X_train[:, :, :, 3]) * bg[:, :, :, 2]) \
#                     + (self.X_train[:, :, :, 3] * self.X_train[:, :, :, 2])
#                 return np.stack([r, g, b], axis=3), self.y_train
#             else:
#                 return self.X_train, self.y_train

#         else:
#             return self.X_train, self.y_train

#     def get_plot_data(self):
#         return self.X_train, self.y_train