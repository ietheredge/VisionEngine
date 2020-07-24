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
from skimage.util import random_noise


class ButterflyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super( ButterflyDataLoader, self).__init__(config)
        self.data_dir = pathlib.Path('VisionEngine/data_loaders/datasets/butterflies')

    def get_train_data(self):
        def alpha_blend_decoded_png(file):
            # alpha blending with a white background 
            bg = tf.ones((256,256,3))  # change to tf.zeros for a black bg
            r = ((1 - file[:, :, 3]) * bg[:, :, 0]) \
                    + (file[:, :, 3] * file[:, :, 0])
            g = ((1 - file[:, :, 3]) * bg[:, :, 1]) \
                    + (file[:, :, 3] * file[:, :, 1])
            b = ((1 - file[:, :, 3]) * bg[:, :, 2]) \
                    + (file[:, :, 3] * file[:, :, 2])
            rgb = tf.stack([r, g, b], axis=2)
            return rgb

        def preprocess_input(path):
            FILE = tf.io.read_file(path)
            label = tf.strings.split(path, os.path.sep)[-2]
            img = tf.image.decode_png(FILE, channels=0)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = alpha_blend_decoded_png(img)
            output_img = img
            if self.config.model.augment is True:
                img, output_img = self.random_jitter(img, output_img)

            if self.config.model.activation is 'tanh':
                 normalize(img, output_img)

            return img, output_img

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

        # TODO generalize this for a generic dataloader
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
            FILE = tf.io.read_file(path)
            label = tf.strings.split(path, os.path.sep)[-2]
            img = tf.image.decode_png(FILE, channels=0)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = alpha_blend_decoded_png(img)
            label = tf.strings.split(path, os.path.sep)[-2]
            if self.config.model.activation is 'tanh':
                img, _ = self.normalize(img, None)

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

        # TODO generalize this for a generic dataloader
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
            FILE = tf.io.read_file(path)
            img = tf.image.decode_png(FILE)
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

    @staticmethod
    def normalize(input_image, real_image):
        # normalize between [-1, 1] if using tanh activation
        input_image = (input_image / 0.5) - 1
        if real_image:
            real_image = (real_image / 0.5) - 1
            return input_image, real_image

        else:
            return input_image, None    

    @staticmethod
    def resize(input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width], 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        real_image = tf.image.resize(real_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    
    @staticmethod
    def random_crop(input_image, real_image, img_height, img_width):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image,
            size=[2, img_height, img_width, 3])

        return cropped_image[0], cropped_image[1]

    @tf.function()
    def random_jitter(self, input_image, real_image):
        input_image, real_image = self.resize(input_image, real_image, 384, 384)

        # randomly cropping to 256 x 256 x 3
        input_image, real_image = self.random_crop(input_image,
            real_image, self.config.model.input_shape[0], self.config.model.input_shape[1])

        if tf.random.uniform(()) > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image