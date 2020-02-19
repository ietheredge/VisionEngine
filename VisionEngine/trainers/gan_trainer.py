"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from base.base_trainer import BaseTrain

import tensorflow as tf
import numpy as np
import os


class GANTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(VAETrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                      '%s-{epoch:02d}-{loss:.2f}.hdf5' %
                                      self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks
                .checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
                save_freq=self.config.callbacks.save_freq,
            )
        )

        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
                write_images=self.config.callbacks.tensorboard_write_images,
                histogram_freq=self.config.callbacks.tensorboard_histogram_freq,
            )
        )

        if self.config.trainer.use_lr_scheduler is True:
            lr_epochs = 10 ** np.linspace(
                self.config.trainer.lr_start,
                self.config.trainer.lr_stop,
                self.config.trainer.num_epochs
                )

            self.callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(
                    lambda i: lr_epochs[i]
                )
            )

        if self.config.callbacks.use_early_stopping is True:
            self.callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    min_delta=self.config.trainer.min_delta,
                    patience=self.config.trainer.patience
                )
            )

    def train(self):
        # history = self.model.fit(
        self.model.fit(
            self.data[0], self.data[0],
            validation_split=self.config.trainer.validation_split,
            batch_size=self.config.trainer.batch_size,
            epochs=self.config.trainer.num_epochs,
            callbacks=self.callbacks,
            # verbose=self.config.trainer.verbose_training,
            # shuffle=self.config.trainer.shuffle
        )

        # for metric in self.config.callbacks.callback_metrics:
        #     self.loss.extend(history.history[metric])
