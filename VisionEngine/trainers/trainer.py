"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_trainer import BaseTrain

import tensorflow as tf
import numpy as np
import os


class KLWarmUp(tf.keras.callbacks.Callback):
    def __init__(
        self, n_iter=100, start=0.0, stop=1.0, n_cycle=4, ratio=0.5, n_latents=4
    ):

        self.frange = self.frange_cycle_linear(
            n_iter, start=start, stop=stop, n_cycle=n_cycle, ratio=ratio
        )
        self.epoch = 0
        self.n_latents = n_latents

    def on_epoch_end(self, *args, **kwargs):
        new_coef = self.frange[self.epoch]
        self.epoch += 1
        coefs = [
            self.model.get_layer(f"z_{i+1}").coef_kl for i in range(self.n_latents)
        ]

        for coef in coefs:
            coef.assign(new_coef)

    @staticmethod
    def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)  # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L


class Trainer(BaseTrain):
    def __init__(self, model, data, config):
        super(Trainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    os.getenv("VISIONENGINE_HOME"),
                    self.config.callbacks.checkpoint_dir,
                    "{}.hdf5".format(self.config.exp.name),
                ),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
                save_freq=self.config.callbacks.save_freq,
            )
        )

        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    os.getenv("VISIONENGINE_HOME"),
                    self.config.callbacks.tensorboard_log_dir,
                ),
                write_graph=self.config.callbacks.tensorboard_write_graph,
                write_images=self.config.callbacks.tensorboard_write_images,
                histogram_freq=self.config.callbacks.tensorboard_histogram_freq,
            )
        )

        if self.config.trainer.use_lr_scheduler is True:
            lr_epochs = 10 ** np.linspace(
                self.config.trainer.lr_start,
                self.config.trainer.lr_stop,
                self.config.trainer.num_epochs,
            )

            self.callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(lambda i: lr_epochs[i])
            )

        if self.config.trainer.use_early_stopping is True:
            self.callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    min_delta=self.config.trainer.min_delta,
                    patience=self.config.trainer.patience,
                    monitor=self.config.trainer.early_stopping_monitor,
                )
            )

        if self.config.trainer.use_kl_warmup is True:
            self.callbacks.append(
                KLWarmUp(
                    n_iter=self.config.trainer.kl_wu_n_iter,
                    start=self.config.trainer.kl_wu_start,
                    stop=self.config.trainer.kl_wu_stop,
                    n_cycle=self.config.trainer.kl_wu_n_cycle,
                    ratio=self.config.trainer.kl_wu_ratio,
                    n_latents=self.config.model.n_latents,
                )
            )

    def train(self):
        self.model.fit(
            self.data[0],
            epochs=self.config.trainer.num_epochs,
            callbacks=self.callbacks,
            validation_data=self.data[1],
            steps_per_epoch=int(
                (1 - self.config.data_loader.validation_split)
                * self.config.data_loader.n_samples
                / self.config.trainer.batch_size
                + 1
            ),
            validation_steps=int(
                self.config.data_loader.validation_split
                * self.config.data_loader.n_samples
                / self.config.trainer.batch_size
                + 1
            ),
            use_multiprocessing=True,
        )

        # we save the best model during training, don't need this
        # self.model.save_weights(
        #     os.path.join(os.getenv("VISIONENGINE_HOME"),
        #         self.config.callbacks.checkpoint_dir,
        #         '%s-{epoch:02d}-{loss:.2f}.hdf50' % self.config.exp.name)
        # )
