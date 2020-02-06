"""
Copyright (c) 2019 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from custom_objects.guppy_ornaments_custom_objects import CapacityIncrease
from base.base_trainer import BaseTrain

from keras.preprocessing.image import ImageDataGenerator


class GuppyOrnamentsTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(GuppyOrnamentsTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                      '%s-{epoch:02d}-{loss:.2f}.hdf5' %
                                      self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks
                .checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
                period=self.config.callbacks.period,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
                write_images=self.config.callbacks.tensorboard_write_images,
                # histogram_freq=0,  # this is a work around for ResampleLayer
            )
        )

        # use a callback to controll the bottleneck capacity
        self.callbacks.append(
            CapacityIncrease(
                max_capacity=self.config.model.max_capacity,
                max_epochs=self.config.model.max_epochs,
            )
        )

        # reduce the learning rate on plateau
        self.callbacks.append(
            ReduceLROnPlateau(monitor='loss',
                              factor=self.config.trainer.decay_factor,
                              mode='min',
                              patience=self.config.trainer.patience,
                              min_lr=1e-010)
        )

    def train(self):
        if self.config.model.use_tc_discriminator is True:
            raise NotImplementedError
            # if self.config.dataset.aug is True:
            #     shift = self.config.dataset.aug_shift
            #     rotation_range = self.config.dataset.aug_rotation
            #     self.datagen = ImageDataGenerator(
            #         width_shift_range=shift,
            #         height_shift_range=shift,
            #         horizontal_flip=True,
            #         vertical_flip=True,
            #         rotation_range=rotation_range
            #         )

            #     self.datagen.fit(self.data[0])

            #     history = self.model.fit_generator(
            #         self.datagen.flow(
            #             self.data[0],
            #             self.data[0],
            #             batch_size=self.config.trainer.batch_size
            #             ),
            #         epochs=self.config.trainer.num_epochs,
            #         verbose=self.config.trainer.verbose_training,
            #         shuffle=self.config.trainer.shuffle,
            #         steps_per_epoch=len(self.data[0])/self.config.trainer.batch_size,
            #         callbacks=self.callbacks,
            #     )
            # else:

        else:
            if self.config.dataset.aug is True:
                shift = self.config.dataset.aug_shift
                rotation_range = self.config.dataset.aug_rotation
                self.datagen = ImageDataGenerator(
                    width_shift_range=shift,
                    height_shift_range=shift,
                    horizontal_flip=True,
                    vertical_flip=True,
                    rotation_range=rotation_range
                    )

                self.datagen.fit(self.data[0])

                history = self.model.fit_generator(
                    self.datagen.flow(
                        self.data[0],
                        self.data[0],
                        batch_size=self.config.trainer.batch_size,
                        # save_to_dir=self.config.dataset.aug_dir,
                        ),
                    epochs=self.config.trainer.num_epochs,
                    verbose=self.config.trainer.verbose_training,
                    shuffle=self.config.trainer.shuffle,
                    steps_per_epoch=len(self.data[0])/self.config.trainer.batch_size,
                    use_multiprocessing=True,
                    callbacks=self.callbacks,
                )
            else:
                history = self.model.fit(
                    self.data[0], self.data[0],
                    epochs=self.config.trainer.num_epochs,
                    verbose=self.config.trainer.verbose_training,
                    batch_size=self.config.trainer.batch_size,
                    validation_split=self.config.trainer.validation_split,
                    shuffle=self.config.trainer.shuffle,
                    callbacks=self.callbacks,
                )
    
        for metric in self.config.callbacks.callback_metrics:
            self.loss.extend(history.history[metric])
