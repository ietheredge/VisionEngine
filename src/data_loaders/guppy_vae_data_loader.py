"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from base.base_data_loader import BaseDataLoader

from VisionEngine.datasets import guppies, generated_guppies
import numpy as np


class GuppyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(GuppyDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = \
            guppies.load_data()

        if self.config.data_loader.use_generated is True:
            (self.X_trainG, self.y_trainG), (self.X_testG, self.y_testG) = \
                generated_guppies.load_data()

            self.X_train = np.concatenate((self.X_train, self.X_trainG))
            self.y_train = np.concatenate((self.y_train, self.y_trainG))
            self.X_test = np.concatenate((self.X_test, self.X_testG))
            self.y_test = np.concatenate((self.y_test, self.y_testG))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
