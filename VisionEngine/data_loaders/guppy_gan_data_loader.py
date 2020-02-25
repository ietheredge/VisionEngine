"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from VisionEngine.base.base_data_loader import BaseDataLoader

from VisionEngine.data.datasets import guppies


class GuppiesDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(GuppiesDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (_, _) = \
            guppies.load_data()

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_train, self.y_train
