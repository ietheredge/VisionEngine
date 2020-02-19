"""
Copyright (c) 2020 R. Ian Etheredge All rights reserved.

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from base.base_data_loader import BaseDataLoader

from VisionEngine.datasets import butterflies


class ButterflyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ButterflyDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (_, _) = \
            butterflies.load_data()

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_train, self.y_train
