# -*- coding: utf-8 -*-
"""
ModelLogger class used for saving the progress
during the training phase, by storing errors, images, etc.

@author: Denis Tome'

"""
import os
from torch.utils.tensorboard import SummaryWriter


__all__ = [
    'ModelLogger',
]


class ModelLogger:
    """
    Logger used used to save training history
    """

    def __init__(self, dir_path, training_name):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)
        self.training_name = training_name

        self.train = SummaryWriter(
            os.path.join(self.dir_path, 'train'), comment=self.training_name)
        self.val = SummaryWriter(
            os.path.join(self.dir_path, 'val'), comment=self.training_name)

    def close_all(self):
        """Close both SummaryWriters"""
        self.train.close()
        self.val.close()
