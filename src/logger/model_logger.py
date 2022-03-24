# -*- coding: utf-8 -*-
"""
ModelLogger class used for saving the progress
during the training phase, by storing errors, images, etc.

@author: Denis Tome'

"""
import os
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from utils.io import ensure_dir

__all__ = [
    'ModelLogger',
]


class ModelLogger:
    """
    Logger, used by BaseTrainer to save training history
    """

    def __init__(self, dir_path, training_name):
        self.dir_path = dir_path
        ensure_dir(self.dir_path)
        self.training_name = training_name

        self.train = SummaryWriter(
            os.path.join(self.dir_path, 'train'), comment=self.training_name)
        self.val = SummaryWriter(
            os.path.join(self.dir_path, 'val'), comment=self.training_name)

    def add_graph_definition(self, model):
        """Add graph

        Arguments:
            model {Model} -- model
        """

        dummy_input = Variable(torch.rand(1, 3, 224, 224))
        self.train.add_graph(model, dummy_input)

    def close_all(self):
        """Close"""
        self.train.close()
        self.val.close()
