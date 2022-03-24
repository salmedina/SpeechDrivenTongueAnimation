# -*- coding: utf-8 -*-
"""
@author: Salvador Medina
"""

import torch
import torch.nn as nn

__all__ = [
    'GRU'
]


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers,
                 dropout=0.2, bidirectional=False, embedding_dim=0):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.bidir_factor = 2 if bidirectional else 1

        self.feat_fc = None
        if embedding_dim > 0:
            self.feat_fc = nn.Linear(input_dim, embedding_dim)
        gru_input_dim = embedding_dim if embedding_dim > 0 else input_dim
        self.gru = nn.GRU(gru_input_dim,
                          hidden_dim,
                          n_layers,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim * self.bidir_factor, output_dim)

        self.params = dict(input_dim=self.input_dim,
                           hidden_dim=self.hidden_dim,
                           output_dim=self.output_dim,
                           embedding_dim=self.embedding_dim,
                           n_layers=self.n_layers,
                           dropout=self.dropout,
                           bidirectional=self.bidirectional)

    def forward(self, x, h):
        if self.feat_fc is not None:
            x = self.feat_fc(x)
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out))
        return out, h

    def infer(self, x):
        h0 = self.init_hidden(x.shape[0])
        out, _ = self.forward(x, h0)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers * self.bidir_factor,
                            batch_size,
                            self.hidden_dim).zero_()
        return hidden

    def save_model(self, save_path):
        torch.save({'model_params': self.params,
                    'model_state_dict': self.state_dict()
                    }, save_path)

    @staticmethod
    def load_model(checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=str(device))
        model = GRU(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model
